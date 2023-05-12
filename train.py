import itertools as it
import os
import os.path as osp
import sys
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import NamedTuple

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import optax
import polars as pl
import rax
import rich.progress as rp
from transformers import CLIPProcessor, FlaxCLIPModel

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
image_dir = "posts"


def ensure_img(posts: pl.DataFrame) -> pl.DataFrame:
    stems = os.listdir(image_dir)
    return (
        posts.sort("id")
        .with_columns(pl.lit("webp").alias("file_ext"))
        .filter((pl.col("id").cast("str") + "." + pl.col("file_ext")).is_in(stems))
    )


epochs = 1.0
batch_size = 128
# posts = (
#     ensure_img(pl.scan_csv("posts.csv", low_memory=False))
#     .collect()
# )
files = os.listdir("posts")
posts = (
    pl.scan_csv(
        "data.csv",
    )
    .with_columns(
        pl.col("file_name").str.split(".").arr.first().cast(int).alias("id"),
        pl.col("file_name").str.split(".").arr.last().alias("file_ext"),
        pl.col("text_caption").alias("tag_string"),
    )
    .select("id", "file_ext", "tag_string")
).collect()
train_steps = np.ceil(epochs * len(posts) / batch_size).astype(int)
rng = jax.random.PRNGKey(42)
rng, key = jax.random.split(rng)


def unif_batch(
    size: int,
    seed: int,
    posts: pl.DataFrame,
) -> tuple[dict, jax.Array]:
    posts = ensure_img(posts)
    rng = np.random.default_rng(seed)

    def jaccard(a, b):
        n = len(a.intersection(b))
        d = len(a) + len(b) - n
        return n / d

    dims = tuple(map(processor.image_processor.crop_size.get, ("width", "height")))

    def read(post_id, file_ext):
        stem = f"{post_id}.{file_ext}"
        path = osp.join(image_dir, stem)
        if osp.exists(path):
            img = cv2.imread(path)[..., ::-1]
            img = cv2.resize(img, dims)
            return img, post_id
        else:
            return np.zeros([*size, 3]), post_id

    def tags(chunk):
        return (
            chunk.select("id", pl.col("tag_string").str.split(" ").alias("tags"))
            .explode("tags")
            .sample(
                30 * len(chunk),
                shuffle=True,
                seed=rng.bit_generator.random_raw(),
            )
            .groupby("id")
            .all()
            .sort("id")
            .select("tags")
            .to_series()
            .arr.join(" ")
            .to_list()
        )

    table = posts.select("id", "file_ext", "tag_string")
    with ThreadPool() as pool:
        for chunk in (
            table.sample(size, seed=seed).sort("id")
            for seed in iter(rng.bit_generator.random_raw, None)
        ):
            pairs = pool.imap_unordered(
                lambda args: read(*args),
                chunk.select("id", "file_ext").iter_rows(),
            )
            images, image_ids = zip(*pairs)
            select = np.argsort(image_ids)
            images = np.stack(images)[select]
            tag_strs = tags(chunk)
            tag_sets = [set(s.split()) for s in tags(chunk)]
            labels = [jaccard(a, b) for a, b in it.product(tag_sets, tag_sets)]
            labels = np.array(labels).reshape(len(images), len(images))
            inputs = {
                "pixel_values": images.swapaxes(-1, -3),
                **processor.tokenizer(
                    tag_strs,
                    return_tensors="np",
                    padding=True,
                    truncation=True,
                ),
            }
            yield inputs, labels


class EMA(NamedTuple):
    "https://blog.fugue88.ws/archives/2017-01/The-correct-way-to-start-an-Exponential-Moving-Average-EMA"
    r: float
    s: float
    d: float

    @classmethod
    def init(cls, r):
        return cls(r=r, s=0, d=1)

    def update(self, x):
        s = self.r * self.s + (1 - self.r) * x
        d = self.r * self.d
        s /= 1 - d
        return self._replace(r=self.r, s=s, d=d)


class TrainState(NamedTuple):
    params: optax.Params
    opt_st: optax.OptState
    loss: EMA


def optimizer() -> optax.GradientTransformation:
    lsched = optax.cosine_onecycle_schedule(train_steps, 1e-4)
    msched = lambda step: 0.95 - optax.cosine_onecycle_schedule(train_steps, 0.1)(step)
    return optax.inject_hyperparams(optax.lion)(lsched, msched)


backbone = FlaxCLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32", dtype=jnp.float16
)


def train_init() -> TrainState:
    params = backbone.params
    opt_st = optimizer().init(params)
    loss = EMA.init(0.9)
    return TrainState(params, opt_st, loss)


def objective(params, rng, inputs, labels):
    offset = np.array(processor.image_processor.image_mean)[:, None, None]
    stddev = np.array(processor.image_processor.image_std)[:, None, None]
    inputs = {
        **inputs,
        "pixel_values": (inputs["pixel_values"] / 255.0 - offset) / stddev,
    }
    output = backbone(**inputs, params=params, dropout_rng=rng, train=True)
    weights = jnp.any(inputs["pixel_values"] != 0, axis=(-1, -2, -3))
    logits = output.logits_per_image, output.logits_per_text
    losses = map(
        lambda scores: rax.pairwise_logistic_loss(
            scores,
            labels,
            weights=weights,
            lambdaweight_fn=rax.labeldiff_lambdaweight,
        )
        / len(logits),
        logits,
    )
    return sum(losses)


@partial(jax.jit, donate_argnums=0)
def train_step(tstate: TrainState, rng, inputs, labels) -> TrainState:
    loss, grad = jax.value_and_grad(objective)(tstate.params, rng, inputs, labels)
    ascent_stride = 1.0 / optax.global_norm(grad)
    params = jax.tree_util.tree_map(
        lambda w, dw: w + w**2 * dw * ascent_stride, tstate.params, grad
    )
    grad = jax.grad(objective)(params, rng, inputs, labels)
    grad, opt_st = optimizer().update(grad, tstate.opt_st, tstate.params)
    params = optax.apply_updates(tstate.params, grad)
    loss = tstate.loss.update(loss)
    return TrainState(params, opt_st, loss)


batches = unif_batch(batch_size, jax.device_get(key)[0], posts)
tstate = train_init()
with rp.Progress(
    "loss: {task.fields[loss]:.3g}",
    *rp.Progress.get_default_columns()[:-2],
    rp.MofNCompleteColumn(),
    rp.TimeElapsedColumn(),
    console=rp.Console(file=sys.stderr),
) as pbar:
    task = pbar.add_task(
        "training...",
        loss=float("nan"),
        total=train_steps,
        start=False,
    )
    for step, (inputs, labels) in enumerate(it.islice(batches, train_steps)):
        rng, key = jax.random.split(rng)
        pbar.start_task(task)
        tstate = train_step(tstate, key, inputs, labels)
        pbar.update(task, advance=1, loss=jax.device_get(tstate.loss.s))

backbone.save_pretrained("e6clip", params=tstate.params)
