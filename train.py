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
                "pixel_values": images,
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
    params: optax.LookaheadParams
    opt_st: optax.OptState
    loss: EMA


def standardize(images):
    offset = np.array(processor.image_processor.image_mean)
    stddev = np.array(processor.image_processor.image_std)
    images = (images / 255.0 - offset) / stddev
    return images


def augment(rng, images):
    """Random horizontal flip, + HSV transforms courtesy of
    https://beesbuzz.biz/code/16-hsv-color-transforms
    """
    keys = jax.random.split(rng, 3)
    flip_mask = jax.random.bernoulli(keys[0], shape=images.shape[:-3] + (1,) * 3)
    images = jnp.where(flip_mask, jnp.flip(images, axis=-2), images)
    to_yiq = jnp.array(
        [[0.299, 0.587, 0.114], [0.596, -0.274, -0.321], [0.211, -0.523, 0.311]]
    )
    to_rgb = jnp.linalg.inv(to_yiq)
    thetas = jax.random.truncated_normal(
        rng, -0.5 * jnp.pi, 0.5 * jnp.pi, shape=images.shape[:-3]
    )
    leader = tuple(range(images.ndim - 3))

    def rot_proj(theta):
        c = jnp.cos(theta)
        s = jnp.sin(theta)
        return jnp.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    if len(leader) == 0:
        rotate = rot_proj(thetas)
    else:
        rotate = jax.vmap(rot_proj, in_axes=leader)(thetas)
    images = images @ to_yiq
    images = jnp.einsum("... h w d, ... d d -> ... h w d", images, rotate)
    images = images.at[..., 1:].mul(jax.random.truncated_normal(keys[1], 0.5, 1.5))
    images *= jax.random.truncated_normal(keys[2], 0.5, 1.5)
    images = images @ to_rgb
    return images


def main():
    epochs = 8.0
    sharding = jax.sharding.PositionalSharding(np.array(jax.devices()).reshape(-1, 1))
    n_device = sharding.shape[-1]
    batch_size = 128 * n_device
    posts = ensure_img(pl.scan_csv("posts.csv")).collect()
    train_steps = np.ceil(epochs * len(posts) / batch_size).astype(int)
    epoch_steps = np.ceil(len(posts) / batch_size).astype(int)

    rng = jax.random.PRNGKey(42)
    rng, key = jax.random.split(rng)

    half_dtype = jnp.bfloat16 if jax.devices()[0].platform == "tpu" else jnp.float16
    backbone = FlaxCLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", dtype=half_dtype
    )

    def optimizer() -> optax.GradientTransformation:
        lsched = optax.cosine_decay_schedule(1e-4 * n_device, train_steps)
        msched = lambda step: 0.9 + 0.09 * step / train_steps
        xforms = optax.chain(
            optax.adaptive_grad_clip(1e-3),
            optax.inject_hyperparams(optax.lion)(
                lsched,
                msched,
                weight_decay=0.3,
            ),
            optax.zero_nans(),
        )
        return optax.lookahead(xforms, sync_period=8, slow_step_size=0.9)

    def train_init() -> TrainState:
        params = jax.device_put(backbone.params, sharding.replicate())
        params = optax.LookaheadParams(
            fast=params,
            slow=jax.tree_util.tree_map(lambda a: a.astype(half_dtype), params),
        )
        opt_st = optimizer().init(params)
        loss = EMA.init(0.9)
        return TrainState(params, opt_st, loss)

    def objective(params, rng, inputs, labels):
        dropout_rng, augment_rng = jax.random.split(rng)
        images = augment(augment_rng, inputs["pixel_values"])
        images = standardize(images)
        images = images.swapaxes(-1, -3)
        inputs = {**inputs, "pixel_values": images}
        output = backbone(**inputs, params=params, dropout_rng=dropout_rng, train=True)
        losses = jax.tree_util.tree_map(
            lambda scores: rax.pairwise_logistic_loss(
                scores, labels, lambdaweight_fn=rax.labeldiff_lambdaweight
            ),
            (output.logits_per_image, output.logits_per_text),
        )
        return jnp.stack(losses).mean()

    @partial(jax.jit, donate_argnums=0)
    def train_step(tstate: TrainState, rng, inputs, labels) -> TrainState:
        loss, grad = jax.value_and_grad(objective)(
            tstate.params.fast, rng, inputs, labels
        )
        ascent_stride = 0.2 / optax.global_norm(grad)
        params = jax.tree_util.tree_map(
            lambda w, dw: w + w**2 * dw * ascent_stride, tstate.params.fast, grad
        )
        grad = jax.grad(objective)(params, rng, inputs, labels)
        grad, opt_st = optimizer().update(grad, tstate.opt_st, tstate.params)
        params = optax.apply_updates(tstate.params, grad)
        loss = tstate.loss.update(loss)
        return TrainState(params, opt_st, loss)

    batches = unif_batch(batch_size * n_device, jax.device_get(key)[0], posts)
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
            rng, prngs = jax.random.split(rng, 1 + n_device)
            (inputs, labels), prngs = jax.device_put(
                ((inputs, labels), prngs), sharding
            )
            tstate = train_step(tstate, prngs, inputs, labels)
            pbar.start_task(task)
            pbar.update(task, advance=1, loss=jax.device_get(tstate.loss.s))
            if step > 0 and step % epoch_steps == 0:
                epoch = step // epoch_steps
                backbone.save_pretrained(f"e6clip-e{epoch}", params=tstate.params.slow)


if __name__ == "__main__":
    main()
