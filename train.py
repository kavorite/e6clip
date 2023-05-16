import itertools as it
import os
import os.path as osp
import sys
from functools import partial
from glob import glob
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
from transformers import FlaxViTModel

jax.config.update("jax_threefry_partitionable", True)


base_ckpt = "google/vit-base-patch16-224"
image_dir = "posts"
ckpts_dir = "ckpts"
image_dim = 224
os.makedirs(ckpts_dir, exist_ok=True)


def latest_ckpt():
    ckpt_names = glob(osp.join(ckpts_dir, "e6clip-e*"))
    if ckpt_names:
        ckpt = max(ckpt_names, key=lambda f: os.stat(f).st_mtime)
    else:
        ckpt = base_ckpt
    return ckpt


def ensure_img(posts: pl.DataFrame) -> pl.DataFrame:
    stems = os.listdir(image_dir)
    return posts.filter((pl.col("md5") + ".webp").is_in(stems))


def read_image(md5):
    stem = f"{md5}.webp"
    path = osp.join(image_dir, stem)
    dims = [image_dim, image_dim, 3]
    if not osp.exists(path):
        return np.zeros(dims), md5
    img = cv2.imread(path)
    if img is None:
        return np.zeros(dims), md5
    img = cv2.resize(img, dims[:-1])
    return img, md5


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

    def tags(chunk):
        return (
            chunk.select("md5", pl.col("tag_string").str.split(" ").alias("tags"))
            .explode("tags")
            .sample(
                30 * len(chunk),
                shuffle=True,
                with_replacement=True,
                seed=rng.bit_generator.random_raw(),
            )
            .unique(["md5", "tags"])
            .groupby("md5")
            .all()
            .select("md5", pl.col("tags").arr.join(" "))
            .sort("md5")
            .drop("md5")
            .to_series()
            .to_list()
        )

    table = posts.select("md5", "tag_string")
    with ThreadPool() as pool:
        for chunk in (
            table.sample(size, seed=seed).sort("md5")
            for seed in iter(rng.bit_generator.random_raw, None)
        ):
            pairs = pool.imap_unordered(read_image, chunk.select("md5").to_series())
            images, image_md5s = zip(*pairs)
            sorter = np.argsort(image_md5s)
            images = np.stack(images)[sorter]
            tag_sets = [set(s.split()) for s in tags(chunk)]
            labels = [jaccard(a, b) for a, b in it.product(tag_sets, tag_sets)]
            labels = np.array(labels).reshape(len(images), len(images))
            inputs = {"pixel_values": images}
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


dtype_half = jnp.bfloat16 if jax.devices()[0].platform == "tpu" else jnp.float16


def main():
    epochs = 1.0
    sharding = jax.sharding.PositionalSharding(np.array(jax.devices()).reshape(-1, 1))
    batch_size = 512 * jax.device_count()
    posts = ensure_img(
        pl.scan_csv("posts.csv", low_memory=True).select("md5", "tag_string")
    ).collect()
    train_steps = np.ceil(epochs * len(posts) / batch_size).astype(int)
    epoch_steps = np.ceil(len(posts) / batch_size).astype(int)

    rng = jax.random.PRNGKey(42)
    rng, key = jax.random.split(rng)

    backbone = FlaxViTModel.from_pretrained(latest_ckpt(), dtype=dtype_half)
    backbone.module.config.image_size = image_dim

    def optimizer() -> optax.GradientTransformation:
        peak_lr = 3.75e-4 * jax.device_count()
        lsched = optax.cosine_onecycle_schedule(
            peak_lr,
            train_steps,
        )
        msched = lambda step: 0.99 - 0.09 * lsched(step) / peak_lr
        xforms = optax.chain(
            optax.adaptive_grad_clip(1e-3),
            optax.inject_hyperparams(optax.lion)(
                lsched,
                msched,
                weight_decay=0.3,
            ),
            optax.zero_nans(),
        )
        return optax.lookahead(optax.flatten(xforms), sync_period=6, slow_step_size=0.5)

    def train_init() -> TrainState:
        params = {**backbone.params, "logit_scale": 0.2 / 768**0.5}
        params = jax.device_put(params, sharding.replicate())
        params = optax.LookaheadParams(
            fast=params,
            slow=jax.tree_util.tree_map(
                lambda a: a.astype(dtype_half),
                params,
            ),
        )
        opt_st = optimizer().init(params)
        loss = EMA.init(0.9)
        return TrainState(params, opt_st, loss)

    def clip_error(params, rng, inputs, labels):
        dropout_rng, augment_rng = jax.random.split(rng)
        images = augment(augment_rng, inputs["pixel_values"])
        images = (images - 127.5) / 255.0
        images = images.swapaxes(-1, -3)
        inputs = {**inputs, "pixel_values": images}
        output = backbone(
            **inputs,
            params=params,
            dropout_rng=dropout_rng,
            train=True,
        )
        latent = output.pooler_output
        scores = params["logit_scale"] * latent @ latent.swapaxes(-1, -2)
        return rax.pairwise_logistic_loss(
            scores,
            labels,
            weights=jnp.all(images != 0, axis=(-1, -2, -3))[..., None, :],
            lambdaweight_fn=rax.labeldiff_lambdaweight,
        )

    @partial(jax.jit, donate_argnums=0)
    def train_step(tstate: TrainState, rng, inputs, labels) -> TrainState:
        loss, grad = jax.value_and_grad(clip_error)(
            tstate.params.fast, rng, inputs, labels
        )
        ascent_stride = 0.2 / optax.global_norm(grad)
        params = jax.tree_util.tree_map(
            lambda w, dw: w + w**2 * dw * ascent_stride, tstate.params.fast, grad
        )
        grad = jax.grad(clip_error)(params, rng, inputs, labels)
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
            images = inputs.pop("pixel_values")
            inputs, labels = jax.device_put((inputs, labels), sharding)
            images = jax.device_put(
                images, sharding.reshape((-1,) + (1,) * (images.ndim - 1))
            )
            inputs["pixel_values"] = images
            tstate = train_step(tstate, key, inputs, labels)
            pbar.start_task(task)
            pbar.update(task, advance=1, loss=jax.device_get(tstate.loss.s))
            final_step = step == train_steps - 1
            if (step > 0 and step % epoch_steps == 0) or final_step:
                epoch = (step + 1) // epoch_steps
                params = jax.device_get(tstate.params.slow)
                params["pooler"]["dense"]["kernel"] = params["pooler"]["dense"][
                    "kernel"
                ] * params.pop("logit_scale")
                backbone.save_pretrained(
                    osp.join(".", ckpts_dir, f"e6clip-e{epoch}"),
                    params=tstate.params.slow,
                    push_to_hub=final_step,
                    repo_id="e6clip",
                )


if __name__ == "__main__":
    main()
