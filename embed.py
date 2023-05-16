import os
from functools import partial
from multiprocessing.pool import ThreadPool

import jax
import numpy as np
import polars as pl
import rich.progress as rp
from transformers import FlaxViTModel

from train import dtype_half, ensure_img, latest_ckpt, read_image


def batches(posts: pl.DataFrame, tags: pl.DataFrame, size: int):
    with ThreadPool() as pool:
        for chunk in posts.iter_slices(size):
            chunk = (
                chunk.with_columns(pl.col("tag_string").str.split(" ").alias("tags"))
                .drop("tag_string")
                .explode("tags")
                .join(tags, left_on="tags", right_on="name")
                .sort("post_count")
                .groupby("id")
                .first(30)
                .select(chunk.columns)
            )
            images, image_md5s = zip(
                *pool.imap_unordered(read_image, tags.select("md5", "file_ext"))
            )
            sorter = np.argsort(image_md5s)
            images = np.stack(images, axis=0)[sorter]
            if images.shape[0] < size:
                padding = [(0, 0)] * images.ndim
                padding[0] = (0, size - images.shape[0])
                images = np.pad(images, padding)
            yield chunk, {"pixel_values": images.swapaxes(-1, -3)}


def main(batch_size=1024):
    backbone = FlaxViTModel.from_pretrained(latest_ckpt(), dtype=dtype_half)

    def infer(**inputs):
        output = backbone(**inputs, train=False)
        return output

    infer = jax.jit(partial(backbone, train=False))
    posts = ensure_img(
        pl.scan_csv("posts.csv")
        .select("id", "md5", "file_ext", "tag_string")
        .sort("id")
        .unique("id")
    ).collect()
    tags = (
        pl.scan_csv("tags.csv")
        .filter(pl.col("post_count") > 1)
        .select("name", "post_count")
    )
    os.makedirs("embeds", exist_ok=True)
    table = np.memmap(
        "embeds/latent.npy",
        shape=[len(posts), backbone.module.config.hidden_size],
        mode="w+",
        dtype=np.float16,
    )
    total = np.ceil(len(posts) / batch_size).astype(int)
    for i, (chunk, batch) in rp.track(
        enumerate(batches(posts, tags, batch_size)),
        total=total,
        description="compute embeddings...",
    ):
        embed = jax.device_get(infer(**batch)).astype(table.dtype)
        table[i : i + len(chunk)] = embed[:]
    np.save("embeds/id.npy", posts["id"])


if __name__ == "__main__":
    main()
