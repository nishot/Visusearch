import argparse
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

from src.config import settings
from src.models.image_encoder import ImageEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reset", action="store_true", help="Discard an existing checkpoint and start over.")
    args = parser.parse_args()
    settings.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    partial_embeddings = settings.embeddings_path.with_suffix(".partial.npy")
    partial_metadata = settings.embedding_metadata_path.with_suffix(".partial.csv")
    progress_path = settings.embeddings_path.with_suffix(".progress.json")
    articles = pd.read_csv(settings.articles_path, dtype={"article_id": str})
    records = []
    for row in articles.itertuples(index=False):
        article_id = str(row.article_id).zfill(10)
        path = settings.images_root / article_id[:3] / f"{article_id}.jpg"
        if path.exists():
            records.append((article_id, str(path)))
        if args.limit and len(records) >= args.limit:
            break
    if args.reset:
        for path in (partial_embeddings, partial_metadata, progress_path):
            path.unlink(missing_ok=True)

    progress = {}
    if progress_path.exists() and partial_embeddings.exists() and partial_metadata.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("record_count") != len(records):
            raise RuntimeError("The checkpoint does not match this dataset or --limit. Use --reset to restart.")

    encoder = ImageEncoder(settings.model_name, settings.resolved_device())
    start = int(progress.get("next_record", 0))
    processed = int(progress.get("processed", 0))
    embedding_store = None
    if partial_embeddings.exists():
        embedding_store = np.load(partial_embeddings, mmap_mode="r+")

    for start in range(start, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        images, valid = [], []
        for article_id, path in batch:
            try:
                images.append(Image.open(path).convert("RGB"))
                valid.append((article_id, path))
            except OSError:
                continue
        if valid:
            vectors = encoder.encode_images(images).numpy().astype("float32")
            if embedding_store is None:
                embedding_store = np.lib.format.open_memmap(
                    partial_embeddings, mode="w+", dtype="float32", shape=(len(records), vectors.shape[1])
                )
            embedding_store[processed:processed + len(valid)] = vectors
            embedding_store.flush()
            metadata = pd.DataFrame(valid, columns=["article_id", "image_path"])
            metadata.to_csv(partial_metadata, mode="a", header=not partial_metadata.exists(), index=False)
            processed += len(valid)
        progress_path.write_text(json.dumps({"record_count": len(records), "next_record": start + len(batch), "processed": processed}))
        print(f"Embedded {min(start + len(batch), len(records))}/{len(records)}", flush=True)

    if embedding_store is None:
        raise RuntimeError("No readable product images were found.")
    embedding_store.flush()
    del embedding_store
    os.replace(partial_embeddings, settings.embeddings_path)
    os.replace(partial_metadata, settings.embedding_metadata_path)
    progress_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
