"""Migration helper: rebuild FAISS index from TMDB metadata (compute embeddings), write sidecars,
and optionally remove embedding-related fields from Mongo.

Usage:
    python -m tools.migrate_embeddings_to_faiss --dim 384 --batch 256 --dry-run
    python -m tools.migrate_embeddings_to_faiss --dim 384 --batch 256 --commit
    python -m tools.migrate_embeddings_to_faiss --device mps

Flags:
  --dim: embedding dim to request (passed to builder)
  --batch: batch size used when computing embeddings (passed to FAISS builder)
  --dry-run: compute and write sidecars but do NOT delete fields from Mongo
  --commit: after successful rebuild, remove `embedding`, `embedding_meta`, and `has_embedding` fields from tmdb_metadata
  --preview: show counts of docs with embedding fields before deletion
  --device: set embedding device for this run (overrides EMBED_DEVICE env var)
"""

import argparse
import sys
import time
import os
from app.utils.logger import get_logger
from app.db import tmdb_metadata_collection

logger = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--dim", type=int, default=384)
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--commit", action="store_true")
parser.add_argument("--preview", action="store_true")
parser.add_argument("--device", type=str, choices=["cpu", "mps", "cuda"], help="Set embedding device for this run (overrides EMBED_DEVICE env)")
args = parser.parse_args()

# If user specified a device on the CLI, set the EMBED_DEVICE env var before importing modules that may use it.
if args.device:
    os.environ["EMBED_DEVICE"] = args.device
    print(f"Using embedding device from CLI: {args.device}")

from app.faiss_index import build_faiss_index_from_mongo_embeddings, load_sidecars

print("Starting FAISS rebuild (this will compute embeddings for all metadata docs).")
start = time.time()
idx = build_faiss_index_from_mongo_embeddings(args.dim, batch_size=args.batch)
elapsed = time.time() - start
if idx is None:
    print("FAISS rebuild failed or produced no index.")
    sys.exit(2)
print(f"FAISS rebuild completed in {elapsed:.1f}s")

# verify sidecars
load_sidecars()
from app.faiss_index import LABELS_FILE, VECS_FILE

if os.path.exists(LABELS_FILE) and os.path.exists(VECS_FILE):
    import numpy as np

    labels = np.load(LABELS_FILE)
    vecs = np.load(VECS_FILE)
    print(f"Sidecars: labels={labels.shape}, vecs={vecs.shape}")
else:
    print("Sidecars missing after rebuild; aborting migration")
    sys.exit(3)

# preview or commit removal of fields
count_with_embedding = tmdb_metadata_collection.count_documents(
    {
        "$or": [
            {"embedding": {"$exists": True}},
            {"embedding_meta": {"$exists": True}},
            {"has_embedding": {"$exists": True}},
        ]
    }
)
print(
    f"Documents in tmdb_metadata with embedding-related fields: {count_with_embedding}"
)

if args.preview or args.dry_run:
    print("Dry run/preview mode; not deleting fields from DB.")
    sys.exit(0)

if args.commit:
    # perform batched removal
    BATCH = 1000
    cursor = tmdb_metadata_collection.find(
        {
            "$or": [
                {"embedding": {"$exists": True}},
                {"embedding_meta": {"$exists": True}},
                {"has_embedding": {"$exists": True}},
            ]
        },
        {"_id": 1},
    )
    to_delete = []
    cnt = 0
    for doc in cursor:
        _id = doc.get("_id")
        to_delete.append(_id)
        if len(to_delete) >= BATCH:
            res = tmdb_metadata_collection.update_many(
                {"_id": {"$in": to_delete}},
                {
                    "$unset": {
                        "embedding": "",
                        "embedding_meta": "",
                        "has_embedding": "",
                    }
                },
            )
            cnt += len(to_delete)
            print(f"Processed batch, removed fields from {len(to_delete)} docs")
            to_delete = []
    if to_delete:
        res = tmdb_metadata_collection.update_many(
            {"_id": {"$in": to_delete}},
            {"$unset": {"embedding": "", "embedding_meta": "", "has_embedding": ""}},
        )
        cnt += len(to_delete)
        print(f"Processed final batch, removed fields from {len(to_delete)} docs")
    print(f"Removal complete. Processed {cnt} documents.")
else:
    print("No --commit provided. Exiting without modifying DB.")
    sys.exit(0)
