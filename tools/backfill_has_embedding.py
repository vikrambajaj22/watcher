"""Backfill `has_embedding` field on tmdb_metadata documents in batches.

This will update existing documents with has_embedding=True if they have an embedding,
or has_embedding=False if they do not. It operates in batches to avoid overloading the DB.

Usage:
    python -m tools.backfill_has_embedding --batch 1000 --dry-run
    python -m tools.backfill_has_embedding --batch 1000
"""
import argparse
from app.db import tmdb_metadata_collection

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=1000)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

batch_size = args.batch

def iter_batches(batch_size=1000):
    cursor = tmdb_metadata_collection.find({}, {"_id": 1, "embedding": 1, "embedding_meta": 1})
    batch = []
    for doc in cursor:
        batch.append(doc)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

updated = 0
checked = 0
for b in iter_batches(batch_size):
    ops = []
    for d in b:
        checked += 1
        has = bool(d.get("embedding") or d.get("embedding_meta"))
        if args.dry_run:
            if not has:
                print("Would set has_embedding=False for", d.get("_id"))
            else:
                print("Would set has_embedding=True for", d.get("_id"))
        else:
            res = tmdb_metadata_collection.update_one({"_id": d.get("_id")}, {"$set": {"has_embedding": has}})
            if res.modified_count:
                updated += 1
    if not args.dry_run:
        print(f"Processed batch: checked={checked} updated={updated}")

print(f"Done. checked={checked}, updated={updated}")

