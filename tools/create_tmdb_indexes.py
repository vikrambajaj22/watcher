"""Create recommended indexes for tmdb_metadata collection.

Run this once (during low traffic). It only creates indexes (no document updates).

These indexes optimize the common query patterns in tmdb_sync.py:
- Finding existing IDs (by id + media_type)
- Finding docs that need embeddings (by has_embedding flag)

Usage:
    python -m tools.create_tmdb_indexes
"""

from app.db import tmdb_metadata_collection
import pprint

print("Creating index: {id:1, media_type:1} (background)")
print("  -> Optimizes: lookups by ID and media type")
name1 = tmdb_metadata_collection.create_index(
    [("id", 1), ("media_type", 1)], background=True
)
print("Created index:", name1)

print("\nCreating index: {has_embedding:1} (background, sparse)")
print("  -> Optimizes: finding docs that need embeddings")
name2 = tmdb_metadata_collection.create_index(
    [("has_embedding", 1)],
    background=True,
    sparse=True,  # Only index docs where has_embedding exists
)
print("Created index:", name2)

print("\nCreating compound index: {id:1, media_type:1, has_embedding:1} (background)")
print(
    "  -> Optimizes: finding existing docs needing embeddings in _fetch_docs_needing_embedding()"
)
print("  -> Query: {id: {$in: [...]}, media_type: 'movie', has_embedding: {$ne: true}}")
print(
    "  -> Note: id + media_type together form the unique key (same ID can exist for movie/tv)"
)
name3 = tmdb_metadata_collection.create_index(
    [("id", 1), ("media_type", 1), ("has_embedding", 1)],
    background=True,
)
print("Created index:", name3)

print("\n" + "=" * 70)
print("CURRENT INDEXES:")
print("=" * 70)
pprint.pprint(tmdb_metadata_collection.index_information())

print("\n" + "=" * 70)
print("INDEX SUMMARY")
print("=" * 70)
print(f"✓ {name1}: Lookups by (id, media_type)")
print(f"✓ {name2}: Finding docs without embeddings (has_embedding)")
print(f"✓ {name3}: Combined query for existing docs needing embeddings")
print("=" * 70)
