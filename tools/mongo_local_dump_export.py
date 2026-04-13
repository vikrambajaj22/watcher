"""Export watcher DB to flat per-collection .bson files for small hosts (e.g. e2-micro).

By default, tmdb_metadata is pruned to TMDB_FIELDS below to save disk and transfer size.
Use MONGO_DUMP_PRUNE_TMDB=0 for full TMDB documents. See DEPLOYMENT.md Option B.
"""
import os
from pymongo import MongoClient
from bson import encode
from pathlib import Path

# ----- Config -----
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "watcher"
OUTPUT_DIR = Path("mongo_dumps")
OUTPUT_DIR.mkdir(exist_ok=True)

# Preserve ObjectIds by default so mongorestore round-trips match your local DB.
# Set MONGO_DUMP_STRIP_OBJECT_ID=1 to drop _id (smaller files; new ids on restore).
_STRIP_OBJECT_ID = os.environ.get("MONGO_DUMP_STRIP_OBJECT_ID", "0") == "1"
# Default pruning keeps exporter output small; set MONGO_DUMP_PRUNE_TMDB=0 for full tmdb_metadata docs.
_PRUNE_TMDB = os.environ.get("MONGO_DUMP_PRUNE_TMDB", "1") == "1"

# Columns to keep for tmdb_metadata
TMDB_FIELDS = [
    "id",
    "media_type",
    "title",
    "overview",
    "tagline",
    "release_date",
    "first_air_date",
    "poster_path",
    "backdrop_path",
    "name",
    "genres",
    "original_title",
    "original_name",
    "rewatch_engagement",
    "credits",
    "keywords",
    "popularity",
    # runtime fields
    "runtime",
    "episode_run_time",
    "episode_runtime",
]

# ----- Connect -----
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

def _prune_credits(credits: dict) -> dict:
    """
    Keep only the subfields required by:
      - _extract_actors: cast[].name, cast[].order
      - _extract_directors: crew[].name, crew[].job, crew[].department
    """
    if not isinstance(credits, dict):
        return {}

    cast = credits.get("cast") or []
    crew = credits.get("crew") or []

    pruned_cast = [
        {
            "name": c.get("name"),
            "order": c.get("order"),
        }
        for c in cast
        if isinstance(c, dict) and c.get("name")
    ]

    pruned_crew = [
        {
            "name": c.get("name"),
            "job": c.get("job"),
            "department": c.get("department"),
        }
        for c in crew
        if isinstance(c, dict)
        and c.get("name")
        and (c.get("job") == "Director" or c.get("department") == "Directing")
    ]

    return {
        "cast": pruned_cast,
        "crew": pruned_crew,
    }

print(
    "mongo_local_dump_export: "
    f"STRIP_OBJECT_ID={_STRIP_OBJECT_ID}, PRUNE_TMDB={_PRUNE_TMDB} "
    "(set MONGO_DUMP_STRIP_OBJECT_ID=1 / MONGO_DUMP_PRUNE_TMDB=0 to change)"
)

# ----- Iterate collections -----
for coll_name in db.list_collection_names():
    coll = db[coll_name]
    output_file = OUTPUT_DIR / f"{coll_name}.bson"
    count = 0
    print(f"Exporting {coll_name} -> {output_file} ...")

    with open(output_file, "wb") as f:
        cursor = coll.find({})
        for doc in cursor:
            doc_copy = doc.copy()
            if _STRIP_OBJECT_ID:
                doc_copy.pop("_id", None)

            # Field filtering for tmdb_metadata (optional; off = full documents)
            if coll_name == "tmdb_metadata" and _PRUNE_TMDB:
                doc_copy = {k: v for k, v in doc_copy.items() if k in TMDB_FIELDS}

                if "credits" in doc_copy:
                    doc_copy["credits"] = _prune_credits(doc_copy["credits"])

            f.write(encode(doc_copy))
            count += 1

    print(f"Exported {count} documents from {coll_name} to {output_file}")