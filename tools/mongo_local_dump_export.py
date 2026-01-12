import os
from pymongo import MongoClient
from bson import encode
from pathlib import Path

# ----- Config -----
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "watcher"
OUTPUT_DIR = Path("mongo_dumps")
OUTPUT_DIR.mkdir(exist_ok=True)

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
    "popularity"
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
            doc_copy.pop("_id", None)  # optional: remove ObjectId to reduce size

            # Field filtering for tmdb_metadata
            if coll_name == "tmdb_metadata":
                doc_copy = {k: v for k, v in doc_copy.items() if k in TMDB_FIELDS}

                if "credits" in doc_copy:
                    doc_copy["credits"] = _prune_credits(doc_copy["credits"])

            f.write(encode(doc_copy))
            count += 1

    print(f"Exported {count} documents from {coll_name} to {output_file}")