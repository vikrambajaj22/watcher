"""LLM orchestration utilities and MCP handler in a single module.

This file centralizes the production mcp handler (resolve_query_vector, call_mcp)
and the model orchestration (call_model_with_mcp_function). Combining them keeps
the call flow colocated and reduces indirection.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np

from app.db import tmdb_metadata_collection
from app.embeddings import embed_text
from app.vector_store import query
from app.schemas.api import MCPPayload
from app.utils.logger import get_logger
from app.utils.openai_client import get_openai_client
from app.utils.prompt_registry import PromptRegistry

logger = get_logger(__name__)


def load_tool(tool_name: str) -> Dict[str, Any] | None:
    """Load tool schema from JSON file with name <tool_name>.json in tools/."""
    schema_path = os.path.join(
        os.path.dirname(__file__), "..", "tools", tool_name + ".json"
    )
    try:
        with open(schema_path, "r") as f:
            mcp_function_schema = json.load(f)
    except Exception:
        mcp_function_schema = None
    return mcp_function_schema


def resolve_query_vector(payload: MCPPayload) -> np.ndarray:
    """Resolve a numeric query vector from a validated payload.

    Raises ValueError on missing item or embedding.
    """
    if payload.tmdb_id is not None:
        doc = tmdb_metadata_collection.find_one({"id": payload.tmdb_id}, {"_id": 0})
        if not doc:
            raise ValueError(f"tmdb_id {payload.tmdb_id} not found")
        emb = doc.get("embedding")
        if emb is None:
            raise ValueError(f"embedding missing for tmdb_id {payload.tmdb_id}")
        return np.array(emb, dtype=np.float32)

    if payload.text is not None:
        # enrich free-text queries with an LLM before embedding so the vector
        # captures additional structured signals (genres, actors, directors, etc.).
        try:
            enriched = enrich_text_for_embedding(payload.text)
            logger.info("Enriched input text '%s': %s", payload.text, enriched)
        except Exception as e:
            logger.exception("Text enrichment failed, falling back to original text: %s", repr(e), exc_info=True)
            enriched = payload.text
        return embed_text([enriched])[0]

    if payload.vector is not None:
        return np.array(payload.vector, dtype=np.float32)

    raise ValueError("invalid payload; provide tmdb_id, text, or vector")


def enrich_text_for_embedding(text: str, model: str = "gpt-4.1-nano", max_tokens: int = 150) -> str:
    """Use an LLM to enrich a free-text query with likely genres, actors, and other
    relevant descriptors before embedding.

    The function returns a short, enhanced description (string). On failure, it
    raises an exception so callers can fall back if desired.
    """
    client = get_openai_client()

    registry = PromptRegistry("app/prompts/embedding")
    template = registry.load_prompt_template("enrich_for_embedding", 1)
    system_prompt = template.render(user_query=text)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt}
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )

    choice = resp.choices[0]
    msg = choice.message
    content = msg.content
    if not content:
        raise ValueError("LLM returned empty enrichment")
    return content.strip()


def call_mcp_knn(payload: MCPPayload) -> Dict[str, Any]:
    """Execute the KNN query for the given validated payload and return structured results.

    Returns a dict with a `results` key containing list of {id, title, media_type, score}.
    """
    k = int(payload.k)
    qvec = resolve_query_vector(payload)

    # get the input ID to exclude it from results (if searching by tmdb_id)
    exclude_id = payload.tmdb_id if payload.tmdb_id is not None else None

    # query with k+1 to account for excluding the input item
    query_k = k + 1 if exclude_id is not None else k
    vs_res = query(qvec, query_k)
    logger.info("Found %s results", len(vs_res))
    ids = [int(r[0]) for r in vs_res]
    docs = list(tmdb_metadata_collection.find({"id": {"$in": ids}}, {"_id": 0}))
    docs_by_id = {d.get("id"): d for d in docs}
    logger.info("Found %s matching documents in tmdb_metadata", len(docs_by_id))
    # optional media_type filtering
    requested_media = None
    if getattr(payload, "media_type", None):
        requested_media = str(payload.media_type).lower()
        if requested_media not in {"movie", "tv", "all"}:
            raise ValueError("media_type must be one of: 'movie', 'tv', 'all'")

    results: List[Dict[str, Any]] = []
    for tid, score in vs_res:
        if exclude_id is not None and int(tid) == int(exclude_id):
            continue

        doc = docs_by_id.get(tid, {})
        title = doc.get("title") or doc.get("name")
        media = (doc.get("media_type") or "").lower() if doc else ""
        if requested_media and requested_media != "all" and media != requested_media:
            continue
        results.append(
            {
                "id": int(tid),
                "title": title,
                "media_type": doc.get("media_type"),
                "score": float(score),
                "poster_path": doc.get("poster_path"),
                "overview": doc.get("overview", "No description available"),
            }
        )

        if len(results) >= k:
            break
    logger.info("Found %s results after media type filtering", len(results))
    return {"results": results}


def get_payload_type(tool_name):
    """Get the Pydantic model type for the given tool_name."""
    match tool_name:
        case "mcp_knn":
            return MCPPayload
    return None


def get_tool_function(tool_name: str):
    """Get the local function to execute for the given tool_name."""
    match tool_name:
        case "mcp_knn":
            return call_mcp_knn
    return None


def call_model_with_mcp_function(
    tool_name: str, messages: List[Dict[str, Any]], model: str, max_tokens: int
) -> Dict[str, Any]:
    """Send messages to the model, handle a tool_name function_call, and return a dict with:
    - `final_content`: assistant natural text reply
    - `function_result`: structured result returned from the function (if any)
    - `raw`: raw responses for debugging

    This function is safe for production use: it validates function args and logs issues.
    """
    client = get_openai_client()
    mcp_function_schema = load_tool(tool_name)
    payload_type = get_payload_type(tool_name)
    functions = [mcp_function_schema] if mcp_function_schema else []

    # initial model call
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        functions=functions,
        function_call="auto",
        temperature=0.0,
        max_tokens=max_tokens,
    )

    choice = resp.choices[0]
    msg = choice.message

    # check if model called a function
    if msg.get("function_call"):
        fn_call = msg["function_call"]
        fn_name = fn_call.get("name")
        args_text = fn_call.get("arguments")
        logger.info("Model requested function call: %s", fn_name)

        try:
            args = json.loads(args_text)
        except Exception as e:
            logger.error(
                "Failed to parse function arguments JSON: %s", repr(e), exc_info=True
            )
            # return the model's reply as-is and indicate failure
            return {"final_content": None, "function_result": None, "raw": resp}

        # validate using Pydantic
        try:
            payload = payload_type.model_validate(args)
        except Exception as e:
            logger.error(
                "%s validation failed: %s", payload_type, repr(e), exc_info=True
            )
            return {"final_content": None, "function_result": None, "raw": resp}

        # execute the mcp function (local call)
        try:
            tool_function = get_tool_function(fn_name)
            fn_result = tool_function(payload)
        except Exception as e:
            logger.error(
                "MCP handler failed for tool call for %s: %s",
                fn_name,
                repr(e),
                exc_info=True,
            )
            return {"final_content": None, "function_result": None, "raw": resp}

        # send function result back to the model so it can produce a final assistant reply
        followup = client.chat.completions.create(
            model=model,
            messages=messages
            + [
                {"role": "assistant", "content": "", "function_call": fn_call},
                {"role": "function", "name": fn_name, "content": json.dumps(fn_result)},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )

        final_msg = followup.choices[0].message
        return {
            "final_content": final_msg.get("content"),
            "function_result": fn_result,
            "raw": {"initial": resp, "followup": followup},
        }

    # no function call, return assistant content
    return {"final_content": msg.get("content"), "function_result": None, "raw": resp}
