"""Chat assistant using a LangGraph StateGraph (agent ↔ tool nodes)."""
import json
import time
from datetime import date as _date
from typing import Annotated, Any, AsyncGenerator, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.actor_history import get_actor_history
from app.config.settings import settings
from app.dao.history import get_watch_history
from app.process.describe_discover import discover_by_description
from app.process.tmdb_recommendation import TmdbRecommender
from app.tmdb_client import get_metadata, search_by_title, search_persons
from app.tmdb_discover import fetch_similar_and_recommendations
from app.will_like import WillLikeError, compute_will_like
from app.utils.logger import get_logger
from app.utils.prompt_registry import PromptRegistry

logger = get_logger(__name__)

_MODEL = "gpt-4.1-mini"
_registry = PromptRegistry("app/prompts/chat")


def _system_prompt() -> str:
    template = _registry.load_prompt_template("system", 1)
    return template.render(today=_date.today().isoformat())

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def get_recommendations(media_type: str, count: int = 5, genres: Optional[List[str]] = None) -> str:
    """Get personalized movie or TV recommendations based on watch history.

    Args:
        media_type: 'movie', 'tv', or 'all'
        count: number of recommendations (max 10)
        genres: optional list of genre constraints (e.g. ['comedy'], ['sci-fi', 'thriller'])
    """
    count = min(int(count), 10)
    result, _ = TmdbRecommender().generate(media_type=media_type, recommend_count=count, genre_hint=genres or None)
    return json.dumps({
        "type": "recommendations",
        "items": [
            {
                "id": int(r.id) if str(r.id).isdigit() else 0,
                "title": r.title,
                "media_type": r.media_type,
                "reasoning": r.reasoning,
                "poster_path": (r.metadata or {}).get("poster_path"),
                "overview": (r.metadata or {}).get("overview"),
            }
            for r in result.recommendations
        ],
    })


@tool
def find_similar(title: str, media_type: str, count: int = 8) -> str:
    """Find movies or TV shows similar to a given title.

    Args:
        title: title to find similar content for
        media_type: 'movie' or 'tv'
        count: number of results (max 20)
    """
    count = min(int(count), 20)
    md = search_by_title(title, media_type=media_type)
    if not md or not md.get("id"):
        return json.dumps({"type": "error", "message": f"'{title}' not found on TMDB"})
    tmdb_id = int(md["id"])
    raw = fetch_similar_and_recommendations(tmdb_id, media_type, per_endpoint=count)
    return json.dumps({
        "type": "similar",
        "source_title": md.get("title") or md.get("name"),
        "items": [
            {
                "id": int(r["id"]),
                "title": r.get("title"),
                "media_type": r.get("media_type"),
                "poster_path": r.get("poster_path"),
                "overview": r.get("overview"),
            }
            for r in raw[:count]
        ],
    })


@tool
def will_i_like(title: str, media_type: str) -> str:
    """Predict whether the user will like a specific movie or TV show.

    Args:
        title: title to evaluate
        media_type: 'movie' or 'tv'
    """
    try:
        return json.dumps({"type": "will_like", **compute_will_like(None, title, media_type).model_dump()})
    except WillLikeError as e:
        return json.dumps({"type": "error", "message": str(e)})


@tool
def search_by_description(query: str, limit: int = 10, exclude_watched: bool = True) -> str:
    """Find titles matching a natural language description, mood, genre, era, or theme.
    Cross-references the user's watch history and marks each result as watched or not.

    Args:
        query: natural language description
        limit: max results (default 10)
        exclude_watched: True (default) to omit titles already in the user's history — set False if the user wants to include or find titles they've already seen
    """
    limit = min(int(limit), 20)
    fetch_limit = min(limit * 2, 40) if exclude_watched else limit
    res = discover_by_description(query, limit=fetch_limit)
    all_items = [r.model_dump() for r in res.results]
    if exclude_watched:
        items = [i for i in all_items if not i.get("watched")][:limit]
        watched_excluded = sum(1 for i in all_items if i.get("watched"))
    else:
        items = all_items[:limit]
        watched_excluded = 0
    return json.dumps({
        "type": "discover",
        "items": items,
        "filters": res.filters.model_dump() if res.filters else None,
        "watched_excluded": watched_excluded,
    })


@tool
def get_cast(tmdb_id: int, media_type: str) -> str:
    """Get the top-billed cast of a specific movie or TV show. Use this to answer questions
    about who stars in a title — including questions about titles just returned by another
    tool (pass the title's id from that tool's result).

    Args:
        tmdb_id: TMDB id of the title
        media_type: 'movie' or 'tv'
    """
    if media_type not in ("movie", "tv"):
        return json.dumps({"type": "error", "message": "media_type must be 'movie' or 'tv'"})
    try:
        md = get_metadata(int(tmdb_id), media_type=media_type)
    except Exception:
        return json.dumps({"type": "error", "message": f"id {tmdb_id} not found on TMDB"})
    cast = (md.get("credits") or {}).get("cast") or []
    return json.dumps({
        "type": "cast",
        "title": md.get("title") or md.get("name"),
        "cast": [
            {"name": c.get("name"), "character": c.get("character")}
            for c in cast[:8] if c.get("name")
        ],
    })


@tool
def lookup_person(name: str) -> str:
    """Look up an actor or director on TMDB to get their profile and known-for titles.
    Use this to answer questions about a person or to confirm who the user is referring to
    before calling actor_in_history.

    Args:
        name: actor or director name
    """
    results = search_persons(name, limit=3)
    if not results:
        return json.dumps({"type": "error", "message": f"'{name}' not found on TMDB"})
    return json.dumps({"type": "person_lookup", "results": results})


@tool
def actor_in_history(name: str) -> str:
    """Find titles in the user's watch history featuring a specific actor or director.

    Args:
        name: actor or director name
    """
    result = get_actor_history(name)
    if not result.person:
        return json.dumps({"type": "error", "message": f"'{name}' not found on TMDB"})
    return json.dumps({"type": "actor_history", **result.model_dump()})


@tool
def get_history(media_type: Optional[str] = None, limit: int = 20) -> str:
    """Get the user's watch history. Use only when the user asks about what they've watched
    or when dates/counts matter. Do not call just to filter search results.

    Args:
        media_type: 'movie' or 'tv' (optional)
        limit: max results (default 20)
    """
    limit = min(int(limit), 50)
    history = get_watch_history(media_type=media_type, include_posters=True)
    return json.dumps({
        "type": "history",
        "items": [
            {
                "id": h.get("id") or h.get("tmdb_id"),
                "title": h.get("title") or h.get("name"),
                "media_type": h.get("media_type"),
                "poster_path": h.get("poster_path"),
                "watched_at": h.get("latest_watched_at"),
            }
            for h in history[:limit]
        ],
    })


_LC_TOOLS = [get_recommendations, find_similar, will_i_like, search_by_description, get_cast, lookup_person, actor_in_history, get_history]

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]


# LLM is initialised lazily so the app can start without OPENAI_API_KEY set.
_llm_with_tools: Any = None


def _get_llm() -> Any:
    global _llm_with_tools
    if _llm_with_tools is None:
        llm = ChatOpenAI(
            model=_MODEL,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE_URL or None,
        )
        _llm_with_tools = llm.bind_tools(_LC_TOOLS)
    return _llm_with_tools


async def _agent_node(state: State) -> Dict[str, Any]:
    messages = [SystemMessage(content=_system_prompt())] + state["messages"]
    return {"messages": [await _get_llm().ainvoke(messages)]}


def _should_continue(state: State) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


_tool_node = ToolNode(_LC_TOOLS, handle_tool_errors=True)

_graph_builder = StateGraph(State)
_graph_builder.add_node("agent", _agent_node)
_graph_builder.add_node("tools", _tool_node)
_graph_builder.add_edge(START, "agent")
_graph_builder.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
_graph_builder.add_edge("tools", "agent")
_checkpointer = MemorySaver()
_graph = _graph_builder.compile(checkpointer=_checkpointer)

# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------

def _sse(data: Any) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _tool_label(name: str, args: Dict[str, Any]) -> str:
    labels: Dict[str, Any] = {
        "get_recommendations": lambda a: f"Getting {a.get('media_type', 'all')} recommendations…",
        "find_similar": lambda a: f"Finding titles similar to {a.get('title', '?')}…",
        "will_i_like": lambda a: f"Checking if you'll like {a.get('title', '?')}…",
        "search_by_description": lambda a: f"Searching: {a.get('query', '?')}…",
        "get_cast": lambda _: "Looking up the cast…",
        "lookup_person": lambda a: f"Looking up {a.get('name', '?')} on TMDB…",
        "actor_in_history": lambda a: f"Looking up {a.get('name', '?')} in your history…",
        "get_history": lambda _: "Loading your watch history…",
    }
    fn = labels.get(name)
    return fn(args) if fn else f"Running {name}…"


async def stream_chat(thread_id: str, message: str) -> AsyncGenerator[str, None]:
    """Run the LangGraph chat loop, yielding SSE event strings."""
    try:
        async for chunk in _stream_chat_inner(thread_id, message):
            yield chunk
    except Exception as e:
        logger.error("chat stream error: %s", repr(e), exc_info=True)
        yield _sse({"type": "error", "message": "An error occurred"})
        yield _sse({"type": "done"})


async def _stream_chat_inner(thread_id: str, message: str) -> AsyncGenerator[str, None]:
    pending_times: Dict[str, float] = {}  # run_id → monotonic start time

    async for event in _graph.astream_events(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 10},
        version="v2",
    ):
        kind = event["event"]

        if kind == "on_tool_start":
            run_id = event["run_id"]
            name = event["name"]
            args = event["data"].get("input") or {}
            pending_times[run_id] = time.monotonic()
            logger.debug("tool_call: %s %s", name, args)
            yield _sse({
                "type": "tool_start",
                "tool": name,
                "label": _tool_label(name, args),
                "args": args,
                "run_id": run_id,
            })

        elif kind == "on_tool_end":
            run_id = event["run_id"]
            name = event["name"]
            raw = event["data"].get("output")
            if hasattr(raw, "content"):
                raw = raw.content
            if isinstance(raw, list):
                raw = json.dumps(raw)
            raw = str(raw) if raw is not None else "{}"
            try:
                data = json.loads(raw)
            except Exception:
                data = {"type": "text", "content": raw}
            duration_ms = int((time.monotonic() - pending_times.pop(run_id, time.monotonic())) * 1000)
            yield _sse({
                "type": "tool_result",
                "tool": name,
                "run_id": run_id,
                "data": data,
                "duration_ms": duration_ms,
            })

        elif kind == "on_chat_model_end":
            output = event["data"].get("output")
            if not output:
                continue
            if getattr(output, "tool_calls", None):
                continue  # intermediate LLM call that produced tool invocations
            content = getattr(output, "content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                )
            if content:
                yield _sse({"type": "message", "content": str(content)})

    yield _sse({"type": "done"})
