"""Chat assistant using a LangGraph StateGraph (agent ↔ tool nodes)."""
import json
from datetime import date as _date
from typing import Annotated, Any, Dict, Generator, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.actor_history import get_actor_history
from app.config.settings import settings
from app.dao.history import get_watch_history
from app.process.describe_discover import discover_by_description
from app.process.tmdb_recommendation import TmdbRecommender
from app.schemas.api import ChatMessage
from app.tmdb_client import search_by_title
from app.tmdb_discover import fetch_similar_and_recommendations
from app.will_like import WillLikeError, compute_will_like
from app.utils.logger import get_logger

logger = get_logger(__name__)

_MODEL = "gpt-4.1-nano"

_SYSTEM_TEMPLATE = """You are Watcher, a personal movie and TV assistant. You help users explore their watch history, discover new titles, and understand their taste.

Today's date is {today}. Use this to resolve relative time references like "yesterday" or "last week" against the watched_at dates returned by get_history.

Available tools: get_recommendations, find_similar, will_i_like, search_by_description, actor_in_history, get_history.

Keep responses concise and conversational. Briefly say what you're looking up before calling a tool."""


def _system_prompt() -> str:
    return _SYSTEM_TEMPLATE.format(today=_date.today().isoformat())

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def get_recommendations(media_type: str, count: int = 5) -> str:
    """Get personalized movie or TV recommendations based on watch history.

    Args:
        media_type: 'movie', 'tv', or 'all'
        count: number of recommendations (max 10)
    """
    count = min(int(count), 10)
    result, _ = TmdbRecommender().generate(media_type=media_type, recommend_count=count)
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
def search_by_description(query: str, limit: int = 10) -> str:
    """Find titles matching a natural language description, mood, genre, era, or theme.

    Args:
        query: natural language description
        limit: max results (default 10)
    """
    limit = min(int(limit), 20)
    res = discover_by_description(query, limit=limit)
    return json.dumps({
        "type": "discover",
        "items": [r.model_dump() for r in res.results],
        "filters": res.filters.model_dump() if res.filters else None,
    })


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
    """Get the user's watch history, optionally filtered by type.

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


_LC_TOOLS = [get_recommendations, find_similar, will_i_like, search_by_description, actor_in_history, get_history]

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


def _agent_node(state: State) -> Dict[str, Any]:
    messages = [SystemMessage(content=_system_prompt())] + state["messages"]
    return {"messages": [_get_llm().invoke(messages)]}


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
_graph = _graph_builder.compile()

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
        "actor_in_history": lambda a: f"Looking up {a.get('name', '?')} in your history…",
        "get_history": lambda _: "Loading your watch history…",
    }
    fn = labels.get(name)
    return fn(args) if fn else f"Running {name}…"


def stream_chat(messages: List[ChatMessage]) -> Generator[str, None, None]:
    """Run the LangGraph chat loop, yielding SSE event strings."""
    try:
        yield from _stream_chat_inner(messages)
    except Exception as e:
        logger.error("chat stream error: %s", repr(e), exc_info=True)
        yield _sse({"type": "error", "message": "An error occurred"})
        yield _sse({"type": "done"})


def _stream_chat_inner(messages: List[ChatMessage]) -> Generator[str, None, None]:
    lc_messages = []
    for m in messages:
        if m.role == "user":
            lc_messages.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            lc_messages.append(AIMessage(content=m.content))

    # Maps tool_call_id → tool_name for correlating ToolMessages back to their tool.
    pending: Dict[str, str] = {}

    for chunk in _graph.stream(
        {"messages": lc_messages},
        stream_mode="updates",
        config={"recursion_limit": 10},
    ):
        for node_name, update in chunk.items():
            msgs: list = update.get("messages", [])

            if node_name == "agent":
                last = msgs[-1] if msgs else None
                if not isinstance(last, AIMessage):
                    continue
                if last.tool_calls:
                    for tc in last.tool_calls:
                        pending[tc["id"]] = tc["name"]
                        yield _sse({
                            "type": "tool_start",
                            "tool": tc["name"],
                            "label": _tool_label(tc["name"], tc.get("args", {})),
                        })
                else:
                    content = last.content
                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in content
                        )
                    yield _sse({"type": "message", "content": str(content or "")})

            elif node_name == "tools":
                for msg in msgs:
                    if not isinstance(msg, ToolMessage):
                        continue
                    tool_name = msg.name or pending.get(msg.tool_call_id, "")
                    raw = msg.content
                    if isinstance(raw, list):
                        raw = json.dumps(raw)
                    try:
                        data = json.loads(raw)
                    except Exception:
                        data = {"type": "text", "content": str(raw)}
                    yield _sse({"type": "tool_result", "tool": tool_name, "data": data})

    yield _sse({"type": "done"})
