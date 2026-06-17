"""Chat assistant using a LangGraph StateGraph (agent ↔ tool nodes)."""
import json
import time
from datetime import date as _date
from typing import Annotated, Any, AsyncGenerator, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.actor_history import get_actor_history
from app.config.settings import settings
from app.dao.history import get_watch_history
from app.dao.watchlist import get_watchlist
from app.process.describe_discover import discover_by_description
from app.process.tmdb_recommendation import TmdbRecommender
from app.similar import SimilarError, compute_similar
from app.taste_profile import compute_taste_profile, get_taste_text
from app.tmdb_client import get_metadata, search_persons
from app.watchlist_sync import add_to_watchlist, remove_from_watchlist
from app.will_like import WillLikeError, compute_will_like
from app.utils.logger import get_logger
from app.utils.openai_client import get_openai_client
from app.utils.prompt_registry import PromptRegistry

logger = get_logger(__name__)

_registry = PromptRegistry("app/prompts/chat")


def _system_prompt() -> str:
    try:
        taste = get_taste_text()
    except Exception:
        taste = ""
    template = _registry.load_prompt_template("system", 2)
    return template.render(today=_date.today().isoformat(), taste=taste)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _similar_json(res: Any, watched: bool = False) -> str:
    """Serialise a SimilarResponse for the chat 'similar' card payload."""
    return json.dumps({
        "type": "similar",
        "source_title": res.source_title,
        "items": [
            {
                "id": r.id,
                "title": r.title,
                "media_type": r.media_type,
                "poster_path": r.poster_path,
                "overview": r.overview,
                **({"watched": True} if watched else {}),
            }
            for r in res.results
        ],
    })


@tool
def get_recommendations(media_type: str, count: int = 5, genres: Optional[List[str]] = None) -> str:
    """Get personalized recommendations from the user's overall taste. Use for open-ended
    "recommend me something" requests with no specific title or description. For a free-text
    description, mood, era, or theme use search_by_description instead; for titles similar to
    one named title use find_similar; to recommend from the user's own watchlist use
    get_watchlist_tool.

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
def find_similar(title: str, media_type: str, count: int = 8, cross_type: bool = False) -> str:
    """Find movies or TV shows similar to one specific title the user names. Returns titles in
    general (not limited to what they've watched). Use only when the user references a concrete
    title to anchor on; for taste-based or descriptive requests use get_recommendations or
    search_by_description, and to find what they've ALREADY watched that resembles a title use
    find_similar_in_history.

    Args:
        title: title to find similar content for
        media_type: the media type of the NAMED title ('movie' or 'tv') — i.e. what `title` is,
            not what the user wants back
        count: number of results (max 20)
        cross_type: set True to return the OPPOSITE type from the source — e.g. TV shows similar
            to a movie, or movies similar to a show. When the user asks for "similar shows" about
            a film (or "similar movies" about a show), keep media_type as the source title's own
            type and set cross_type=True. Do not ask the user to name a title of the other type.
    """
    try:
        res = compute_similar(
            title=title, media_type=media_type, k=min(int(count), 20), cross_type=cross_type
        )
    except SimilarError as e:
        return json.dumps({"type": "error", "message": str(e)})
    return _similar_json(res)


@tool
def find_similar_in_history(title: str, media_type: str, count: int = 8, cross_type: bool = False) -> str:
    """Find titles the user has ALREADY watched that are similar to a given title — i.e. search
    their watch history for resemblances. Use when the user asks what in their history is like a
    title, e.g. "have I seen anything like Inception?" or "what have I watched similar to Dune?".
    For similar titles in general (not limited to their history) use find_similar instead.

    Args:
        title: the title to find history matches for
        media_type: the media type of the NAMED title ('movie' or 'tv') — i.e. what `title` is
        count: max results (max 20)
        cross_type: set True to match watched titles of the OPPOSITE type from the source — e.g.
            "have I watched any shows like this movie?". Keep media_type as the source title's
            own type and set cross_type=True.
    """
    try:
        res = compute_similar(
            title=title, media_type=media_type, k=min(int(count), 20),
            cross_type=cross_type, filter_to_history=True,
        )
    except SimilarError as e:
        return json.dumps({"type": "error", "message": str(e)})
    return _similar_json(res, watched=True)


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
    Use for "find me X" / "films matching Y" requests. Already cross-references the user's
    watch history internally and marks each result watched/not — do NOT also call get_history
    to filter or post-process these results.

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
    """Find titles in the user's watch history featuring a specific actor or director. Resolves
    the person's name internally, so call this directly — only use lookup_person first if the
    name is genuinely ambiguous.

    Args:
        name: actor or director name
    """
    result = get_actor_history(name)
    if not result.person:
        return json.dumps({"type": "error", "message": f"'{name}' not found on TMDB"})
    return json.dumps({"type": "actor_history", **result.model_dump()})


@tool
def get_history(media_type: Optional[str] = None, limit: int = 20) -> str:
    """Get the user's watch history. Use ONLY when the user explicitly asks about what or when
    they watched, or when raw dates/counts matter. Never call it to filter, rank, or
    post-process the output of another tool — search_by_description and the recommendation
    tools already account for history internally.

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


@tool
def get_watchlist_tool(media_type: Optional[str] = None) -> str:
    """Return the user's watchlist — the titles they want to watch. This is also how you
    recommend from the watchlist: fetch it (filter by media_type when the user asks for a movie
    or a show), then choose the single best fit for the user's taste — which you already know
    from the taste profile in context — and tell them which one and why. To show a poster for
    your pick, call get_title_details on that one title. Do not present the whole list back as
    your recommendation.

    Args:
        media_type: 'movie', 'tv', or None for all
    """
    items = get_watchlist(media_type)
    return json.dumps({
        "type": "watchlist",
        "titles": [
            {
                "tmdb_id": w.get("tmdb_id"),
                "title": w.get("title"),
                "media_type": w.get("media_type"),
                "genres": w.get("genres") or [],
            }
            for w in items
        ],
    })


@tool
def add_to_watchlist_tool(tmdb_id: int, media_type: str) -> str:
    """Add a movie or TV show to the user's watchlist.

    Args:
        tmdb_id: TMDB id of the title
        media_type: 'movie' or 'tv'
    """
    if media_type not in ("movie", "tv"):
        return json.dumps({"type": "error", "message": "media_type must be 'movie' or 'tv'"})
    try:
        doc = add_to_watchlist(int(tmdb_id), media_type)
        return json.dumps({"type": "watchlist_add", "title": doc.get("title"), "media_type": media_type})
    except Exception as e:
        return json.dumps({"type": "error", "message": str(e)})


@tool
def remove_from_watchlist_tool(tmdb_id: int, media_type: str) -> str:
    """Remove a movie or TV show from the user's watchlist.

    Args:
        tmdb_id: TMDB id of the title
        media_type: 'movie' or 'tv'
    """
    if media_type not in ("movie", "tv"):
        return json.dumps({"type": "error", "message": "media_type must be 'movie' or 'tv'"})
    try:
        remove_from_watchlist(int(tmdb_id), media_type)
        return json.dumps({"type": "watchlist_remove", "tmdb_id": tmdb_id, "media_type": media_type})
    except Exception as e:
        return json.dumps({"type": "error", "message": str(e)})


@tool
def get_taste_profile() -> str:
    """Describe the user's overall taste — signature, summary, favourite genres, recurring
    themes, and what they tend to avoid. Use when the user asks about their own taste, viewing
    patterns, or "what kind of viewer am I".
    """
    try:
        profile = compute_taste_profile()
    except Exception as e:
        return json.dumps({"type": "error", "message": str(e)})
    return json.dumps({"type": "taste_profile", **profile.model_dump()})


@tool
def get_title_details(tmdb_id: int, media_type: str) -> str:
    """Get details about a specific movie or TV show — overview, genres, release year, runtime,
    tagline, and TMDB rating. Use to answer factual questions about a title (e.g. "how long is
    it", "what year", "what's it about"), including titles just returned by another tool (pass
    that title's id). For the cast, use get_cast instead.

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
    release = md.get("release_date") or md.get("first_air_date") or ""
    runtime = md.get("runtime")
    if runtime is None:
        rt = md.get("episode_run_time") or []
        runtime = rt[0] if rt else None
    return json.dumps({
        "type": "title_details",
        "id": int(md.get("id")) if md.get("id") else int(tmdb_id),
        "title": md.get("title") or md.get("name"),
        "media_type": media_type,
        "year": release[:4] if release else None,
        "genres": [g.get("name") for g in (md.get("genres") or []) if g.get("name")],
        "runtime_minutes": runtime,
        "number_of_seasons": md.get("number_of_seasons"),
        "tagline": md.get("tagline") or None,
        "overview": md.get("overview") or None,
        "vote_average": round(float(md.get("vote_average")), 1) if md.get("vote_average") else None,
        "poster_path": md.get("poster_path"),
    })


_LC_TOOLS = [get_recommendations, find_similar, find_similar_in_history, will_i_like, search_by_description, get_cast, lookup_person, actor_in_history, get_history, get_watchlist_tool, add_to_watchlist_tool, remove_from_watchlist_tool, get_taste_profile, get_title_details]

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]
    revised: bool  # True once the verify node has triggered a single self-correction


# LLM is initialised lazily so the app can start without OPENAI_API_KEY set.
_llm_with_tools: Any = None


def _get_llm() -> Any:
    global _llm_with_tools
    if _llm_with_tools is None:
        llm = ChatOpenAI(
            model=settings.CHAT_MODEL,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_BASE_URL or None,
        )
        _llm_with_tools = llm.bind_tools(_LC_TOOLS)
    return _llm_with_tools


async def _agent_node(state: State) -> Dict[str, Any]:
    messages = [SystemMessage(content=_system_prompt())] + state["messages"]
    return {"messages": [await _get_llm().ainvoke(messages)]}


def _turn_context(messages: list) -> Dict[str, str]:
    """Extract the current user question, tool outputs, and final answer for this turn."""
    user, tool_outputs, answer = "", [], ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user = str(msg.content)
            break
        if isinstance(msg, ToolMessage):
            tool_outputs.append(str(msg.content))
        elif isinstance(msg, AIMessage) and not msg.tool_calls and not answer:
            content = msg.content
            if isinstance(content, list):
                content = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
            answer = str(content)
    return {"user": user, "tools": "\n".join(reversed(tool_outputs)), "answer": answer}


def _verify_node(state: State) -> Dict[str, Any]:
    """Cheaply judge whether the final answer addressed the user's question; if not, inject a
    correction nudge and route back to the agent once. Runs only on tool-using, un-revised turns."""
    ctx = _turn_context(state["messages"])
    try:
        template = _registry.load_prompt_template("verify", 1)
        prompt = template.render(user=ctx["user"], tools=ctx["tools"][:4000], answer=ctx["answer"])
        resp = get_openai_client().chat.completions.create(
            model=settings.CHAT_VERIFY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content or "{}")
    except Exception as e:
        logger.warning("verify node failed, accepting answer: %s", repr(e))
        return {"revised": True}

    if result.get("ok", True):
        return {"revised": True}
    gap = str(result.get("gap", "")).strip()
    logger.debug("verify gap: %s", gap)
    return {
        "revised": True,
        "messages": [SystemMessage(content=(
            f"Your answer so far has not fully addressed the user's question: {gap} "
            "Provide the complete answer now, written as a single clean response to the user. "
            "Do not apologise or mention that you are correcting or revising anything — the user "
            "has not seen any earlier draft."
        ))],
    }


def _should_continue(state: State) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    # Final answer produced. Verify once, but only on turns that actually used tools.
    if not state.get("revised") and _turn_context(state["messages"])["tools"]:
        return "verify"
    return END


def _after_verify(state: State) -> str:
    return "agent" if isinstance(state["messages"][-1], SystemMessage) else END


_tool_node = ToolNode(_LC_TOOLS, handle_tool_errors=True)

_graph_builder = StateGraph(State)
_graph_builder.add_node("agent", _agent_node)
_graph_builder.add_node("tools", _tool_node)
_graph_builder.add_node("verify", _verify_node)
_graph_builder.add_edge(START, "agent")
_graph_builder.add_conditional_edges("agent", _should_continue, {"tools": "tools", "verify": "verify", END: END})
_graph_builder.add_edge("tools", "agent")
_graph_builder.add_conditional_edges("verify", _after_verify, {"agent": "agent", END: END})
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
        "find_similar_in_history": lambda a: f"Searching your history for titles like {a.get('title', '?')}…",
        "will_i_like": lambda a: f"Checking if you'll like {a.get('title', '?')}…",
        "search_by_description": lambda a: f"Searching: {a.get('query', '?')}…",
        "get_cast": lambda _: "Looking up the cast…",
        "lookup_person": lambda a: f"Looking up {a.get('name', '?')} on TMDB…",
        "actor_in_history": lambda a: f"Looking up {a.get('name', '?')} in your history…",
        "get_history": lambda _: "Loading your watch history…",
        "get_watchlist_tool": lambda _: "Loading your watchlist…",
        "add_to_watchlist_tool": lambda a: f"Adding id {a.get('tmdb_id')} to watchlist…",
        "remove_from_watchlist_tool": lambda a: f"Removing id {a.get('tmdb_id')} from watchlist…",
        "get_taste_profile": lambda _: "Analysing your taste…",
        "get_title_details": lambda _: "Looking up the details…",
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
    final_message = ""  # buffered so a verify-driven correction supersedes the first answer

    async for event in _graph.astream_events(
        {"messages": [HumanMessage(content=message)], "revised": False},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 15},
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
                final_message = str(content)  # keep only the latest (possibly corrected) answer

    if final_message:
        yield _sse({"type": "message", "content": final_message})
    yield _sse({"type": "done"})
