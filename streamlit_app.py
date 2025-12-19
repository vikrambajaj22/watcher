"""
Streamlit Web App for Watcher - Movie & TV Recommendation System
Features:
- Trakt Authentication
- View & Sync Watch History
- Get Recommendations (Movies/TV/All)
- Admin Panel (Embeddings & FAISS)
- Similar Items via KNN
"""
import numpy as np
import streamlit as st
import requests
import json
import os
import time
from typing import Optional, Dict, Any

from dateutil import parser

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")
TOKEN_FILE = ".env.trakt_token"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"  # w500 for good quality

st.set_page_config(
    page_title="Watcher - Media Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Watcher\nMovie & TV Recommendation System"
    }
)

st.markdown("""
<style>
    /* hide footer only */
    footer {
        visibility: hidden;
    }
    
    /* gradient header */
    .main-header {
        font-size: 1.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* full width buttons */
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def is_authenticated() -> bool:
    """Check if user has a valid Trakt token."""
    return os.path.exists(TOKEN_FILE)


def get_poster_url(poster_path: Optional[str], fallback_text: str = "No Poster", size: str = "w500") -> str:
    """Get TMDB poster URL or fallback to placeholder.

    Args:
        poster_path: TMDB poster path (e.g., "/abc123.jpg")
        fallback_text: Text to show in placeholder if no poster
        size: TMDB image size (w92, w154, w185, w342, w500, w780, original)

    Returns:
        Full URL to poster image
    """
    if poster_path:
        return f"https://image.tmdb.org/t/p/{size}{poster_path}"
    else:
        return f"https://via.placeholder.com/500x750/667eea/ffffff?text={fallback_text.replace(' ', '+')}"


def get_token_info() -> Optional[Dict[str, Any]]:
    """Get token information from file."""
    if not is_authenticated():
        return None
    try:
        with open(TOKEN_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error reading token: {e}")
        return None


@st.cache_data(ttl=300, show_spinner=False)
def cached_api_get(endpoint: str) -> Optional[Dict]:
    """Cached GET request - 5 minute TTL (300 seconds).

    Watch history is intensive (DB queries + poster enrichment) but doesn't
    change frequently - only on manual sync. Cache for 5 minutes to reduce load.
    """
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def cached_recommendations(media_type: str, recommend_count: int) -> Optional[Dict]:
    """Cached recommendations - 10 minute TTL (600 seconds).

    Recommendations are expensive (LLM calls, embeddings, FAISS queries) but
    don't change frequently. Cache based on media_type and count to avoid
    regenerating the same recommendations.

    Args:
        media_type: Type of media (movie, tv, all)
        recommend_count: Number of recommendations requested

    Returns:
        Recommendations response or None if failed
    """
    url = f"{API_BASE_URL}/recommend/{media_type}"
    try:
        response = requests.post(url, json={"recommend_count": recommend_count})
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def cached_sync_status() -> Optional[Dict]:
    """Cached fetch of sync status from backend."""
    try:
        url = f"{API_BASE_URL}/admin/sync/status"
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def format_ts(ts: Any) -> str:
    """Format either epoch int or ISO string to a human-readable timestamp.

    Returns 'Never' for falsy values.
    """
    if not ts:
        return "Never"
    try:
        if isinstance(ts, (int, float)) or (isinstance(ts, str) and ts.isdigit()):
            return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(ts)))

        dt = parser.isoparse(ts)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return str(ts)


def clear_caches():
    """Clear local Streamlit caches used for API GETs and sync status."""
    try:
        cached_api_get.clear()
    except Exception:
        pass
    try:
        cached_sync_status.clear()
    except Exception:
        pass


def start_trakt_sync_shared(admin=False) -> Optional[Dict]:
    """Trigger /admin/sync/trakt and store returned job id in session state.

    If admin=True store under 'admin_last_sync_job_id' to avoid clashing with UI auto-sync keys.
    """
    try:
        res = api_request("/admin/sync/trakt", method="POST", data={})
        job_id = None
        if isinstance(res, dict):
            job_id = res.get('job_id')
        if job_id:
            key = 'admin_last_sync_job_id' if admin else 'last_sync_job_id'
            polled_key = 'admin_last_sync_polled' if admin else 'last_sync_polled'
            inprog_key = 'sync_in_progress'
            st.session_state[key] = job_id
            st.session_state[inprog_key] = True
            st.session_state[polled_key] = False
        else:
            # fallback: clear caches so UI will pick up later
            clear_caches()
        return res
    except Exception as e:
        return {"error": str(e)}


def poll_job_status(job_id: str, max_wait: int = 60, interval: int = 2) -> Optional[dict]:
    """Poll /admin/sync/job/{job_id} until finished or timeout.

    Returns the job JSON when finished, or None if not finished within max_wait or request fails.
    If max_wait <= 0 this will perform a single probe and return the job JSON or None.
    """
    waited = 0
    while True:
        try:
            url = f"{API_BASE_URL}/admin/sync/job/{job_id}"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                js = r.json()
                status = js.get('status')
                if status in ('completed', 'failed'):
                    return js
            else:
                # unexpected status -> treat as transient
                js = None
        except Exception:
            js = None

        # if this was a single probe, return whatever we have (likely None)
        if max_wait <= 0:
            return js

        if waited >= max_wait:
            return None

        time.sleep(interval)
        waited += interval


def api_request(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Optional[Dict]:
    """Make API request to backend."""
    # Use cache for GET requests
    if method == "GET" and data is None:
        return cached_api_get(endpoint)

    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            st.error(f"Unsupported method: {method}")
            return None

        if response.status_code in [200, 202]:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


def _set_similar_item(tmdb_id: Optional[int], title: Optional[str], media_type: Optional[str]):
    """Helper to set the similar_item in session_state from button callbacks."""
    try:
        st.session_state.similar_item = {
            'tmdb_id': tmdb_id,
            'title': title,
            'media_type': media_type,
        }
        st.session_state.active_tab = 4
    except Exception:
        st.session_state['similar_item'] = {
            'tmdb_id': tmdb_id,
            'title': title,
            'media_type': media_type,
        }
        st.session_state['active_tab'] = 4


def _check_will_like_inline(tmdb_id: int, media_type: str, result_key: str):
    """Call the will-like API and store the result under a unique session_state key for inline display.

    This does NOT change tabs - show results inline.
    """
    try:
        payload = {"tmdb_id": int(tmdb_id), "media_type": str(media_type)}
        res = api_request('/mcp/will-like', method='POST', data=payload)
    except Exception as e:
        res = {"error": str(e)}
    st.session_state[result_key] = res
    # mark that we should restore similar results on next run and show inline result
    st.session_state['_restore_similar'] = True


def _safe_rerun():
    """Call Streamlit's experimental_rerun if available, swallow errors."""
    fn = getattr(st, "experimental_rerun", None)
    if fn:
        try:
            fn()
        except Exception:
            pass


def show_auth_page():
    """Display authentication page."""
    st.markdown('<h1 class="main-header">Watcher</h1>', unsafe_allow_html=True)
    st.subheader("watchu lookin at?")

    st.info("👋 Welcome! Please authenticate with Trakt to get started.")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔐 Login with Trakt", type="primary", use_container_width=True):
            auth_url = f"{API_BASE_URL}/auth/trakt/start?from_ui=true"
            st.markdown(f"""
            ### Authentication Steps:
            1. Click the link below to authorize with Trakt
            2. After authorization, you'll be redirected back
            3. Refresh this page
            
            [🔗 Authorize with Trakt]({auth_url})
            """)

        st.markdown("---")

        if st.button("🔄 Refresh Page"):
            st.rerun()


def show_dashboard():
    """Display main dashboard with tab-based navigation."""

    col1, col2 = st.columns([2, 1])
    with col1:
        st.image("static/ui/images/watcher-logo.jpeg", width=100)
    with col2:
        if is_authenticated():
            if st.button("→ Logout", use_container_width=True):
                if os.path.exists(TOKEN_FILE):
                    os.remove(TOKEN_FILE)
                st.rerun()
        else:
            st.warning("⚠️ Not Authenticated")

    st.markdown("---")

    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0

    tab_labels = [
        "🏠 Home",
        "📺 Watch History",
        "✨ Recommendations",
        "🤔 Will I Like?",
        "🔍 Similar Items",
        "⚙️ Admin Panel"
    ]

    selected = st.radio(
        "Navigation",
        options=tab_labels,
        index=st.session_state.active_tab,
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # update active tab if selection changed
    new_index = tab_labels.index(selected)
    if new_index != st.session_state.active_tab:
        st.session_state.active_tab = new_index
        st.rerun()

    # show content based on active tab
    if st.session_state.active_tab == 0:
        show_home_page()
    elif st.session_state.active_tab == 1:
        show_history_page()
    elif st.session_state.active_tab == 2:
        show_recommendations_page()
    elif st.session_state.active_tab == 3:
        show_will_like_page()
    elif st.session_state.active_tab == 4:
        show_similar_items_page()
    elif st.session_state.active_tab == 5:
        show_admin_page()


def show_home_page():
    """Display home page with overview."""
    st.header("Welcome to Watcher! 👋")

    st.write("Use the tabs above to navigate between different sections:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('''### 📺 Watch History
        View your watched movies & shows''')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('''### ✨ Recommendations
        Get personalized suggestions''')
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('''### 🔍 Similar Items
        Find similar movies & shows''')
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")


def show_history_page():
    """Display watch history page."""
    st.header("📺 Watch History")

    # show last sync timestamps (small, cached call)
    try:
        sync_status = cached_sync_status() or {}
    except Exception:
        sync_status = {}

    last_trakt = format_ts(sync_status.get('trakt_last_activity'))
    last_tmdb_movie = format_ts(sync_status.get('tmdb_movie_last_sync'))
    last_tmdb_tv = format_ts(sync_status.get('tmdb_tv_last_sync'))
    st.caption(f"Last sync — Trakt: {last_trakt} | TMDB(movie): {last_tmdb_movie} | TMDB(tv): {last_tmdb_tv}")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("View and manage your Trakt watch history")
    with col2:
        # auto-trigger a background Trakt sync periodically (aligned with history cache TTL)
        # throttle to once per 300s (the cached_api_get TTL) to avoid spamming the API on every rerun
        last_sync_ts = st.session_state.get('history_auto_sync_ts', 0)
        try:
            last_sync_ts = float(last_sync_ts)
        except Exception:
            last_sync_ts = 0.0

        if time.time() - last_sync_ts > 300:
            try:
                res = start_trakt_sync_shared()
                st.session_state['history_auto_sync_ts'] = int(time.time())
                # if backend returned a job_id, store and poll it
                job_id = None
                if isinstance(res, dict):
                    job_id = res.get('job_id')
                if job_id:
                    st.session_state['last_sync_job_id'] = job_id
                    st.session_state['sync_in_progress'] = True
                    st.session_state['last_sync_polled'] = False
                else:
                    # fallback: clear cache so UI will pick up changes later
                    clear_caches()
            except Exception:
                # best-effort - don't block the page if sync trigger fails
                pass

    # filters shown above the history list; build them before fetching to keep UI stable
    col1, col2, col3 = st.columns(3)
    with col1:
        media_filter = st.selectbox("Media Type", ["All", "Movies", "TV Shows"])
    with col2:
        sort_by = st.selectbox("Sort By", ["Latest Watched", "Earliest Watched", "Title", "Watch Count"])
    with col3:
        search = st.text_input("🔍 Search", placeholder="Search titles...")

    # determine server-side media_type param and fetch history accordingly (server-side filtering when possible)
    if media_filter == "Movies":
        media_type_param = "movie"
    elif media_filter == "TV Shows":
        media_type_param = "tv"
    else:
        media_type_param = None

    endpoint = "/history"
    if media_type_param:
        endpoint = f"/history?media_type={media_type_param}"
    history_data = api_request(endpoint, method="GET")

    # after fetching history_data, clear any transient sync-in-progress UI flags
    try:
        if history_data and st.session_state.get('sync_in_progress'):
            for k in ['sync_in_progress', 'last_sync_job_id', 'last_sync_polled']:
                if k in st.session_state:
                    try:
                        del st.session_state[k]
                    except Exception:
                        pass
    except Exception:
        pass

    # history_data already fetched (possibly filtered server-side)
    if not history_data:
        st.info("No watch history found. Wait for sync to finish or try Refresh Data.")
        return

    # compute metrics: if the user applied a server-side media filter, fetch an unfiltered (cached) summary
    # to show totals across all media; otherwise compute from the returned history_data.
    try:
        if media_type_param:
            # fetch unfiltered, no-posters summary (cached) for totals
            overall = api_request("/history?include_posters=false", method="GET") or []
            total_movies = sum(1 for it in overall if it.get("media_type") == "movie")
            total_shows = sum(1 for it in overall if it.get("media_type") == "tv")
        else:
            total_movies = sum(1 for it in history_data if it.get("media_type") == "movie")
            total_shows = sum(1 for it in history_data if it.get("media_type") == "tv")
    except Exception:
        total_movies = sum(1 for it in history_data if it.get("media_type") == "movie")
        total_shows = sum(1 for it in history_data if it.get("media_type") == "tv")

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("🎬 Movies Watched", total_movies)
    mcol2.metric("📺 Shows Watched", total_shows)

    # if a sync job was scheduled, probe job status and poll if needed; once completed, clear job flags
    if st.session_state.get('sync_in_progress', False):
        job_id = st.session_state.get('last_sync_job_id')
        polled = st.session_state.get('last_sync_polled', False)

        # quick probe: if we've already polled once but flags still set, do a single status check and clear if finished
        if job_id and polled:
            try:
                url = f"{API_BASE_URL}/admin/sync/job/{job_id}"
                r = requests.get(url, timeout=3)
                if r.status_code == 200:
                    js = r.json()
                    status = js.get('status')
                    if status in ('completed', 'failed'):
                        # job finished - clear cache and session flags, then rerun to reflect updates
                        clear_caches()
                        st.session_state['sync_in_progress'] = False
                        # remove job id and polled marker so UI won't show sync messages
                        try:
                            del st.session_state['last_sync_job_id']
                        except Exception:
                            pass
                        try:
                            del st.session_state['last_sync_polled']
                        except Exception:
                            pass
                        _safe_rerun()
                else:
                    # still running - show ephemeral message and allow polling below
                    st.warning("⏳ Sync running in background... Polling job status...")
            except Exception:
                st.warning("⏳ Sync running in background... Polling job status...")

        # Poll job status (blocking spinner for up to 60s) if not already polled in this session run
        elif job_id and not polled:
            st.warning("⏳ Sync running in background... Polling job status...")
            with st.spinner("Waiting for sync to complete (this may take a while)..."):
                interval = 2
                waited = 0
                max_wait = 60
                while waited < max_wait:
                    try:
                        url = f"{API_BASE_URL}/admin/sync/job/{job_id}"
                        r = requests.get(url, timeout=5)
                        if r.status_code == 200:
                            js = r.json()
                            status = js.get('status')
                            if status in ('completed', 'failed'):
                                # job finished - clear cache and session flags, then rerun to reflect updates
                                clear_caches()
                                st.session_state['sync_in_progress'] = False
                                # remove job id and polled marker so UI won't show sync messages
                                try:
                                    del st.session_state['last_sync_job_id']
                                except Exception:
                                    pass
                                try:
                                    del st.session_state['last_sync_polled']
                                except Exception:
                                    pass
                                _safe_rerun()
                                break
                    except Exception:
                        pass
                    time.sleep(interval)
                    waited += interval
                # mark polled even if not completed to avoid repeated polling in this render loop
                st.session_state['last_sync_polled'] = True
        else:
            # no job id or already cleared — ensure flag is off
            st.session_state['sync_in_progress'] = False

    st.markdown("---")

    with st.spinner("Loading history..."):
        if search:
            history_data = [
                item for item in history_data
                if search.lower() in item.get("title", "").lower()
            ]

        if sort_by == "Latest Watched":
            history_data = sorted(history_data, key=lambda x: x.get("latest_watched_at", ""), reverse=True)
        elif sort_by == "Earliest Watched":
            history_data = sorted(history_data, key=lambda x: x.get("earliest_watched_at", ""))
        elif sort_by == "Title":
            history_data = sorted(history_data, key=lambda x: x.get("title", ""))
        elif sort_by == "Watch Count":
            history_data = sorted(history_data, key=lambda x: x.get("watch_count", 0), reverse=True)

        if not history_data:
            st.warning("No items match your filters.")
            return

        st.write(f"**Showing {len(history_data)} items**")
        st.markdown("---")

        for idx, item in enumerate(history_data):
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([1, 3, 1, 1, 1])

                with col1:
                    poster_url = get_poster_url(item.get("poster_path"), item.get("title", "No Title"))
                    st.image(poster_url, use_container_width=True)

                with col2:
                    icon = "🎬" if item["media_type"] == "movie" else "📺"
                    st.markdown(f"### {icon} {(item.get('title') or 'Unknown')}")
                    if item["media_type"] == "tv":
                        st.write(f"Episodes: {item.get('watched_episodes', 0)}/{item.get('total_episodes', '?')}")
                    else:
                        st.write(f"Year: {item.get('year', 'N/A')}")

                with col3:
                    if item["media_type"] == "movie":
                        st.metric("Watch Count", item.get("watch_count", 0))
                    else:
                        completion = item.get("completion_ratio", 0) * 100
                        st.metric("Completion", f"{completion:.0f}%")

                with col4:
                    st.write(f"Last: {item.get('latest_watched_at', 'N/A')[:10]}")

                with col5:
                    # avoid closure capture by binding current item values into the button args
                    _tmdb_id = item.get('tmdb_id') or item.get('id')
                    _title = item.get('title')
                    _mtype = item.get('media_type')
                    st.button(
                        "🔍 Find Similar",
                        key=f"similar_hist_{idx}",
                        on_click=_set_similar_item,
                        args=(_tmdb_id, _title, _mtype),
                    )
                st.markdown("---")


def show_recommendations_page():
    """Display recommendations page."""
    st.header("✨ Get Recommendations")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("Get personalized movie and TV show recommendations based on your watch history")

    col1, col2 = st.columns(2)
    with col1:
        media_type = st.selectbox(
            "What would you like recommendations for?",
            ["All", "Movies", "TV Shows"],
            index=0
        )
    with col2:
        recommend_count = st.slider("Number of recommendations", 1, 20, 5)

    media_type_map = {
        "All": "all",
        "Movies": "movie",
        "TV Shows": "tv"
    }

    if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
        api_media_type = media_type_map[media_type]

        with st.spinner("Generating recommendations..."):
            result = cached_recommendations(api_media_type, recommend_count)

            if result and "recommendations" in result:
                st.success(f"✅ Found {len(result['recommendations'])} recommendations!")

                st.markdown("---")

                for idx, rec in enumerate(result["recommendations"], 1):
                    with st.container():
                        st.markdown(f'<div class="recommendation-card">', unsafe_allow_html=True)

                        col1, col2 = st.columns([1, 3])

                        with col1:
                            poster_path = rec.get('metadata', {}).get('poster_path') if rec.get('metadata') else None
                            poster_url = get_poster_url(poster_path, (rec.get('title') or 'No Title'))
                            st.image(
                                poster_url,
                                use_container_width=True
                            )

                        with col2:
                            media_icon = "🎬" if rec.get('media_type') == 'movie' else "📺" if rec.get(
                                'media_type') == 'tv' else "🎭"
                            st.markdown(f"### {idx}. {media_icon} {(rec.get('title') or 'Unknown Title')}")
                            st.markdown(f"**Reasoning:** {rec.get('reasoning', 'No reasoning provided')}")

                            if rec.get('metadata'):
                                if rec.get("metadata").get("overview"):
                                    st.markdown(f"**Overview:** {rec['metadata']['overview'][:500]}...")

                            col_a, col_b = st.columns(2)
                            with col_a:
                                # bind rec values explicitly to avoid loop closure issues
                                _tmdb_id = rec.get('id')
                                _title = rec.get('title') or 'Unknown Title'
                                _mtype = rec.get('media_type', 'movie')
                                st.button(
                                    f"🔍 Find Similar",
                                    key=f"similar_rec_{idx}",
                                    on_click=_set_similar_item,
                                    args=(_tmdb_id, _title, _mtype),
                                )
                            with col_b:
                                tmdb_id = rec.get('id')
                                if tmdb_id:
                                    media_type_for_link = rec.get('media_type', 'movie')
                                    st.markdown(
                                        f"[View on TMDB](https://www.themoviedb.org/{media_type_for_link}/{tmdb_id})")

                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("---")


def show_will_like_page():
    """Display the 'Will I Like?' page where users can query by TMDB ID or by title."""
    st.header("🤔 Will I Like This?")
    st.write("Check whether an item matches your taste based on your watch history.")

    tab_id, tab_title = st.tabs(["By TMDB ID", "By Title"])

    with tab_id:
        st.subheader("Check by TMDB ID")
        col1, col2 = st.columns([3, 1])
        with col1:
            tmdb_id = st.number_input("TMDB ID", min_value=1, value=550, key="will_tmdb_id")
        with col2:
            media_type = st.selectbox("Media Type", ["movie", "tv"], key="will_id_media")
        if st.button("🤔 Check Will I Like (by ID)", key="will_check_id"):
            with st.spinner("Checking..."):
                payload = {"tmdb_id": int(tmdb_id), "media_type": media_type}
                res = api_request('/mcp/will-like', method='POST', data=payload)
                st.session_state['will_like_result'] = res
                st.session_state['will_like_tmdb_id'] = int(tmdb_id)

    with tab_title:
        st.subheader("Check by Title")
        title_str = st.text_input("Movie / TV show title", key="will_title_input")
        media_type_title = st.selectbox("Media Type", ["movie", "tv"], key="will_title_media")
        if st.button("🤔 Check Will I Like (by Title)", key="will_check_title"):
            if not title_str:
                st.error("Please provide a title")
            else:
                with st.spinner("Checking..."):
                    payload = {"title": title_str, "media_type": media_type_title}
                    res = api_request('/mcp/will-like', method='POST', data=payload)
                    st.session_state['will_like_result'] = res

    st.markdown("---")
    # show last computed will-like result if available
    if st.session_state.get('will_like_result'):
        res = st.session_state.get('will_like_result')
        if isinstance(res, dict) and 'error' in res:
            st.warning(f"Will I like? check failed: {res.get('error')}")
        else:
            item = res.get('item', {})
            score = res.get('score')
            will = res.get('will_like')
            expl = res.get('explanation')
            col1, col2 = st.columns([1, 3])
            with col1:
                poster_url = get_poster_url(item.get('poster_path'), (item.get('title') or 'No Title'))
                st.image(poster_url, use_container_width=True)
            with col2:
                emoji = '❤️' if will else '🤷'
                st.markdown(f"### {emoji} Will you like: **{(item.get('title') or 'Unknown')}**")
                if isinstance(score, (int, float)):
                    st.write(f"**Score:** {score:.3f}")
                st.write(expl)
        st.markdown('---')


def show_similar_items_page():
    """Display similar items finder using KNN."""
    st.header("🔍 Find Similar Items")

    # if we previously performed an inline check, restore persisted results now so the UI shows results + inline output
    if st.session_state.get('_restore_similar'):
        persisted = st.session_state.get('_persisted_similar_results')
        if persisted is not None:
            st.session_state['similar_results'] = list(persisted)
        try:
            del st.session_state['_restore_similar']
        except Exception:
            pass

    # if results were cleared by a rerun (e.g. after a button callback), try to restore from persisted backup
    if st.session_state.get('similar_results') is None:
        persisted = st.session_state.get('_persisted_similar_results')
        if persisted is not None:
            st.session_state['similar_results'] = list(persisted)
            # render immediately
            render_similar_results(st.session_state.get('similar_results', []), st.session_state.get('similar_source_title'))
            return

        # if no persisted copy, try to re-run last search payload
        if st.session_state.get('last_search_payload'):
            payload = st.session_state.get('last_search_payload')
            try:
                search_similar(
                    tmdb_id=payload.get('tmdb_id'),
                    title=payload.get('title'),
                    text=payload.get('text'),
                    input_media_type=payload.get('input_media_type'),
                    results_media_type=payload.get('results_media_type', 'all'),
                    k=payload.get('k', 10),
                    source_title=st.session_state.get('similar_source_title')
                )
            except Exception:
                # if re-run fails, clear last payload to avoid loops
                try:
                    del st.session_state['last_search_payload']
                except Exception:
                    pass

    # check if we came here from watch history or recommendations
    if 'similar_item' in st.session_state:
        item = st.session_state.similar_item
        st.info(f"🎯 Finding items similar to: **{item['title']}**")

        # when coming from history/recommendations, the triggering item provides its media_type as its type
        # treat that as the input_media_type (the ID's type) and keep results filter as 'all' by default
        search_similar(
            tmdb_id=item['tmdb_id'],
            input_media_type=item['media_type'],
            results_media_type='all',
            k=10,
            source_title=item['title']
        )

        # remove the trigger and render stored results (search_similar writes to session_state)
        try:
            del st.session_state.similar_item
        except Exception:
            pass
        st.markdown("---")
        # render persisted results if any
        if st.session_state.get('similar_results') is not None:
            render_similar_results(st.session_state.get('similar_results', []), st.session_state.get('similar_source_title'))
            return

    st.write("Find movies and TV shows similar to what you're looking for")

    # optional debug toggle to inspect session_state relevant keys
    debug = st.checkbox("Show debug info (session state)", key="similar_debug_toggle")
    if debug:
        st.write("session_state keys:", list(st.session_state.keys()))
        st.write({
            'similar_results': st.session_state.get('similar_results'),
            'last_search_payload': st.session_state.get('last_search_payload'),
            'similar_source_title': st.session_state.get('similar_source_title'),
        })

    tab_id, tab_title, tab_text = st.tabs(["By TMDB ID", "By Title", "By Text Description"])

    # --- By TMDB ID ---
    with tab_id:
        st.subheader("Search by TMDB ID")
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            tmdb_id = st.number_input("Enter TMDB ID", min_value=1, value=550, key="tmdb_id_input")
        with col2:
            # input media type: used to resolve the TMDB id (movie or tv)
            input_media = st.selectbox("ID Type", ["movie", "tv"], key="tmdb_input_media", help="Type of the provided TMDB ID (movie or tv)")
        with col3:
            # results media type: filter applied to KNN results
            results_media = st.selectbox("Results Type", ["movie", "tv", "all"], index=2, key="tmdb_results_media", help="Filter results to movie/tv/all")
        with col4:
            k = st.number_input("Results", min_value=1, max_value=50, value=10, key="tmdb_k")

        if st.button("🔍 Find Similar by ID", type="primary"):
            # resolve metadata using the backend API (/admin/tmdb/<id>)
            source_title = None
            try:
                endpoint = f"/admin/tmdb/{tmdb_id}"
                if input_media:
                    endpoint = f"{endpoint}?media_type={input_media}"
                md_resp = api_request(endpoint, method="GET")
                md = None
                # admin returns a list of matching documents; pick first if present
                if isinstance(md_resp, list):
                    md = md_resp[0] if len(md_resp) > 0 else None
                elif isinstance(md_resp, dict):
                    md = md_resp

                if not md:
                    raise Exception("metadata not found")

                source_title = md.get('title') or md.get('name')
                st.session_state['similar_source_metadata'] = md
                if source_title:
                    st.info(f"Searching similar to: {source_title} (TMDB id={tmdb_id}, id_type={input_media})")
            except Exception as e:
                st.warning(f"Could not fetch metadata for TMDB id {tmdb_id} (type={input_media}): {e}")
                if 'similar_source_metadata' in st.session_state:
                    try:
                        del st.session_state['similar_source_metadata']
                    except Exception:
                        pass

            # Use results_media as the results_media_type filter for returned items; pass input_media_type for ID resolution
            search_similar(tmdb_id=tmdb_id, input_media_type=input_media, results_media_type=results_media, k=k, source_title=source_title)

    # --- By Title ---
    with tab_title:
        st.subheader("Search by TMDB Title")
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            title_in = st.text_input("Title (TMDB search)", key="title_in")
        with col2:
            # input media type for the title lookup
            title_input_media = st.selectbox("Title Type", ["movie", "tv"], key="title_input_media", help="Search TMDB in this type for the provided title")
        with col3:
            # results media type filter
            title_results_media = st.selectbox("Results Type", ["movie", "tv", "all"], index=2, key="title_results_media")
        with col4:
            k_title = st.number_input("Results", min_value=1, max_value=50, value=10, key="title_k")

        if st.button("🔎 Resolve Title & Find Similar", type="primary"):
            if not title_in:
                st.error("Please provide a title to search")
            else:
                # delegate title resolution to the backend /mcp/knn API
                search_similar(title=title_in, input_media_type=title_input_media, results_media_type=title_results_media, k=k_title, source_title=title_in)

    # --- By Text Description (free-text embeddings) ---
    with tab_text:
        st.subheader("Search by Text Description")
        text_query = st.text_area(
            "Describe what you're looking for",
            placeholder="e.g., 'mind-bending thriller with a twist ending' or 'heartwarming family drama'",
            key="text_query"
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            # results media type filter only — this is the output filter
            text_results_media = st.selectbox("Results Type", ["movie", "tv", "all"], index=2, key="text_results_media")
        with col2:
            k_text = st.number_input("Results", min_value=1, max_value=50, value=10, key="text_k")

        if st.button("🔍 Find Similar by Description", type="primary") and text_query:
            # free-text search uses only the output filter
            search_similar(text=text_query, results_media_type=text_results_media, k=k_text)

    # if results exist in session state (from previous search), render them here so callbacks don't clear the page
    if st.session_state.get('similar_results') is not None:
        render_similar_results(st.session_state.get('similar_results', []), st.session_state.get('similar_source_title'))


def search_similar(tmdb_id: Optional[int] = None, title: Optional[str] = None, text: Optional[str] = None,
                    input_media_type: Optional[str] = None, results_media_type: str = "all",
                    k: int = 10, source_title: Optional[str] = None):
    """Search for similar items using the KNN endpoint."""
    payload = {
        "k": k,
        "results_media_type": results_media_type
    }

    if tmdb_id:
        payload["tmdb_id"] = tmdb_id
        if input_media_type:
            payload["input_media_type"] = input_media_type
    elif title:
        payload["title"] = title
        if input_media_type:
            payload["input_media_type"] = input_media_type
    elif text:
        payload["text"] = text
    else:
        st.error("Please provide either a TMDB ID or text description")
        return

    search_label = f"Searching for items similar to '{source_title}'..." if source_title else "Searching for similar items..."
    # perform the API call and store results in session_state so callbacks won't clear them on rerun
    with st.spinner(search_label):
        result = api_request("/mcp/knn", method="POST", data=payload)
    # if this was not an ID-based search, clear any previous persisted source metadata
    if not tmdb_id and 'similar_source_metadata' in st.session_state:
        try:
            del st.session_state['similar_source_metadata']
        except Exception:
            pass

    if result and "results" in result:
        st.session_state['similar_results'] = result['results']
        # keep a persisted backup so inline callbacks / reruns can restore results
        st.session_state['_persisted_similar_results'] = list(result['results'])
        st.session_state['similar_source_title'] = source_title
        # persist the payload so we can re-run automatically after reruns (e.g., inline button clicks)
        st.session_state['last_search_payload'] = payload
        # header/success is rendered by render_similar_results to avoid duplicate banners
    elif result:
        st.session_state['similar_results'] = []
        st.session_state['_persisted_similar_results'] = []
        st.session_state['similar_source_title'] = source_title
        st.session_state['last_search_payload'] = payload
        st.warning("No similar items found. Try adjusting your search.")


def render_similar_results(results, source_title: Optional[str] = None):
    """Render similar results list (reads inline will-like keys from session_state)."""
    # Determine a display title: prefer explicit source_title argument, then persisted metadata title
    display_title = source_title
    metadata = st.session_state.get('similar_source_metadata')
    metadata_title = None
    metadata_poster = None
    if not display_title and metadata:
        metadata_title = (metadata.get('title') or metadata.get('name'))
        display_title = metadata_title
        metadata_poster = metadata.get('poster_path')

    if display_title:
        st.markdown(f"### 🔍 Results similar to: **{display_title}**")
        # show poster if available
        poster_to_show = metadata_poster or (None)
        if poster_to_show:
            st.image(get_poster_url(poster_to_show, display_title), width=120)
        st.markdown("---")
    else:
        st.markdown(f"✅ Found {len(results)} similar items!")
        st.markdown("---")

    for idx, item in enumerate(results, 1):
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])

            with col1:
                poster_url = get_poster_url(item.get("poster_path"), (item.get("title") or "Unknown"))
                st.image(poster_url, use_container_width=True)

            with col2:
                icon = "🎬" if item.get("media_type") == "movie" else "📺"
                st.markdown(f"### {icon} {item.get('title', 'Unknown')}")
                if item.get("overview") and item.get("overview").strip():
                    st.write(f"**Overview:** {item.get('overview')[:500]}...")

                if "score" in item:
                    d = item.get("score")
                    sigma = 1.0
                    score = np.exp(-d / (2 * sigma ** 2))
                    st.progress(float(score))

            with col3:
                st.metric("ID", item.get("id", "N/A"))
                st.metric("Type", item.get("media_type", "N/A"))
                _id = item.get("id")
                _mtype = item.get("media_type")
                if _id:
                    btn_key = f"will_like_sim_{idx}_{_id}"
                    inline_key = f"will_like_inline_{_id}_{_mtype}"
                    # imperative button handling to avoid callback-ordering issues
                    clicked = st.button("🤔 Will I Like?", key=btn_key)
                    if clicked:
                        # perform the inline check immediately and persist result in session_state
                        _check_will_like_inline(_id, _mtype, inline_key)
                        # force a rerun so the inline result and persisted results display together
                        _safe_rerun()
                    # render inline result if present
                    res = st.session_state.get(inline_key)
                    if res:
                        if isinstance(res, dict) and 'error' in res:
                            st.warning(f"Will I like? check failed: {res.get('error')}")
                        else:
                            item2 = res.get('item', {})
                            score = res.get('score')
                            will = res.get('will_like')
                            expl = res.get('explanation')
                            emoji = '❤️' if will else '🤷'
                            st.markdown(f"**{emoji} {(item2.get('title') or 'Unknown')}**")
                            if isinstance(score, (int, float)):
                                st.write(f"Score: {score:.3f}")
                            st.write(expl)

            st.markdown("---")


def show_admin_page():
    """Display admin panel for management tasks."""
    st.header("⚙️ Admin Panel")
    st.warning("⚠️ These actions can be resource-intensive. Use with caution.")
    tab_status, tab_sync, tab_embedding, tab_faiss, tab_state = st.tabs([
        "📊 Status",
        "🔄 Manual Sync",
        "🧠 Embeddings",
        "📇 FAISS Index",
        "♻️ State/Cache Management",
    ])

    with tab_status:
        st.subheader("System Status")
        col1, col2 = st.columns(2)
        with col1:
            api_status = api_request("/health")
            is_online = api_status is not None and api_status.get("status") == "ok"
            st.metric("API Status", "🟢 Online" if is_online else "🔴 Offline")
        with col2:
            st.metric("Auth Status", "✅ Authenticated" if is_authenticated() else "❌ Not Authenticated")

        st.markdown("---")

        # show last sync metadata for convenience
        try:
            s = cached_sync_status() or {}
        except Exception:
            s = {}


        st.caption(
            f"Last sync — Trakt: {format_ts(s.get('trakt_last_activity'))} | TMDB(movie): {format_ts(s.get('tmdb_movie_last_sync'))} | TMDB(tv): {format_ts(s.get('tmdb_tv_last_sync'))}"
        )
        if st.button("Refresh sync status"):
            try:
                clear_caches()
            except Exception:
                pass

        st.subheader("🔑 Token Information")
        if st.button("Show Token Info"):
            token_info = get_token_info()
            if token_info:
                st.json(token_info)
            else:
                st.info("No token information available")

    with tab_embedding:
        st.subheader("🧠 Embedding Management")
        st.write("Generate embeddings for similarity search")

        st.markdown("#### Embed Single Item")
        col1, col2 = st.columns(2)
        with col1:
            embed_id = st.number_input("TMDB ID", min_value=1, key="embed_id")
        with col2:
            embed_media_type = st.selectbox("Media Type", ["movie", "tv"], key="embed_media")

        if st.button("🎯 Embed Single Item"):
            with st.spinner("Generating embedding..."):
                result = api_request(
                    "/admin/embed/item",
                    method="POST",
                    data={"id": embed_id, "media_type": embed_media_type}
                )
                if result:
                    st.success("✅ Embedding generation started!")
                    st.json(result)

        st.markdown("---")

        st.markdown("#### Generate All Embeddings")
        st.warning("⚠️ This will generate embeddings for all items in the database. This may take a while.")

        batch_size = st.number_input("Batch Size", min_value=1, max_value=1000, value=256, key="batch_size")

        if st.button("🚀 Generate All Embeddings", type="primary"):
            with st.spinner("Starting full embedding generation..."):
                result = api_request(
                    "/admin/embed/full",
                    method="POST",
                    data={"batch_size": batch_size}
                )
                if result:
                    st.success("✅ Full embedding generation started!")
                    st.json(result)

    with tab_faiss:
        st.subheader("📇 FAISS Index Management")

        st.write("Build or rebuild the FAISS index for fast similarity search")

        col1, col2 = st.columns(2)
        with col1:
            dim = st.number_input("Embedding Dimensions", min_value=1, value=384, key="faiss_dim")
        with col2:
            factory = st.text_input("Factory String", value="IDMap,IVF100,Flat", key="faiss_factory")

        st.info("""
        **Factory String Examples:**
        - `Flat` - Exact search, no compression
        - `IVF100,Flat` - Inverted file with 100 clusters
        - `IVF100,PQ8` - Inverted file + product quantization
        - `IDMap,IVF100,Flat` - With ID mapping (recommended)
        """)

        if st.button("🔨 Rebuild FAISS Index", type="primary"):
            st.warning("⚠️ This operation will rebuild the entire FAISS index!")
            with st.spinner("Rebuilding FAISS index..."):
                result = api_request(
                    "/admin/faiss/rebuild",
                    method="POST",
                    data={"dim": dim, "factory": factory}
                )
                if result:
                    st.success("✅ FAISS rebuild started!")
                    st.json(result)
                    if "log" in result:
                        st.info(f"Check logs at: {result['log']}")

    with tab_state:
        st.write("Manage caches.")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Clear API GET cache"):
                try:
                    cached_api_get.clear()
                    st.success("API GET cache cleared")
                except Exception as e:
                    st.error(f"Failed to clear API GET cache: {e}")
            if st.button("Clear Recommendations cache"):
                try:
                    cached_recommendations.clear()
                    st.success("Recommendations cache cleared")
                except Exception as e:
                    st.error(f"Failed to clear recommendations cache: {e}")
        with col_b:
            if st.button("Clear Persisted Similar Results"):
                for k in ['similar_results', '_persisted_similar_results', 'similar_source_title', 'last_search_payload']:
                    if k in st.session_state:
                        try:
                            del st.session_state[k]
                        except Exception:
                            pass
                st.success("Cleared Persisted Similar Search state")

    with tab_sync:
        st.subheader("🔄 Manual Trakt Sync")
        st.write("Trigger a manual Trakt history sync and monitor its job status.")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("Use this to start a manual Trakt sync (runs on the backend). The UI can poll the job status until completion.")
        with col2:
            if st.button("Start Trakt Sync", type="primary"):
                with st.spinner("Starting Trakt sync..."):
                    res = start_trakt_sync_shared(admin=True)
                    if isinstance(res, dict) and res.get('job_id'):
                        st.success(f"Sync started (job_id={res.get('job_id')})")
                    else:
                        st.error(f"Sync request failed: {res}")

        job_id = st.session_state.get('admin_last_sync_job_id')
        if job_id:
            st.markdown(f"**Last job id:** {job_id}")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Check job status now"):
                    js = poll_job_status(job_id, max_wait=0)
                    if js:
                        st.json(js)
                        if js.get('status') in ('completed', 'failed'):
                            clear_caches()
                            try:
                                del st.session_state['admin_last_sync_job_id']
                            except Exception:
                                pass
                    else:
                        st.warning("Job status not available (still running or not found)")
            with col_b:
                if st.button("Poll until complete (60s)"):
                    with st.spinner("Polling job status until completion..."):
                        js = poll_job_status(job_id, max_wait=60, interval=2)
                        if js:
                            st.success(f"Job finished: {js.get('status')}")
                            clear_caches()
                            try:
                                del st.session_state['admin_last_sync_job_id']
                            except Exception:
                                pass
                            st.json(js)
                        else:
                            st.warning("Polling ended without completion. Check later or view job status.")

        st.markdown("---")
        if st.button("Clear History Cache"):
            try:
                res = api_request('/admin/clear-history-cache', method='POST', data={})
                if res and res.get('status') == 'cleared':
                    cached_api_get.clear()
                    st.success('History cache cleared')
                else:
                    st.error(f'Failed to clear history cache: {res}')
            except Exception as e:
                st.error(f'Error clearing history cache: {e}')


def main():
    """Main application entry point."""

    if not is_authenticated():
        show_auth_page()
    else:
        show_dashboard()


if __name__ == "__main__":
    main()
