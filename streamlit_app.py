"""
Streamlit Web App for Watcher - Movie & TV Recommendation System
Features:
- Trakt Authentication
- View & Sync Watch History
- Get Recommendations (Movies/TV/All)
- Admin Panel (Embeddings & FAISS)
- Similar Items via KNN
"""

import streamlit as st
import requests
import json
import os
from typing import Optional, Dict, Any

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


@st.cache_data(ttl=600, show_spinner=False)
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
        show_similar_items_page()
    elif st.session_state.active_tab == 4:
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

    st.subheader("👨🏻‍💻 Quick Stats")

    history = api_request("/history", method="GET")
    if history:
        movies = [item for item in history if item.get("media_type") == "movie"]
        shows = [item for item in history if item.get("media_type") == "tv"]

        col1, col2, col3 = st.columns(3)
        col1.metric("🎬 Movies Watched", len(movies))
        col2.metric("📺 Shows Watched", len(shows))


def show_history_page():
    """Display watch history page."""
    st.header("📺 Watch History")

    def trigger_sync():
        """Trigger sync and clear cache."""
        result = api_request("/admin/sync/trakt", method="POST", data={})
        if result:
            # clear the cache immediately
            cached_api_get.clear()
            st.session_state.sync_in_progress = True

    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("View and manage your Trakt watch history")
    with col2:
        if st.button("🔄 Sync from Trakt", type="primary", on_click=trigger_sync):
            pass  # callback handles sync

    if st.session_state.get('sync_in_progress', False):
        st.warning("⏳ Sync running in background... Wait 5-10 seconds, then click refresh below.")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔄 Refresh Data", type="primary"):
                cached_api_get.clear()
                del st.session_state.sync_in_progress
                st.rerun()
        with col2:
            st.caption("Click this after waiting for the sync to complete on the server.")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        media_filter = st.selectbox("Media Type", ["All", "Movies", "TV Shows"])
    with col2:
        sort_by = st.selectbox("Sort By", ["Latest Watched", "Earliest Watched", "Title", "Watch Count"])
    with col3:
        search = st.text_input("🔍 Search", placeholder="Search titles...")

    with st.spinner("Loading history..."):
        media_type_param = None
        if media_filter == "Movies":
            media_type_param = "movie"
        elif media_filter == "TV Shows":
            media_type_param = "tv"

        endpoint = "/history"
        if media_type_param:
            endpoint = f"/history?media_type={media_type_param}"

        history_data = api_request(endpoint, method="GET")

        if not history_data:
            st.info("No watch history found. Click 'Sync from Trakt' to fetch your history.")
            return

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
                    st.markdown(f"### {icon} {item['title']}")
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
                    def switch_to_similar_hist():
                        st.session_state.similar_item = {
                            'tmdb_id': item.get('tmdb_id') or item.get('id'),
                            'title': item.get('title'),
                            'media_type': item.get('media_type')
                        }
                        st.session_state.active_tab = 3

                    st.button(
                        "🔍 Find Similar",
                        key=f"similar_hist_{idx}",
                        on_click=switch_to_similar_hist
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
                            poster_url = get_poster_url(poster_path, rec.get('title', 'No Title'))
                            st.image(
                                poster_url,
                                use_container_width=True
                            )

                        with col2:
                            media_icon = "🎬" if rec.get('media_type') == 'movie' else "📺" if rec.get(
                                'media_type') == 'tv' else "🎭"
                            st.markdown(f"### {idx}. {media_icon} {rec.get('title', 'Unknown Title')}")
                            st.markdown(f"**Reasoning:** {rec.get('reasoning', 'No reasoning provided')}")

                            if rec.get('metadata'):
                                if rec.get("metadata").get("overview"):
                                    st.markdown(f"**Overview:** {rec['metadata']['overview'][:500]}...")

                            col_a, col_b = st.columns(2)
                            with col_a:
                                def switch_to_similar_rec():
                                    st.session_state.similar_item = {
                                        'tmdb_id': rec.get('id'),
                                        'title': rec.get('title', 'Unknown Title'),
                                        'media_type': rec.get('media_type', 'movie')
                                    }
                                    st.session_state.active_tab = 3

                                st.button(
                                    f"🔍 Find Similar",
                                    key=f"similar_rec_{idx}",
                                    on_click=switch_to_similar_rec
                                )
                            with col_b:
                                tmdb_id = rec.get('id')
                                if tmdb_id:
                                    media_type_for_link = rec.get('media_type', 'movie')
                                    st.markdown(
                                        f"[View on TMDB](https://www.themoviedb.org/{media_type_for_link}/{tmdb_id})")

                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("---")


def show_similar_items_page():
    """Display similar items finder using KNN."""
    st.header("🔍 Find Similar Items")

    # check if we came here from watch history or recommendations
    if 'similar_item' in st.session_state:
        item = st.session_state.similar_item
        st.info(f"🎯 Finding items similar to: **{item['title']}**")

        search_similar(
            tmdb_id=item['tmdb_id'],
            media_type=item['media_type'],
            k=10,
            source_title=item['title']
        )

        del st.session_state.similar_item
        st.markdown("---")

    st.write("Find movies and TV shows similar to what you're looking for")

    tab1, tab2 = st.tabs(["By TMDB ID", "By Text Description"])

    with tab1:
        st.subheader("Search by TMDB ID")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            tmdb_id = st.number_input("Enter TMDB ID", min_value=1, value=550, key="tmdb_id_input")
        with col2:
            media_type = st.selectbox("Media Type", ["movie", "tv", "all"], key="tmdb_media_type")
        with col3:
            k = st.number_input("Results", min_value=1, max_value=50, value=10, key="tmdb_k")
        if st.button("🔍 Find Similar by ID", type="primary"):
            search_similar(tmdb_id=tmdb_id, media_type=media_type, k=k)

    with tab2:
        st.subheader("Search by Text Description")
        text_query = st.text_area(
            "Describe what you're looking for",
            placeholder="e.g., 'mind-bending thriller with a twist ending' or 'heartwarming family drama'",
            key="text_query"
        )
        col1, col2 = st.columns(2)
        with col1:
            media_type = st.selectbox("Media Type", ["movie", "tv", "all"], key="text_media_type")
        with col2:
            k = st.number_input("Results", min_value=1, max_value=50, value=10, key="text_k")
        if st.button("🔍 Find Similar by Description", type="primary") and text_query:
            search_similar(text=text_query, media_type=media_type, k=k)


def search_similar(tmdb_id: Optional[int] = None, text: Optional[str] = None,
                   media_type: str = "all", k: int = 10, source_title: Optional[str] = None):
    """Search for similar items using the KNN endpoint."""
    payload = {
        "k": k,
        "media_type": media_type
    }

    if tmdb_id:
        payload["tmdb_id"] = tmdb_id
    elif text:
        payload["text"] = text
    else:
        st.error("Please provide either a TMDB ID or text description")
        return

    search_label = f"Searching for items similar to '{source_title}'..." if source_title else "Searching for similar items..."
    with st.spinner(search_label):
        result = api_request("/mcp/knn", method="POST", data=payload)

        if result and "results" in result:
            header = f"✅ Found {len(result['results'])} items similar to **{source_title}**!" if source_title else f"✅ Found {len(result['results'])} similar items!"
            st.success(header)

            st.markdown("---")

            for idx, item in enumerate(result["results"], 1):
                with st.container():
                    col1, col2, col3 = st.columns([1, 3, 1])

                    with col1:
                        poster_url = get_poster_url(item.get("poster_path"), item.get("title", "Unknown"))
                        st.image(
                            poster_url,
                            use_container_width=True
                        )

                    with col2:
                        icon = "🎬" if item.get("media_type") == "movie" else "📺"
                        st.markdown(f"### {icon} {item.get('title', 'Unknown')}")
                        st.write(f"**Overview:** {item.get('overview', 'No description available')[:500]}...")

                        if "distance" in item or "similarity" in item:
                            score = item.get("similarity", 1 - item.get("distance", 0))
                            st.progress(float(score))

                    with col3:
                        st.metric("ID", item.get("id", "N/A"))
                        st.metric("Type", item.get("media_type", "N/A"))

                    st.markdown("---")
        elif result:
            st.warning("No similar items found. Try adjusting your search.")


def show_admin_page():
    """Display admin panel for management tasks."""
    st.header("⚙️ Admin Panel")
    st.warning("⚠️ These actions can be resource-intensive. Use with caution.")
    tab1, tab2, tab3 = st.tabs(["📊 Status", "🧠 Embeddings", "📇 FAISS Index"])

    with tab1:
        st.subheader("System Status")

        col1, col2, col3 = st.columns(3)
        with col1:
            api_status = api_request("/health")
            is_online = api_status is not None and api_status.get("status") == "ok"
            st.metric("API Status", "🟢 Online" if is_online else "🔴 Offline")
        with col2:
            st.metric("Auth Status", "✅ Authenticated" if is_authenticated() else "❌ Not Authenticated")

        st.markdown("---")

        st.subheader("🔑 Token Information")
        if st.button("Show Token Info"):
            token_info = get_token_info()
            if token_info:
                st.json(token_info)
            else:
                st.info("No token information available")

    with tab2:
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

    with tab3:
        st.subheader("📇 FAISS Index Management")

        st.write("Build or rebuild the FAISS index for fast similarity search")

        col1, col2 = st.columns(2)
        with col1:
            dim = st.number_input("Embedding Dimensions", min_value=1, value=768, key="faiss_dim")
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


def main():
    """Main application entry point."""

    if not is_authenticated():
        show_auth_page()
    else:
        show_dashboard()


if __name__ == "__main__":
    main()
