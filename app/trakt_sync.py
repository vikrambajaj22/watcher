import hashlib
import json
import requests
from app.config.settings import settings
from app.dao.history import get_watch_history, store_watch_history
from app.tmdb_client import get_metadata
from app.utils.logger import get_logger

logger = get_logger(__name__)


def sync_trakt_history():
    all_history = []
    seen_movies = {}
    seen_shows = {}
    page = 1
    per_page = 100  # Trakt's max limit per page is 100
    while True:
        params = {"page": page, "limit": per_page}
        response = requests.get(settings.TRAKT_HISTORY_API_URL, headers=settings.trakt_headers, params=params)
        logger.info(f"Syncing Trakt history: {response.status_code} (page {page})")
        if response.status_code != 200:
            break
        page_data = response.json()
        logger.info(f"Fetched {len(page_data)} items on page {page}")
        if not page_data:
            break
        for item in page_data:
            item_type = item.get("type")
            watched_at = item.get("watched_at")
            if item_type == "movie":
                movie = item.get("movie")
                movie_id = movie["ids"]["trakt"] if movie and movie.get("ids") else None
                if not movie_id:
                    continue
                movie_entry = seen_movies.setdefault(
                    movie_id,
                    {"item": item, "count": 0, "earliest": watched_at, "latest": watched_at}
                )
                movie_entry["count"] += 1
                # Update earliest and latest watched_at
                if watched_at:
                    if not movie_entry["earliest"] or watched_at < movie_entry["earliest"]:
                        movie_entry["earliest"] = watched_at
                    if not movie_entry["latest"] or watched_at > movie_entry["latest"]:
                        movie_entry["latest"] = watched_at
            elif item_type == "episode":
                show = item.get("show")
                show_id = show["ids"]["trakt"] if show and show.get("ids") else None
                episode = item.get("episode")
                if not (show_id and episode and episode.get("season") is not None and episode.get("number") is not None):
                    continue
                ep_key = (episode["season"], episode["number"])
                show_entry = seen_shows.setdefault(
                    show_id,
                    {"item": item, "episodes": {}, "episode_watch_total": 0, "earliest": watched_at, "latest": watched_at}
                )
                show_entry["episodes"].setdefault(ep_key, 0)
                show_entry["episodes"][ep_key] += 1
                show_entry["episode_watch_total"] += 1
                # Update earliest and latest watched_at
                if watched_at:
                    if not show_entry["earliest"] or watched_at < show_entry["earliest"]:
                        show_entry["earliest"] = watched_at
                    if not show_entry["latest"] or watched_at > show_entry["latest"]:
                        show_entry["latest"] = watched_at
        if len(page_data) < per_page:
            break
        page += 1

    # Process movies
    for movie in seen_movies.values():
        item = movie["item"]
        movie_data = item.copy()
        if "movie" in movie_data and isinstance(movie_data["movie"], dict):
            for k, v in movie_data["movie"].items():
                if k not in movie_data:
                    movie_data[k] = v
            del movie_data["movie"]
        movie_data["watch_count"] = movie["count"]
        movie_data["earliest_watched_at"] = movie["earliest"]
        movie_data["latest_watched_at"] = movie["latest"]
        all_history.append(movie_data)

    # Process shows
    for show_id, show in seen_shows.items():
        item = show["item"]
        show_data = item.copy()
        show_data["type"] = "show"
        if "show" in show_data and isinstance(show_data["show"], dict):
            for k, v in show_data["show"].items():
                if k not in show_data:
                    show_data[k] = v
            del show_data["show"]
        if "episode" in show_data:
            del show_data["episode"]
        tmdb_show_id = show_data.get("ids", {}).get("tmdb")
        total_episodes = None
        if tmdb_show_id:
            try:
                meta = get_metadata(tmdb_show_id, media_type="tv")
                total_episodes = meta.get("number_of_episodes")
            except Exception as e:
                logger.warning(f"Could not fetch TMDB metadata for show {show_id}: {e}")
        watched_episodes = len(show["episodes"])
        show_data["watched_episodes"] = watched_episodes
        show_data["total_episodes"] = total_episodes
        show_data["completion_ratio"] = (watched_episodes / total_episodes) if total_episodes else None
        if total_episodes and watched_episodes == total_episodes:
            watch_count = min(show["episodes"].values())
        else:
            watch_count = 0
        show_data["watch_count"] = watch_count
        show_data["episode_watch_count"] = show["episode_watch_total"]
        show_data["earliest_watched_at"] = show["earliest"]
        show_data["latest_watched_at"] = show["latest"]
        all_history.append(show_data)

    logger.info(f"Total unique movies: {len(seen_movies)}, unique shows: {len(seen_shows)}")
    if not all_history:
        return
    current_history = get_watch_history()
    if not current_history:
        logger.info("No existing watch history found, updating database.")
        store_watch_history(all_history)
        return
    def hash_history(hist):
        def item_key(item):
            if item.get("type") == "movie":
                return (
                    "movie",
                    item.get("ids", {}).get("trakt"),
                    item.get("watch_count"),
                    item.get("earliest_watched_at"),
                    item.get("latest_watched_at"),
                )
            else:
                return (
                    "show",
                    item.get("ids", {}).get("trakt"),
                    item.get("watch_count"),
                    item.get("episode_watch_count"),
                    item.get("watched_episodes"),
                    item.get("total_episodes"),
                    item.get("earliest_watched_at"),
                    item.get("latest_watched_at"),
                )
        sorted_hist = sorted(hist, key=item_key)
        return hashlib.sha256(json.dumps([item_key(item) for item in sorted_hist], sort_keys=True).encode()).hexdigest()

    if hash_history(all_history) != hash_history(current_history):
        logger.info("Watch history changed, updating database.")
        store_watch_history(all_history)
    else:
        logger.info("Watch history unchanged, not updating database.")
