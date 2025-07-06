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
    per_page = 100  # trakt's max limit per page is 100
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
            if item.get("type") == "movie":
                movie = item.get("movie")
                movie_id = movie["ids"]["trakt"] if movie and movie.get("ids") else None
                if movie_id:
                    if movie_id not in seen_movies:
                        seen_movies[movie_id] = {"item": item, "count": 0, "earliest": item.get("watched_at"), "latest": item.get("watched_at")}
                    seen_movies[movie_id]["count"] += 1
                    # Update earliest and latest watched_at (ISO 8601 string comparison is safe)
                    watched_at = item.get("watched_at")
                    if watched_at:
                        if not seen_movies[movie_id]["earliest"] or watched_at < seen_movies[movie_id]["earliest"]:
                            seen_movies[movie_id]["earliest"] = watched_at
                        if not seen_movies[movie_id]["latest"] or watched_at > seen_movies[movie_id]["latest"]:
                            seen_movies[movie_id]["latest"] = watched_at
            elif item.get("type") == "episode":
                show = item.get("show")
                show_id = show["ids"]["trakt"] if show and show.get("ids") else None
                episode = item.get("episode")
                if show_id and episode and episode.get("season") is not None and episode.get("number") is not None:
                    ep_key = (episode["season"], episode["number"])
                    if show_id not in seen_shows:
                        seen_shows[show_id] = {"item": item, "episodes": {}, "episode_watch_total": 0, "earliest": item.get("watched_at"), "latest": item.get("watched_at")}
                    # Count watches per episode
                    seen_shows[show_id]["episodes"].setdefault(ep_key, 0)
                    seen_shows[show_id]["episodes"][ep_key] += 1
                    seen_shows[show_id]["episode_watch_total"] += 1  # Total episode watches including repeats
                    # Update earliest and latest watched_at
                    watched_at = item.get("watched_at")
                    if watched_at:
                        if not seen_shows[show_id]["earliest"] or watched_at < seen_shows[show_id]["earliest"]:
                            seen_shows[show_id]["earliest"] = watched_at
                        if not seen_shows[show_id]["latest"] or watched_at > seen_shows[show_id]["latest"]:
                            seen_shows[show_id]["latest"] = watched_at
        if len(page_data) < per_page:
            break  # last page
        page += 1
    # add watch_count (how many times the full show / movie was watched) and episode completion info to each show
    for movie in seen_movies.values():
        # Flatten all children of movie["item"]["movie"] to the parent level
        if "movie" in movie["item"] and isinstance(movie["item"]["movie"], dict):
            for k, v in movie["item"]["movie"].items():
                if k not in movie["item"]:
                    movie["item"][k] = v
            del movie["item"]["movie"]
        movie["item"]["watch_count"] = movie["count"]
        movie["item"]["earliest_watched_at"] = movie["earliest"]
        movie["item"]["latest_watched_at"] = movie["latest"]
        all_history.append(movie["item"])
    for show_id, show in seen_shows.items():
        show["item"]["type"] = "show"  # Ensure type is 'show' for show items
        # Flatten all children of show["item"]["show"] to the parent level
        if "show" in show["item"] and isinstance(show["item"]["show"], dict):
            for k, v in show["item"]["show"].items():
                if k not in show["item"]:
                    show["item"][k] = v
            del show["item"]["show"]
        # Remove the episode key and its children if present
        if "episode" in show["item"]:
            del show["item"]["episode"]
        tmdb_show_id = show["item"].get("ids", {}).get("tmdb")
        total_episodes = None
        if tmdb_show_id:
            try:
                meta = get_metadata(tmdb_show_id, media_type="tv")
                total_episodes = meta.get("number_of_episodes")
            except Exception as e:
                logger.warning(f"Could not fetch TMDB metadata for show {show_id}: {e}")
        watched_episodes = len(show["episodes"])
        show["item"]["watched_episodes"] = watched_episodes
        show["item"]["total_episodes"] = total_episodes
        show["item"]["completion_ratio"] = (watched_episodes / total_episodes) if total_episodes else None
        # Calculate watch_count (number of times all episodes were watched)
        if total_episodes and watched_episodes == total_episodes:
            watch_count = min(show["episodes"].values())
        else:
            watch_count = 0
        show["item"]["watch_count"] = watch_count
        show["item"]["episode_watch_count"] = show["episode_watch_total"]  # Total episode watches
        show["item"]["earliest_watched_at"] = show["earliest"]
        show["item"]["latest_watched_at"] = show["latest"]
        all_history.append(show["item"])
    logger.info(f"Total unique movies: {len(seen_movies)}, unique shows: {len(seen_shows)}")
    if all_history:
        # only update DB if something changed
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
