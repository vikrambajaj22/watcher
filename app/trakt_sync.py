import hashlib
import json
import os
import time

import requests

from app.auth.trakt_auth import refresh_token
from app.config.settings import settings
from app.dao.history import get_watch_history, store_watch_history
from app.tmdb_client import get_metadata
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _ensure_valid_token():
    """Refresh Trakt token if it's expired or about to expire."""
    token_file = ".env.trakt_token"
    if not os.path.exists(token_file):
        logger.warning("Token file not found. Sync will fail.")
        return

    try:
        with open(token_file) as f:
            data = json.load(f)

        created_at = data.get("created_at", 0)
        expires_in = data.get("expires_in", 604800)  # default 7 days
        current_time = int(time.time())
        token_age = current_time - created_at

        # refresh if token is expired or will expire within 1 hour (3600 seconds)
        if token_age > (expires_in - 3600):
            logger.info(
                "Token expired or expiring soon (age: %s seconds, expires_in: %s). Refreshing...",
                token_age,
                expires_in,
            )
            refresh_token()
            logger.info("Token refreshed successfully.")
    except Exception as e:
        logger.error("Error checking/refreshing token: %s", repr(e), exc_info=True)
        # continue anyway - let the API call fail if token is actually invalid


def sync_trakt_history():
    # ensure token is valid before syncing
    _ensure_valid_token()

    all_history = []
    seen_movies = {}
    seen_shows = {}
    unique = set()
    # --- Movies: use /sync/watched/movies for full watched list ---
    try:
        resp_movies = requests.get(
            settings.TRAKT_WATCHED_MOVIES_API_URL,
            headers=settings.trakt_headers,
        )
        logger.info(
            "Syncing Trakt watched movies: %s", resp_movies.status_code
        )
        if resp_movies.status_code == 200:
            movies_data = resp_movies.json() or []
            logger.info("Fetched %s watched movies from Trakt", len(movies_data))
            for item in movies_data:
                movie = item.get("movie") or {}
                ids = movie.get("ids") or {}
                tmdb_movie_id = ids.get("tmdb")
                if not tmdb_movie_id:
                    continue
                unique.add(("movie", tmdb_movie_id))
                watched_at = item.get("last_watched_at")
                plays = int(item.get("plays") or 0)
                movie_entry = seen_movies.setdefault(
                    tmdb_movie_id,
                    {
                        "item": item,
                        "count": 0,
                        "earliest": watched_at,
                        "latest": watched_at,
                    },
                )
                movie_entry["count"] += plays
                if watched_at:
                    if (
                        not movie_entry["earliest"]
                        or watched_at < movie_entry["earliest"]
                    ):
                        movie_entry["earliest"] = watched_at
                    if not movie_entry["latest"] or watched_at > movie_entry["latest"]:
                        movie_entry["latest"] = watched_at
        else:
            logger.warning(
                "Failed to fetch Trakt watched movies: %s", resp_movies.status_code
            )
    except Exception as e:
        logger.error("Error fetching Trakt watched movies: %s", repr(e), exc_info=True)

    # --- Shows: use /sync/watched/shows for full watched list + per-episode plays ---
    try:
        resp_shows = requests.get(
            settings.TRAKT_WATCHED_SHOWS_API_URL,
            headers=settings.trakt_headers,
        )
        logger.info(
            "Syncing Trakt watched shows: %s", resp_shows.status_code
        )
        if resp_shows.status_code == 200:
            shows_data = resp_shows.json() or []
            logger.info("Fetched %s watched shows from Trakt", len(shows_data))
            for item in shows_data:
                show = item.get("show") or {}
                ids = show.get("ids") or {}
                tmdb_show_id = ids.get("tmdb")
                if not tmdb_show_id:
                    continue
                unique.add(("tv", tmdb_show_id))

                # Build episode watch map from seasons/episodes
                episodes_map = {}
                episode_watch_total = 0
                earliest = None
                latest = None
                seasons = item.get("seasons") or []
                for season in seasons:
                    season_number = season.get("number")
                    if season_number is None:
                        continue
                    for ep in season.get("episodes") or []:
                        ep_number = ep.get("number")
                        if ep_number is None:
                            continue
                        ep_key = (season_number, ep_number)
                        plays = int(ep.get("plays") or 0)
                        episodes_map[ep_key] = episodes_map.get(ep_key, 0) + plays
                        episode_watch_total += plays
                        ep_last = ep.get("last_watched_at") or item.get(
                            "last_watched_at"
                        )
                        if ep_last:
                            if not earliest or ep_last < earliest:
                                earliest = ep_last
                            if not latest or ep_last > latest:
                                latest = ep_last

                if not episodes_map:
                    # no usable episode data; skip
                    continue

                show_entry = seen_shows.setdefault(
                    tmdb_show_id,
                    {
                        "item": item,
                        "episodes": {},
                        "episode_watch_total": 0,
                        "earliest": earliest,
                        "latest": latest,
                    },
                )
                # merge episodes and totals in case of duplicates
                for k, v in episodes_map.items():
                    show_entry["episodes"][k] = show_entry["episodes"].get(k, 0) + v
                show_entry["episode_watch_total"] += episode_watch_total
                if earliest:
                    if not show_entry["earliest"] or earliest < show_entry["earliest"]:
                        show_entry["earliest"] = earliest
                if latest:
                    if not show_entry["latest"] or latest > show_entry["latest"]:
                        show_entry["latest"] = latest
        else:
            logger.warning(
                "Failed to fetch Trakt watched shows: %s", resp_shows.status_code
            )
    except Exception as e:
        logger.error("Error fetching Trakt watched shows: %s", repr(e), exc_info=True)

    logger.info("Total unique items from Trakt watched lists: %s", len(unique))

    for tmdb_id, movie in seen_movies.items():
        item = movie["item"]
        movie_data = item.copy()
        if "movie" in movie_data and isinstance(movie_data["movie"], dict):
            for k, v in movie_data["movie"].items():
                if k not in movie_data:
                    movie_data[k] = v
            del movie_data["movie"]
        movie_data["media_type"] = "movie"
        movie_data["id"] = tmdb_id
        if "type" in movie_data:
            del movie_data["type"]
        movie_data["watch_count"] = movie["count"]
        movie_data["earliest_watched_at"] = movie["earliest"]
        movie_data["latest_watched_at"] = movie["latest"]
        # for movies, rewatch_engagement = watch_count (they're always 100% completion)
        movie_data["rewatch_engagement"] = float(movie["count"])
        all_history.append(movie_data)

    for tmdb_id, show in seen_shows.items():
        item = show["item"]
        show_data = item.copy()
        show_data["media_type"] = "tv"
        if "show" in show_data and isinstance(show_data["show"], dict):
            for k, v in show_data["show"].items():
                if k not in show_data:
                    show_data[k] = v
            del show_data["show"]
        if "episode" in show_data:
            del show_data["episode"]
        show_data["id"] = tmdb_id

        tmdb_show_id = show_data.get("ids", {}).get("tmdb")
        total_episodes = None
        season_episode_counts = {}  # {season: number_of_episodes}
        if tmdb_show_id:
            try:
                meta = get_metadata(tmdb_show_id, media_type="tv")
                total_episodes = meta.get("number_of_episodes")
                # build season_episode_counts from TMDB metadata if available
                if meta.get("seasons"):
                    for season in meta["seasons"]:
                        if (
                            season.get("season_number") is not None
                            and season.get("episode_count") is not None
                        ):
                            season_episode_counts[season["season_number"]] = season[
                                "episode_count"
                            ]
            except Exception as e:
                logger.warning(
                    "Could not fetch TMDB metadata for show %s: %s",
                    tmdb_id,
                    repr(e),
                    exc_info=True,
                )
        watched_episodes = len(show["episodes"])
        show_data["watched_episodes"] = watched_episodes
        show_data["total_episodes"] = total_episodes
        # calculate show completion ratio: unique episodes watched / total episodes
        show_data["completion_ratio"] = (
            watched_episodes / total_episodes if total_episodes else 0.0
        )
        # calculate watch_count (number of times all episodes were watched)
        if total_episodes and watched_episodes == total_episodes:
            watch_count = min(show["episodes"].values())
        else:
            watch_count = 0
        show_data["watch_count"] = watch_count
        show_data["episode_watch_count"] = show["episode_watch_total"]
        show_data["earliest_watched_at"] = show["earliest"]
        show_data["latest_watched_at"] = show["latest"]
        # rewatch_engagement for TV shows
        # = average_watches_per_episode * completion_ratio
        # this represents the effective engagement multiplier
        if watched_episodes > 0:
            avg_watches_per_episode = show["episode_watch_total"] / watched_episodes
            rewatch_engagement = avg_watches_per_episode * show_data["completion_ratio"]
        else:
            rewatch_engagement = 0.0
        show_data["rewatch_engagement"] = rewatch_engagement
        # calculate season completion counts
        season_completion_count = {}
        # build a mapping: season -> [episode watch counts]
        season_episode_watches = {}
        for (season, episode), count in show["episodes"].items():
            season_episode_watches.setdefault(str(season), []).append(count)
        for season, episode_counts in season_episode_watches.items():
            num_episodes = season_episode_counts.get(int(season))
            if num_episodes:
                episodes_watched = len(episode_counts)
                min_watch_count = min(episode_counts) if episode_counts else 0
                # calculate completion ratio: unique episodes watched / total episodes
                completion_ratio = (
                    episodes_watched / num_episodes if num_episodes else 0.0
                )
                # calculate average watch count for the season
                avg_watch_count = (
                    sum(episode_counts) / num_episodes if num_episodes else 0.0
                )
                entry = {
                    "episodes_watched": episodes_watched,
                    "total_episodes": num_episodes,
                    "min_watch_count": min_watch_count,
                    "completion_ratio": completion_ratio,
                    "avg_watch_count": avg_watch_count,
                }
                entry["partial"] = episodes_watched == num_episodes
                season_completion_count[str(season)] = entry
        show_data["media_type"] = "tv"
        if "type" in show_data:
            del show_data["type"]
        show_data["season_completion_count"] = season_completion_count
        all_history.append(show_data)

    logger.info(
        "Total unique movies: %s, unique shows: %s", len(seen_movies), len(seen_shows)
    )
    if not all_history:
        return
    current_history = get_watch_history()
    if not current_history:
        logger.info("No existing watch history found, updating database.")
        store_watch_history(all_history)
        return

    def hash_history(hist):
        def item_key(item):
            if item.get("media_type") == "movie":
                return (
                    "movie",
                    item.get("id"),
                    item.get("watch_count"),
                    item.get("rewatch_engagement"),
                    item.get("earliest_watched_at"),
                    item.get("latest_watched_at"),
                )
            else:
                # for shows, include season_completion_count as a sorted tuple for hash stability
                season_completion = item.get("season_completion_count", {})
                # convert to sorted tuple of (season, tuple(sorted(entry.items())))
                season_completion_tuple = tuple(
                    sorted(
                        (season, tuple(sorted(entry.items())))
                        for season, entry in season_completion.items()
                    )
                )
                return (
                    "tv",
                    item.get("id"),
                    item.get("watch_count"),
                    item.get("rewatch_engagement"),
                    item.get("episode_watch_count"),
                    item.get("watched_episodes"),
                    item.get("total_episodes"),
                    item.get("earliest_watched_at"),
                    item.get("latest_watched_at"),
                    season_completion_tuple,
                )

        sorted_hist = sorted(hist, key=item_key)
        return hashlib.sha256(
            json.dumps(
                [item_key(item) for item in sorted_hist], sort_keys=True
            ).encode()
        ).hexdigest()

    if hash_history(all_history) != hash_history(current_history):
        logger.info("Watch history changed, updating database.")
        store_watch_history(all_history)
    else:
        logger.info("Watch history unchanged, not updating database.")
