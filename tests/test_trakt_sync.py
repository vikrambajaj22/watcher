import unittest
from unittest.mock import patch

from app import trakt_sync


class DummyResponse:
    def __init__(self, payload, headers=None):
        self._payload = payload
        self.headers = headers or {"X-Pagination-Page-Count": "1"}
        self.status_code = 200

    def json(self):
        return self._payload


class TraktSyncTests(unittest.TestCase):
    def test_sync_passes_integer_tmdb_ids_to_watchlist_cleanup(self):
        movies_payload = [
            {
                "movie": {"ids": {"tmdb": "101"}},
                "last_watched_at": "2024-01-01T00:00:00Z",
                "plays": 1,
            }
        ]
        shows_payload = [
            {
                "show": {"ids": {"tmdb": "202"}},
                "seasons": [
                    {
                        "number": 1,
                        "episodes": [{"number": 1, "plays": 1, "last_watched_at": "2024-01-01T00:00:00Z"}],
                    }
                ],
                "last_watched_at": "2024-01-01T00:00:00Z",
            }
        ]

        with patch.object(
            trakt_sync.requests,
            "get",
            side_effect=[DummyResponse(movies_payload), DummyResponse(shows_payload)],
        ), patch.object(trakt_sync, "get_watch_history", return_value=[{"id": 1, "media_type": "movie"}]), patch.object(trakt_sync, "store_watch_history"), patch.object(trakt_sync, "get_metadata", return_value={}), patch.object(trakt_sync, "clear_watchlist_items_in_history") as cleanup_mock, patch("app.watchlist_sync.remove_from_watchlist"):
            trakt_sync.sync_trakt_history()

        watched_ids = cleanup_mock.call_args.args[0]
        self.assertEqual(watched_ids["movie"], {101})
        self.assertEqual(watched_ids["tv"], {202})


if __name__ == "__main__":
    unittest.main()
