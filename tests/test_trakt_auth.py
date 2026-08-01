import unittest
from unittest.mock import patch

from app.auth import trakt_auth
from app.config.settings import settings


class DummyResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"access_token": "abc", "refresh_token": "def"}


class TraktAuthTests(unittest.TestCase):
    def test_refresh_token_includes_redirect_uri_in_payload(self):
        with patch.object(
            trakt_auth, "load_token_data", return_value={"refresh_token": "abc"}
        ), patch.object(trakt_auth, "save_token_data") as save_mock, patch.object(
            trakt_auth.requests, "post", return_value=DummyResponse()
        ) as post_mock:
            trakt_auth.refresh_token()

        payload = post_mock.call_args.kwargs["json"]
        self.assertEqual(payload["grant_type"], "refresh_token")
        self.assertEqual(payload["redirect_uri"], settings.TRAKT_REDIRECT_URI)
        save_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
