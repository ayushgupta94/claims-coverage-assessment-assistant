import pytest

from app.api.security import verify_api_key
from app.config import Settings
from app.core.exceptions import AuthenticationError


def make_settings(**overrides) -> Settings:
    return Settings(mongo_uri="mongodb://localhost:27017", openai_api_key="unused", **overrides)


def test_verify_api_key_accepts_matching_key():
    settings = make_settings(app_api_key="secret-key")
    verify_api_key(provided_key="secret-key", settings=settings)  # no raise


def test_verify_api_key_rejects_wrong_key():
    settings = make_settings(app_api_key="secret-key")
    with pytest.raises(AuthenticationError):
        verify_api_key(provided_key="wrong-key", settings=settings)


def test_verify_api_key_rejects_missing_header():
    settings = make_settings(app_api_key="secret-key")
    with pytest.raises(AuthenticationError):
        verify_api_key(provided_key=None, settings=settings)


def test_verify_api_key_fails_closed_when_unconfigured():
    settings = make_settings(app_api_key=None)
    with pytest.raises(AuthenticationError):
        verify_api_key(provided_key="anything", settings=settings)
