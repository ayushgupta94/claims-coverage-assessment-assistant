"""X-API-Key authentication for inbound requests.

A single static key (`settings.app_api_key`), not a provider credential --
operators generate their own (e.g. a UUID) and distribute it to clients.
Kept as its own dependency (rather than inline in routes) so it can be
attached per-router and unit-tested in isolation.
"""
from __future__ import annotations

import secrets

from fastapi import Depends, Security
from fastapi.security import APIKeyHeader

from app.api.deps import get_app_settings
from app.config import Settings
from app.core.exceptions import AuthenticationError

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    provided_key: str | None = Security(_api_key_header),
    settings: Settings = Depends(get_app_settings),
) -> None:
    if not settings.app_api_key or not provided_key or not secrets.compare_digest(provided_key, settings.app_api_key):
        raise AuthenticationError("Missing or invalid API key.")
