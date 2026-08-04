"""Mongo client lifecycle.

Wraps Motor's AsyncIOMotorClient so the rest of the app depends on a small,
swappable interface rather than importing motor directly everywhere. This is
what points at Azure Cosmos DB for MongoDB (vCore) in production and at a
MongoDB Atlas free-tier cluster (or local mongod) in development — the same
wire protocol means no code changes between environments, only the
connection string.
"""
from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import Settings


class MongoDatabase:
    """Thin holder around a Motor client + selected database."""

    def __init__(self, settings: Settings) -> None:
        self._client: AsyncIOMotorClient = AsyncIOMotorClient(settings.mongo_uri)
        self._db: AsyncIOMotorDatabase = self._client[settings.mongo_db_name]

    @property
    def db(self) -> AsyncIOMotorDatabase:
        return self._db

    async def ping(self) -> bool:
        try:
            await self._client.admin.command("ping")
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()


_instance: MongoDatabase | None = None


def get_mongo_database(settings: Settings) -> MongoDatabase:
    """Process-wide singleton so we reuse one connection pool."""
    global _instance
    if _instance is None:
        _instance = MongoDatabase(settings)
    return _instance


def reset_mongo_database_for_tests() -> None:
    """Test-only helper to clear the singleton between test modules."""
    global _instance
    _instance = None
