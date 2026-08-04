"""Health check route -- used by container orchestrators (Azure Container
Apps liveness/readiness probes) to determine whether the service and its
database connection are healthy."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict:
    mongo_ok = await request.app.state.mongo_database.ping()
    return {
        "status": "ok" if mongo_ok else "degraded",
        "mongo_connected": mongo_ok,
    }
