"""Application entrypoint.

One process, one container: this creates the FastAPI app, wires up Mongo on
startup, mounts the MCP server at /mcp (see app/mcp_server/server.py), and
registers the claims + health routes. Run locally with:

    uvicorn app.main:app --reload

or in Docker via the CMD in docker/Dockerfile -- same command either way.
"""
from __future__ import annotations
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.deps import get_app_settings
from app.api.routes import claims, health
from app.config import Settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.db.mongo_client import get_mongo_database
from app.db.repositories.claim_repository import ClaimRepository
from app.db.repositories.policy_repository import PolicyRepository
from app.mcp_server.server import build_mcp_server
from app.rag.embeddings import get_embedder
from app.rag.retriever import ClauseRetriever

logger = get_logger(__name__)


def create_app(settings: Settings | None = None, mongo_database=None) -> FastAPI:
    """Builds the FastAPI app. `mongo_database` can be injected (e.g. with a
    mongomock-backed instance) for smoke-testing the full app without a real
    MongoDB connection; production and normal local runs leave it as None
    and get a real Motor-backed connection to `settings.mongo_uri`."""
    settings = settings or get_app_settings()
    configure_logging(settings.log_level)

    mongo_database = mongo_database or get_mongo_database(settings)
    policy_repository = PolicyRepository(mongo_database.db)
    claim_repository = ClaimRepository(mongo_database.db)
    retriever = ClauseRetriever(policy_repository, get_embedder(settings), settings)

    mcp_server = build_mcp_server(
        policy_repository=policy_repository,
        claim_repository=claim_repository,
        retriever=retriever,
        settings=settings,
    )
    mcp_asgi_app = mcp_server.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.mongo_database = mongo_database
        await policy_repository.ensure_indexes()
        await claim_repository.ensure_indexes()
        logger.info("Application startup complete", extra={"ctx_environment": settings.environment})
        async with mcp_server.session_manager.run():
            yield
        mongo_database.close()
        logger.info("Application shutdown complete")

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Reviews insurance claims against policy documents and returns a "
            "structured coverage decision, via a tool-calling agent backed by "
            "RAG, MCP-exposed tools, and a deterministic rules/fraud layer."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/", include_in_schema=False)
    def root():
        return FileResponse("app/static/index.html")

    app.include_router(health.router)
    app.include_router(claims.router)
    app.mount("/mcp", mcp_asgi_app)

    register_exception_handlers(app)

    return app


app = create_app()
