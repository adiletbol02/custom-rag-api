import uvicorn
import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncAzureOpenAI  # Import Azure OpenAI client
from redis import Redis
from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex
from langchain_huggingface import HuggingFaceEmbeddings

from config import logger
from vectorstore import initialize_vectorstore
from api import register_api_routes

# --- CORS Configuration ---
CORS_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://open-webui:8080")
allowed_origins = [origin.strip() for origin in CORS_ORIGINS.split(',')]

# Initialize FastAPI app
app = FastAPI(
    title="Custom RAG API with Azure OpenAI",
    description="An API providing chat completions powered by a RAG pipeline using Azure OpenAI and Redis.",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"CORS enabled for the following origins: {allowed_origins}")

@app.on_event("startup")
async def startup_event():
    """
    Asynchronous tasks to be run when the application starts.
    Initializes the Azure OpenAI client, embeddings, vector store, and Redis cache index.
    """
    logger.info("Application startup...")

    # 1. Initialize Azure OpenAI client
    try:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        if not api_key or not endpoint:
            logger.error("Azure OpenAI credentials are missing.")
            raise ValueError("Failed to configure Azure OpenAI: credentials are missing.")
        app.state.openai_client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
        logger.info("Azure OpenAI client initialized successfully.")
    except Exception as e:
        logger.error(f"Azure OpenAI configuration error: {e}", exc_info=True)
        raise

    # 2. Initialize embeddings
    try:
        app.state.embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))
        logger.info("Embeddings initialized and attached to app state.")
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}", exc_info=True)
        app.state.embeddings = None

    # 3. Initialize the vector store
    logger.info("Initializing vectorstore...")
    app.state.vectorstore = await initialize_vectorstore()
    if app.state.vectorstore:
        logger.info("Vectorstore initialized and attached to app state.")
    else:
        logger.error("Failed to initialize vectorstore. The application might not function correctly.")

    # 4. Initialize Redis cache index for semantic caching
    try:
        redis_client = Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379"), decode_responses=True)
        cache_schema = IndexSchema.from_dict({
            "index": {
                "name": "response_cache",
                "prefix": "cache",
                "storage_type": "json",
            },
            "fields": [
                {"name": "query_text", "type": "text"},
                {"name": "response", "type": "text"},
                {
                    "name": "query_embedding",
                    "type": "vector",
                    "attrs": {
                        "algorithm": "hnsw",
                        "dims": len(app.state.embeddings.embed_query("test query")),
                        "distance_metric": "cosine",
                        "datatype": "float32"
                    }
                }
            ]
        })
        app.state.cache_index = SearchIndex(cache_schema, redis_client)
        if not app.state.cache_index.exists():
            app.state.cache_index.create()
            logger.info("Created new Redis cache index")
        else:
            logger.info("Connected to existing Redis cache index")
    except Exception as e:
        logger.error(f"Failed to initialize Redis cache index: {e}", exc_info=True)
        app.state.cache_index = None

# Register API routes
register_api_routes(app)

@app.get("/", tags=["Health Check"])
async def read_root():
    """
    Root endpoint providing a simple health check.
    """
    return {"status": "ok", "message": "Welcome to the Custom RAG API with Azure OpenAI"}

# Run the application using Uvicorn
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)