from ..deps.agent import rag
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
import asyncio
from contextlib import asynccontextmanager

# Import your RAGLangGraphMCP and other code above here
# from your_module import RAGLangGraphMCP, operate

# ---------- Config / Initialization ----------
rag_system = rag  # Using your previously created `rag` instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await rag_system.create_agent()
    yield
    # Shutdown
    pass


# ---------- FastAPI App ----------
app = FastAPI(
    title="RAG LangGraph MCP API",
    version="1.0",
    description="API for querying the RAGLangGraphMCP system.",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    questions: List[str]


@app.get("/")
async def root():
    return {"message": "OK!"}


@app.post("/api/v1/query")
async def query_endpoint(payload: QueryRequest):
    """
    Accept a list of questions and return results.
    Questions will be processed concurrently before sending back response.
    """
    # Ensure the agent is initialized
    if rag_system.agent is None:
        await rag_system.create_agent()

    # Process in parallel using existing query_batch
    results = await rag_system.query_batch(payload.questions, parallel=True)
    return {"results": results}


# Optional: for standalone running
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
