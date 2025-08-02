from ..deps.agent import rag
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import asyncio
from contextlib import asynccontextmanager

# Import your RAGLangGraphMCP and other code above here
# from your_module import RAGLangGraphMCP, operate

# ---------- Config / Initialization ----------
rag_system = rag  # Using your previously created `rag` instance

import re


def replacement(text: str):
    match = re.search(r"[กขคง]", text)
    try:
        if match:
            result = match.group(0)
            return result  # ค
        else:
            return ""
    except:
        print("Error:", text)
        return ""


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


class SingleQueryRequest(BaseModel):
    question: str


class QueryRequest(BaseModel):
    questions: List[str]


class QueryResponse(BaseModel):
    answer: str
    reason: str


class BatchQueryResponse(BaseModel):
    results: List[QueryResponse]


@app.get("/health")
async def health_check():
    return {"message": "OK!"}


@app.post("/", response_model=QueryResponse)
async def single_query_endpoint(payload: SingleQueryRequest):
    """
    Accept a single question and return result.
    Returns simplified schema with only answer and reason fields.
    """
    # Ensure the agent is initialized
    if rag_system.agent is None:
        await rag_system.create_agent()

    # Process single question
    result = await rag_system.query(payload.question)

    return QueryResponse(
        answer=result.get("answer", ""),
        reason=result.get("reason", "This question have no valid answer"),
    )


@app.post("/batch", response_model=BatchQueryResponse)
async def batch_query_endpoint(payload: QueryRequest):
    """
    Accept a list of questions and return results.
    Questions will be processed concurrently before sending back response.
    Returns simplified schema with only answer and reason fields.
    """
    # Ensure the agent is initialized
    if rag_system.agent is None:
        await rag_system.create_agent()

    # Process in parallel using existing query_batch
    results = await rag_system.query_batch(payload.questions, parallel=True)

    # Convert to the simplified response format
    simplified_results = []
    for result in results:
        simplified_results.append(
            QueryResponse(
                answer=replacement(result.get("answer", "Error")),
                reason=result.get("reason", "This question have no valid answer"),
            )
        )

    return BatchQueryResponse(results=simplified_results)


# Optional: for standalone running
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
