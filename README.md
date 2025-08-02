# Overview
# RAGLangGraphMCP Agent

This project implements **RAGLangGraphMCP**, an AI agent designed for **multiple-choice question answering (MCQ)** with medical-focused knowledge bases.  
It combines **RAG (Retrieval-Augmented Generation)** with **MCP (Multi-Server Tooling)** using **LangGraph** and **LangChain**.

---

## Features

### 1. **Hybrid Architecture**
- **RAG (Retrieval-Augmented Generation)**
  - Builds a **local document database** using FAISS for similarity search.
  - Retrieves relevant text chunks to help answer questions.
- **MCP (Multi-Server Client Protocol)**
  - Integrates external or local tool servers for **real-time / dynamic data**.

### 2. **LangGraph ReAct Agent**
- Uses the **ReAct pattern** (Reason + Act) so the agent:
  1. Searches documents (RAG)
  2. Decides whether to use external tools (MCP)
  3. Synthesizes an answer

### 3. **Tool Priority Protocol**
- Always calls **RAG first** (vector_search_tool).
- If keywords indicate real-time/dynamic needs, it **adds MCP calls**.

### 4. **Multiple Choice Question Mode**
- Strict output format:
  - **Answer:** ก / ข / ค / ง  
  - **Reason:** concise reasoning
- Will always select a **single best answer**, even if uncertain.

### 5. **Performance Optimized**
- Uses **VLLM runtime** by default for fast inference (Ollama optional).
- Batch processing support for multiple questions.

---

## How It Works

1. **Data Preparation**
   - Markdown files in `./data/` are indexed with FAISS.
2. **Agent Initialization**
   - MCP servers are registered (e.g., hospital APIs).
3. **Query Handling**
   - Agent retrieves data from vector DB.
   - If required, it calls MCP tools (multi-server) for live data.
4. **Response Generation**
   - Outputs a multiple-choice answer with reasoning.

---

## Key Components

- **FAISS Vector Index:** For semantic document retrieval.
- **MCP Client:** Connects to one or more tool servers.
- **LangGraph ReAct Agent:** Orchestrates reasoning and actions.
- **ChatOpenAI / ChatOllama:** Open-source model backends (Qwen3-32B recommended).

---

## Example Workflow

```python
# Initialize system
rag = RAGLangGraphMCP(use_ollama=False)

# Build or load vector index
rag.build_vector_index("./data/")

# Ask a question
result = await rag.query("ผู้ป่วยมีอาการปวดหัวอย่างรุนแรงและความดันสูง ควรทำอย่างไร?")
print(result)

# Ollama

- secured by vpn, assume connected
- http://172.16.30.137:11434

## to add more model

```
ssh siamai@172.16.30.137
password see in discord
# then
ollama run model-id # see in ollama website explore
# run bye command after chat ui opened, now u can call the model with the same ui passed in
/bye
```

# MCP Server

```
# https://mcp-hackathon.cmkl.ai/mcp
# run mcpo proxy server

```

# Shortcut

```
cd ui-based-experiment && docker compose -f ./docker-compose.ui.yaml up
```
