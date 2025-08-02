# Iteration-1 FPI Endpoint

A FastAPI-based RAG (Retrieval-Augmented Generation) system with LangGraph and MCP (Model Context Protocol) integration for medical assistance.

## Features

- **FastAPI REST API** with automatic documentation
- **RAG System** using FAISS vector store and HuggingFace embeddings
- **LangGraph ReAct Agent** for intelligent query processing
- **MCP Integration** for external tool access
- **Ollama LLM** support for local model inference
- **Docker & Docker Compose** for easy deployment

## Quick Start with Docker

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available for Ollama models

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd iteration-1-fpi-endpoint

# Make the Docker script executable
chmod +x docker-scripts.sh
```

### 2. Start the Application

```bash
# Start all services (FastAPI + Ollama)
./docker-scripts.sh start
```

This will:
- Build the FastAPI application container
- Start the Ollama service for LLM inference
- Create necessary volumes and networks
- Start health checks

### 3. Access the Application

- **FastAPI Documentation**: http://localhost:8000/docs
- **API Endpoint**: http://localhost:8000/api/v1/query
- **Ollama API**: http://localhost:11434

### 4. Pull Required Models

```bash
# Pull the default model (qwen2:7b)
./docker-scripts.sh pull-model qwen2:7b

# Or pull a larger model for better performance
./docker-scripts.sh pull-model qwen2:14b
```

## Docker Management

### Available Commands

```bash
# Start the application
./docker-scripts.sh start

# Stop the application
./docker-scripts.sh stop

# Restart the application
./docker-scripts.sh restart

# View logs
./docker-scripts.sh logs
./docker-scripts.sh logs app      # FastAPI logs only
./docker-scripts.sh logs ollama   # Ollama logs only

# Check status
./docker-scripts.sh status

# Pull a specific model
./docker-scripts.sh pull-model <model_name>

# List available models
./docker-scripts.sh list-models

# Clean up everything (containers, images, volumes)
./docker-scripts.sh cleanup

# Show help
./docker-scripts.sh help
```

### Manual Docker Commands

If you prefer to use Docker commands directly:

```bash
# Build and start
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up --build --force-recreate -d
```

## API Usage

### Query Endpoint

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{
       "questions": [
         "What are the symptoms of diabetes?",
         "How to treat hypertension?"
       ]
     }'
```

### Response Format

```json
{
  "results": [
    {
      "question": "What are the symptoms of diabetes?",
      "answer": "ก",
      "took": 2.34,
      "agent_response": {...},
      "rag_tool_calls": [...],
      "mcp_tool_calls": [...]
    }
  ]
}
```

## Configuration

### Environment Variables

The application can be configured using environment variables:

- `OLLAMA_BASE_URL`: Ollama service URL (default: http://ollama:11434)
- `PYTHONPATH`: Python path (default: /app)

### Data Directory

Place your markdown files (`.mkd` extension) in the `./data/` directory. The application will automatically load and index these files for RAG functionality.

## Development

### Local Development Setup

```bash
# Install uv (Python package manager)
pip install uv

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

# Run the application
uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Adding Dependencies

```bash
# Add a new dependency
uv add fastapi

# Add development dependency
uv add --dev pytest
```

## Architecture

### Services

1. **FastAPI Application** (`app` service)
   - Main API server
   - RAG system with FAISS vector store
   - LangGraph ReAct agent
   - MCP client integration

2. **Ollama Service** (`ollama` service)
   - Local LLM inference
   - Model management
   - REST API for model access

### Network

- Services communicate via Docker network `app-network`
- FastAPI exposed on port 8000
- Ollama exposed on port 11434

### Volumes

- `./data:/app/data`: Mounts local data directory for markdown files
- `ollama_data`: Persistent storage for Ollama models

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce model size: `./docker-scripts.sh pull-model qwen2:7b`
   - Increase Docker memory limit in Docker Desktop settings

2. **Model Not Found**
   - Pull the required model: `./docker-scripts.sh pull-model <model_name>`
   - Check available models: `./docker-scripts.sh list-models`

3. **Service Not Starting**
   - Check logs: `./docker-scripts.sh logs`
   - Verify Docker is running
   - Check port availability

4. **Health Check Failures**
   - Wait for services to fully start (may take 1-2 minutes)
   - Check individual service logs

### Logs and Debugging

```bash
# View all logs
./docker-scripts.sh logs

# View specific service logs
./docker-scripts.sh logs app
./docker-scripts.sh logs ollama

# Check service status
./docker-scripts.sh status
```

## Performance Optimization

### For Production

1. **Use Larger Models**: Pull 14B or 32B models for better accuracy
2. **Increase Resources**: Allocate more CPU/memory to Docker
3. **Persistent Volumes**: Models are cached in `ollama_data` volume
4. **Load Balancing**: Consider using the included nginx configuration

### Resource Requirements

- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores
- **Production**: 16GB+ RAM, 8+ CPU cores

## Security Considerations

- Services run in isolated Docker containers
- No sensitive data in images (mounted as volumes)
- Health checks ensure service availability
- Network isolation via Docker networks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with Docker: `./docker-scripts.sh start`
5. Submit a pull request

## License

[Add your license information here]
