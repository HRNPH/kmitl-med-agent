# UI-Based Experiment with MCP Wrapper

This directory contains a Docker setup for running the MCP wrapper alongside the Open WebUI interface.

## Services

### MCP Wrapper

- **Service Name**: `mcp-wrapper`
- **Purpose**: Connects to the remote MCP server at https://mcp-hackathon.cmkl.ai/mcp
- **Container**: Built from local Dockerfile
- **Features**: Handles MCP protocol communication via stdio

### Open WebUI

- **Service Name**: `open-webui`
- **Purpose**: Web interface for AI interactions
- **Port**: 3000 (mapped to container port 8080)
- **Features**: Full web UI with authentication

## Running the Setup

### Build and Start Services

```bash
# Build the wrapper image and start all services
docker-compose -f docker-compose.ui.yaml up --build

# Run in detached mode
docker-compose -f docker-compose.ui.yaml up -d --build
```

### Access the UI

- Open your browser and navigate to: http://localhost:3000
- Default credentials:
  - Username: `admin`
  - Password: `admin`

### View Logs

```bash
# View all service logs
docker-compose -f docker-compose.ui.yaml logs

# View specific service logs
docker-compose -f docker-compose.ui.yaml logs mcp-wrapper
docker-compose -f docker-compose.ui.yaml logs open-webui
```

### Stop Services

```bash
docker-compose -f docker-compose.ui.yaml down
```

## Development

### Rebuilding the Wrapper

If you modify the wrapper code, rebuild the image:

```bash
docker-compose -f docker-compose.ui.yaml build mcp-wrapper
docker-compose -f docker-compose.ui.yaml up -d
```

### Testing the Wrapper

You can test the wrapper directly:

```bash
# Build the wrapper image
docker build -t mcp-wrapper .

# Run the wrapper interactively
docker run -it --rm mcp-wrapper
```

## Network Configuration

Both services are connected to the `open-webui-network` bridge network, allowing them to communicate with each other if needed.

## Dependencies

The wrapper uses the following Python dependencies (from `pyproject.toml`):

- `fastapi>=0.116.1`
- `httpx>=0.28.1`
- `uvicorn>=0.35.0`
