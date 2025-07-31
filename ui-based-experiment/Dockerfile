FROM python:3.11-slim

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install uv and dependencies
RUN pip install uv && \
    uv sync --frozen

# Copy the wrapper code
COPY wrapper/ ./wrapper/

# Set the entry point to run the wrapper
ENTRYPOINT ["uvx", "mcpo", "--", "python", "./wrapper/remote_mcp_wrapper.py"]