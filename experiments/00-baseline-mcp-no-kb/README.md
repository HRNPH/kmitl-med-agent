# KMITL Medical Agent Experiment - Baseline MCP No KB

This experiment implements a single Microsoft AutoGen agent that connects to an Ollama LLM endpoint and an MCP server to process medical questions from the Thai healthcare system.

## Features

- **Single AutoGen Agent**: Uses Microsoft AutoGen framework with a medical assistant agent
- **Ollama LLM Integration**: Connects to local Ollama server for LLM inference
- **MCP Server Connection**: Connects to MCP server for external data/tools (no authentication required)
- **Thai Medical Context**: Specialized for Thai healthcare system questions
- **Jupyter Notebook Workflow**: Organized as interactive notebooks for experimentation

## Project Structure

```
00-baseline-mcp-no-kb/
├── notebooks/           # Jupyter notebooks for experimentation
│   ├── 01_setup_and_imports.ipynb
│   ├── 02_autogen_agent_setup.ipynb
│   ├── 03_mcp_integration.ipynb
│   └── 04_experiment_execution.ipynb
├── source/             # Source files (empty after conversion)
├── archive/            # Archived source files
├── env.example         # Environment configuration template
├── pyproject.toml      # Project dependencies
└── README.md          # This file
```

## Setup

### 1. Environment Configuration

Copy the environment template and configure your settings:

```bash
cp env.example .env
```

Edit `.env` with your configuration:

```env
# Ollama LLM Configuration
OLLAMA_API_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# MCP Server Configuration
KMITL_MCP_SERVER_URL=http://localhost:3000
```

### 2. Install Dependencies

Using uv (recommended):

```bash
uv sync
```

Or using pip:

```bash
pip install -r requirements.txt
```

### 3. Start Required Services

**Ollama Server:**

```bash
ollama serve
```

**MCP Server:**

```bash
# Start your MCP server on the configured URL
```

## Usage

### Running the Experiment

1. **Open Jupyter Lab or Notebook:**

   ```bash
   jupyter lab
   ```

2. **Navigate to the notebooks directory and run in order:**
   - `01_setup_and_imports.ipynb` - Environment setup and data loading
   - `02_autogen_agent_setup.ipynb` - AutoGen agent configuration
   - `03_mcp_integration.ipynb` - MCP server integration
   - `04_experiment_execution.ipynb` - Run the experiment

### Notebook Workflow

The experiment is organized into 4 sequential notebooks:

1. **Setup and Imports** (`01_setup_and_imports.ipynb`)

   - Environment configuration
   - Dependency installation
   - Test data loading
   - Configuration validation

2. **AutoGen Agent Setup** (`02_autogen_agent_setup.ipynb`)

   - AutoGen configuration with Ollama
   - Medical assistant agent creation
   - User proxy agent setup
   - Agent communication testing

3. **MCP Integration** (`03_mcp_integration.ipynb`)

   - MCP client implementation
   - Server connection setup
   - Tool integration
   - Connection testing

4. **Experiment Execution** (`04_experiment_execution.ipynb`)
   - Main experiment class
   - Question processing
   - Results collection
   - Output generation

## Experiment Details

### AutoGen Configuration

The agent uses the following AutoGen settings:

- Temperature: 0.7 (balanced creativity and accuracy)
- Max tokens: 2000 (sufficient for detailed responses)
- Model: Configurable via OLLAMA_MODEL environment variable
- API Base: Configurable via OLLAMA_API_URL environment variable

### Medical Assistant Agent

The medical assistant agent is specialized for Thai healthcare with knowledge of:

- Hospital departments and services
- Patient rights and insurance coverage
- Medical procedures and treatments
- Emergency protocols
- Healthcare policies and regulations

### MCP Integration

The experiment connects to an MCP server to access external tools and data:

- No authentication required
- Configurable server URL
- Tool discovery and execution
- Error handling for connection failures

## Data

The experiment processes questions from `data/test.csv` which contains:

- Medical questions in Thai
- Multiple choice answers
- Department classifications

## Output

The experiment generates:

- JSON results file with question-response pairs
- Console output with processing status
- Summary statistics (success rate, response quality)

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**

   - Ensure Ollama server is running: `ollama serve`
   - Check URL in `.env` file
   - Verify model is available: `ollama list`

2. **MCP Server Connection Failed**

   - Check MCP server URL in `.env`
   - Ensure MCP server is running
   - Experiment will continue without MCP if connection fails

3. **Import Errors**

   - Install dependencies: `uv sync` or `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

4. **Jupyter Issues**
   - Install jupyter: `pip install jupyter jupyterlab`
   - Start jupyter: `jupyter lab`

## Development Workflow

### Adding New Experiments

1. Create new source files in `source/` directory
2. Use IPython format with Jupytext headers
3. Convert to notebooks: `jupytext --to notebook source/*.py`
4. Move notebooks to `notebooks/` directory
5. Archive source files to `archive/` directory

### Modifying Existing Experiments

1. Edit notebooks directly in Jupyter
2. Convert back to source: `jupytext --to py notebooks/*.ipynb`
3. Move source files to `source/` for further development
4. Re-convert to notebooks when ready

## License

This experiment is part of the KMITL Medical Agent project.
