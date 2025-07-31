# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # KMITL Medical Agent Experiment - Setup and Imports
#
# This notebook sets up the environment and imports required libraries for the KMITL Medical Agent experiment.

# ## 1. Environment Setup

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path
current_dir = Path.cwd()
sys.path.insert(0, str(current_dir))

print("✓ Environment setup completed")

# ## 2. Install and Import Dependencies

# Install required packages if not already installed
try:
    import autogen

    print("✓ autogen already installed")
except ImportError:
    print("Installing autogen...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyautogen"])

try:
    import pandas as pd

    print("✓ pandas already installed")
except ImportError:
    print("Installing pandas...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])

# Import all required libraries
import asyncio
import json
from typing import Dict, Any, List, Optional
import autogen
from autogen import Agent, AssistantAgent, UserProxyAgent
import pandas as pd
import aiohttp

print("✓ All dependencies imported successfully")

# ## 3. Configuration

# Get environment variables
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
MCP_SERVER_URL = os.getenv("KMITL_MCP_SERVER_URL", "http://localhost:3000")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

print(f"Configuration:")
print(f"  Ollama URL: {OLLAMA_API_URL}")
print(f"  MCP Server URL: {MCP_SERVER_URL}")
print(f"  Model: {OLLAMA_MODEL}")

# ## 4. Load Test Data


def load_test_data():
    """Load test data from CSV file"""
    try:
        # Navigate to the data directory
        data_path = current_dir.parent.parent.parent / "data" / "test.csv"
        if data_path.exists():
            df = pd.read_csv(data_path)
            print(f"✓ Loaded {len(df)} test questions from {data_path}")
            return df
        else:
            print(f"✗ Test data not found at: {data_path}")
            return pd.DataFrame()
    except Exception as e:
        print(f"✗ Error loading test data: {e}")
        return pd.DataFrame()


# Load the test data
test_data = load_test_data()

if not test_data.empty:
    print(f"Sample questions:")
    for i, row in test_data.head(3).iterrows():
        question = (
            row.get("question", "")[:100] + "..."
            if len(row.get("question", "")) > 100
            else row.get("question", "")
        )
        print(f"  {i+1}. {question}")

print("\n✓ Setup and imports completed successfully!")
