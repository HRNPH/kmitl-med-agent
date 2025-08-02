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

# # MCP Server Integration
#
# This notebook implements the Model Context Protocol (MCP) server integration for the KMITL Medical Agent experiment.

# ## 1. MCP Client Implementation


class MCPClient:
    """Simple MCP client for connecting to MCP server"""

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session = None

    async def connect(self) -> bool:
        """Connect to MCP server"""
        try:
            self.session = aiohttp.ClientSession()

            # Test connection
            async with self.session.get(f"{self.server_url}/health") as response:
                if response.status == 200:
                    print(f"✓ Connected to MCP server at {self.server_url}")
                    return True
                else:
                    print(f"✗ MCP server returned status {response.status}")
                    return False

        except Exception as e:
            print(f"✗ Failed to connect to MCP server: {e}")
            return False

    async def disconnect(self):
        """Disconnect from MCP server"""
        if self.session:
            await self.session.close()
            self.session = None

    async def call_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        if not self.session:
            raise Exception("Not connected to MCP server")

        try:
            payload = {"tool": tool_name, "parameters": parameters}

            async with self.session.post(
                f"{self.server_url}/tools/call", json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"MCP tool call failed: {response.status}")

        except Exception as e:
            print(f"Error calling MCP tool {tool_name}: {e}")
            return {"error": str(e)}

    async def get_available_tools(self) -> list:
        """Get list of available tools from MCP server"""
        if not self.session:
            raise Exception("Not connected to MCP server")

        try:
            async with self.session.get(f"{self.server_url}/tools/list") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to get tools list: {response.status}")

        except Exception as e:
            print(f"Error getting available tools: {e}")
            return []


# ## 2. MCP-Enhanced Agent


class MCPEnhancedAgent:
    """Agent enhanced with MCP capabilities"""

    def __init__(self, mcp_url: str, autogen_agents: Dict[str, Any]):
        self.mcp_client = MCPClient(mcp_url)
        self.agents = autogen_agents
        self.connected = False

    async def setup(self):
        """Setup MCP connection"""
        self.connected = await self.mcp_client.connect()
        if self.connected:
            tools = await self.mcp_client.get_available_tools()
            print(f"Available MCP tools: {tools}")

    async def process_with_mcp(self, question: str) -> str:
        """Process question using MCP tools"""
        if not self.connected:
            return f"Error: MCP not connected. Question: {question}"

        try:
            # Example: Use MCP tool to get medical information
            result = await self.mcp_client.call_tool(
                "get_medical_info", {"query": question}
            )

            if "error" in result:
                return f"Error getting medical info: {result['error']}"
            else:
                return result.get("response", "No response from MCP tool")

        except Exception as e:
            return f"Error processing with MCP: {e}"

    async def cleanup(self):
        """Cleanup MCP connection"""
        await self.mcp_client.disconnect()


# ## 3. Initialize MCP Integration

# Create MCP client
mcp_client = MCPClient(MCP_SERVER_URL)

# Create MCP-enhanced agent
mcp_agent = MCPEnhancedAgent(MCP_SERVER_URL, agents)

print("✓ MCP integration setup completed")

# ## 4. Test MCP Connection


async def test_mcp_connection():
    """Test MCP server connection"""
    print("\nTesting MCP connection...")

    try:
        # Test connection
        connected = await mcp_client.connect()

        if connected:
            # Get available tools
            tools = await mcp_client.get_available_tools()
            print(f"Available tools: {tools}")

            # Test tool call
            if tools:
                test_result = await mcp_client.call_tool(
                    tools[0], {"test": "parameter"}  # Use first available tool
                )
                print(f"Test tool call result: {test_result}")

            # Cleanup
            await mcp_client.disconnect()
            return True
        else:
            print("✗ MCP connection failed")
            return False

    except Exception as e:
        print(f"✗ MCP test failed: {e}")
        return False


# Test MCP connection (commented out to avoid blocking)
# await test_mcp_connection()

print("✓ MCP integration ready!")

# # Experiment Execution
#
# This notebook runs the KMITL Medical Agent experiment with AutoGen agents and MCP integration.

# ## 1. Main Experiment Class


class KMITLMedicalAgent:
    """Main experiment class for KMITL Medical Agent"""

    def __init__(self, test_data, agents, mcp_agent):
        self.test_data = test_data
        self.agents = agents
        self.mcp_agent = mcp_agent
        self.results = []

    async def process_question(
        self, question: str, question_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process a single medical question"""
        try:
            # Start conversation between agents
            user_proxy = self.agents["user_proxy"]
            medical_assistant = self.agents["medical_assistant"]

            # Create the conversation
            chat_history = await user_proxy.a_initiate_chat(
                medical_assistant,
                message=f"""
                Question ID: {question_id if question_id else 'N/A'}
                Question: {question}
                
                Please provide a comprehensive answer to this medical question.
                Consider the Thai healthcare context and provide practical information.
                """,
                max_turns=5,
            )

            # Extract the response
            response = (
                chat_history[-1]["content"] if chat_history else "No response generated"
            )

            return {
                "question_id": question_id,
                "question": question,
                "response": response,
                "status": "success",
            }

        except Exception as e:
            return {
                "question_id": question_id,
                "question": question,
                "response": f"Error processing question: {str(e)}",
                "status": "error",
            }

    async def run_experiment(self, num_questions: int = 5) -> List[Dict[str, Any]]:
        """Run the experiment with test questions"""
        print(f"Starting KMITL Medical Agent Experiment")
        print(f"Processing {num_questions} questions...")

        # Setup MCP connection
        await self.mcp_agent.setup()

        # Process test questions
        questions_to_process = self.test_data.head(num_questions)

        for idx, row in questions_to_process.iterrows():
            question_id = row.get("id", idx + 1)
            question = row.get("question", "")

            print(f"\nProcessing Question {question_id}: {question[:100]}...")

            result = await self.process_question(question, question_id)
            self.results.append(result)

            # Add delay to avoid overwhelming the LLM
            await asyncio.sleep(1)

        # Cleanup MCP connection
        await self.mcp_agent.cleanup()

        return self.results

    def save_results(self, filename: str = "experiment_results.json"):
        """Save experiment results to file"""
        output_path = Path.cwd() / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        print(f"Results saved to: {output_path}")

    def print_results_summary(self):
        """Print a summary of the experiment results"""
        print("\n" + "=" * 50)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 50)

        successful = sum(1 for r in self.results if r["status"] == "success")
        failed = sum(1 for r in self.results if r["status"] == "error")

        print(f"Total Questions Processed: {len(self.results)}")
        print(f"Successful Responses: {successful}")
        print(f"Failed Responses: {failed}")
        print(
            f"Success Rate: {(successful/len(self.results)*100):.1f}%"
            if self.results
            else "0%"
        )

        # Show sample responses
        print("\nSample Responses:")
        for i, result in enumerate(self.results[:3]):  # Show first 3
            print(f"\nQuestion {i+1}: {result['question'][:100]}...")
            print(f"Response: {result['response'][:200]}...")


# ## 2. Initialize Experiment

# Create the main experiment agent
experiment_agent = KMITLMedicalAgent(test_data, agents, mcp_agent)

print("✓ Experiment agent initialized")

# ## 3. Run Experiment


async def run_experiment(num_questions: int = 5):
    """Run the experiment"""
    print(f"\nRunning experiment with {num_questions} questions...")

    # Run the experiment
    results = await experiment_agent.run_experiment(num_questions=num_questions)

    # Save and display results
    experiment_agent.save_results()
    experiment_agent.print_results_summary()

    return results


# ## 4. Execute Experiment

# Uncomment the line below to run the experiment
# results = await run_experiment(num_questions=3)

print("✓ Experiment execution setup completed!")
print("To run the experiment, uncomment the last line in this cell")
