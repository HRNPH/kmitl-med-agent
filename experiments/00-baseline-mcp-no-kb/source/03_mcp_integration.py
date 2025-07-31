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
