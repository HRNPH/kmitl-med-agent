#!/usr/bin/env python3
"""
Remote MCP Server Wrapper for MCPI
Connects to https://mcp-hackathon.cmkl.ai/mcp
"""

import asyncio
import json
import sys
import httpx
from typing import Any, Dict, List, Optional


class RemoteMCPWrapper:
    def __init__(self):
        self.base_url = "https://mcp-hackathon.cmkl.ai/mcp"
        self.session_id: Optional[str] = None
        self.headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": "MCPI-Remote-Wrapper/1.0",
        }
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            headers=self.headers,
            follow_redirects=True
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.client:
            await self.client.aclose()

    async def initialize_session(self) -> bool:
        """Initialize session with the remote MCP server"""
        try:
            if not self.client:
                self.client = httpx.AsyncClient(
                    timeout=httpx.Timeout(30.0),
                    headers=self.headers,
                    follow_redirects=True
                )

            # Try to establish initial connection
            response = await self.client.get(self.base_url)
            
            # Check for session ID in headers
            if "mcp-session-id" in response.headers:
                self.session_id = response.headers["mcp-session-id"]
                print(f"Session initialized: {self.session_id}", file=sys.stderr)
                return True
            
            # If no session ID in headers, try to get it from response
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "session_id" in data:
                        self.session_id = data["session_id"]
                        print(f"Session initialized: {self.session_id}", file=sys.stderr)
                        return True
                except json.JSONDecodeError:
                    pass
            
            print("No session ID received, continuing without session", file=sys.stderr)
            return True

        except Exception as e:
            print(f"Error initializing session: {e}", file=sys.stderr)
            return False

    async def make_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the remote MCP server"""
        try:
            # Prepare headers with session ID if available
            headers = self.headers.copy()
            if self.session_id:
                headers["mcp-session-id"] = self.session_id

            # Ensure we have a client
            if not self.client:
                self.client = httpx.AsyncClient(
                    timeout=httpx.Timeout(30.0),
                    headers=headers,
                    follow_redirects=True
                )

            # Make the request
            response = await self.client.post(
                self.base_url,
                json=request,
                headers=headers,
                timeout=30.0
            )

            # Update session ID if provided in response
            if "mcp-session-id" in response.headers:
                self.session_id = response.headers["mcp-session-id"]

            # Handle different response types
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                
                # Handle Server-Sent Events (SSE)
                if "text/event-stream" in content_type or "event:" in response.text:
                    # Parse SSE format
                    lines = response.text.strip().split('\n')
                    data_lines = []
                    
                    for line in lines:
                        if line.startswith('data: '):
                            data_content = line[6:]  # Remove 'data: ' prefix
                            try:
                                # Try to parse as JSON
                                json_data = json.loads(data_content)
                                data_lines.append(json_data)
                            except json.JSONDecodeError:
                                # If not JSON, add as string
                                data_lines.append(data_content)
                    
                    if data_lines:
                        # Return the first valid JSON response
                        if isinstance(data_lines[0], dict):
                            return data_lines[0]
                        else:
                            return {
                                "jsonrpc": "2.0",
                                "id": request.get("id"),
                                "result": {"data": data_lines[0]}
                            }
                    else:
                        # No valid data found in SSE
                        return {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "result": {"data": response.text}
                        }
                
                # Handle regular JSON responses
                try:
                    return response.json()
                except json.JSONDecodeError:
                    # If response is not JSON, create a proper MCP response
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {"data": response.text}
                    }
            else:
                # Handle error responses
                error_msg = f"HTTP {response.status_code}: {response.text}"
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {
                        "code": -32603,
                        "message": error_msg
                    }
                }

        except httpx.TimeoutException:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32603,
                    "message": "Request timeout"
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"Remote server error: {str(e)}"
                }
            }

    async def handle_mcp_message(self, line: str) -> Optional[str]:
        """Handle a single MCP message from stdin"""
        try:
            request = json.loads(line.strip())
            response = await self.make_request(request)
            return json.dumps(response)
        except json.JSONDecodeError as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {str(e)}"
                }
            }
            return json.dumps(error_response)
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
            return json.dumps(error_response)


async def main():
    """Main function to handle MCPI communication"""
    async with RemoteMCPWrapper() as wrapper:
        # Initialize session
        await wrapper.initialize_session()
        
        print("Remote MCP wrapper initialized", file=sys.stderr)
        print(f"Connected to: {wrapper.base_url}", file=sys.stderr)
        
        # Read from stdin, write to stdout (stdio mode for MCPI)
        while True:
            try:
                # Read line from stdin
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    break
                
                # Process the message
                response = await wrapper.handle_mcp_message(line)
                if response:
                    # Write response to stdout
                    print(response, flush=True)
                    
            except KeyboardInterrupt:
                print("Received interrupt signal", file=sys.stderr)
                break
            except Exception as e:
                print(f"Unexpected error: {e}", file=sys.stderr)
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32603,
                        "message": f"Fatal error: {str(e)}"
                    }
                }
                print(json.dumps(error_response), flush=True)
                break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Wrapper terminated by user", file=sys.stderr)
    except Exception as e:
        print(f"Fatal error in wrapper: {e}", file=sys.stderr)
        sys.exit(1)
