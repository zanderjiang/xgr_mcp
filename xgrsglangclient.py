import asyncio
import json
import sys
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class XgrMCPClient:
    def __init__(self, sglang_port):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        self.client = OpenAI(api_key="None", base_url=f"http://0.0.0.0:{sglang_port}/v1")
        self.model_name = None  # Will be set during initialization

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # Initialize SGL with XGrammar backend
        try:
            models = self.client.models.list().data
            if models:
                self.model_name = models[0].id
                print(f"Using SGL model: {self.model_name}")
            else:
                raise ValueError("No models available from SGL server")
        except Exception as e:
            print(f"Error initializing SGL client: {str(e)}")
            raise
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    def convert_mcp_tools_to_openai_format(self, mcp_tools):
        """Convert MCP tool format to OpenAI tool format for SGL compatibility"""
        openai_tools = []
        
        for tool in mcp_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            }
            openai_tools.append(openai_tool)
            
        return openai_tools

    async def process_query(self, query: str) -> str:
        """Process a query using SGL and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_mcp_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        openai_tools = self.convert_mcp_tools_to_openai_format(available_mcp_tools)
        
        try:
            sgl_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                top_p=0.95,
                max_tokens=1024,
                stream=False,
                tools=openai_tools
            )
            
            assistant_response = ""
            tool_results_text = []
            
            assistant_message = sgl_response.choices[0].message
            if assistant_message.content:
                assistant_response = assistant_message.content
            
            if assistant_message.tool_calls:
                if assistant_message.content:
                    messages.append({
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            } for tool_call in assistant_message.tool_calls
                        ]
                    })
                
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args_str = tool_call.function.arguments
                    
                    try:
                        tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        tool_args = {}
                    
                    tool_results_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                    result = await self.session.call_tool(tool_name, tool_args)
                    
                    messages.append({
                        "role": "tool", 
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": result.content
                    })
                
                follow_up_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    top_p=0.95,
                    max_tokens=1024,
                    stream=False
                )
                
                if follow_up_response.choices[0].message.content:
                    final_text = follow_up_response.choices[0].message.content
                else:
                    final_text = assistant_response if assistant_response else "No response generated."
            else:
                # If no tool calls, just use the initial response
                final_text = assistant_response
            
            return final_text
            
        except Exception as e:
            return f"Error processing query: {str(e)}"

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 3:
        print("Usage: python client.py <path_to_server_script> [<sglang_port>]")
        sys.exit(1)
    
    if len(sys.argv) > 2:
        sglang_port = int(sys.argv[2])
    
    client = XgrMCPClient(sglang_port)
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())