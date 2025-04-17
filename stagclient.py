import asyncio
import json
import sys
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv
import xgrammar as xgr

load_dotenv()

class XgrStagClient:
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

    def convert_mcp_tools_to_tags(self, mcp_tools):
        """Convert MCP tools to structural tag format"""
        tags = []
        
        for tool in mcp_tools:
            if 'required' in tool['description']:
                structure = {
                    "begin": f"<function={tool['name']}>",
                    "schema": tool["input_schema"],
                    "end": "</function>",
                    "required": "true"
                }
            else:
                structure = {
                    "begin": f"<function={tool['name']}>",
                    "schema": tool["input_schema"],
                    "end": "</function>",
                    "required": "false"
                }
            structures.append(structure)
            
        return tags

    def create_system_prompt(self, mcp_tools):
        """Create a system prompt that instructs the model on how to use the tools"""
        tool_instructions = []
        
        for tool in mcp_tools:
            tool_instructions.append(f"Use the function '{tool['name']}' to: {tool['description']}")
            tool_instructions.append(json.dumps(tool["input_schema"], indent=2))
            
        system_prompt = f"""
# Tool Instructions
- Always execute python code in messages that you share.
- When looking for real time information use relevant functions if available else fallback to brave_search
You have access to the following functions:
{chr(10).join(tool_instructions)}

If a you choose to call a function ONLY reply in the following format:
<{{start_tag}}={{function_name}}>{{parameters}}{{end_tag}}
where
start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`
Here is an example,
<function=example_function_name>{{"example_name": "example_value"}}</function>
Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- Always add your sources when using search results to answer the user query
You are a helpful assistant.
"""
        return system_prompt

    async def process_query(self, query: str) -> str:
        """Process a query using SGL and available tools"""
        # Get available tools
        response = await self.session.list_tools()
        available_mcp_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Convert tools to structural tag format
        tags = self.convert_mcp_tools_to_tags(available_mcp_tools)
        
        # Create system prompt
        system_prompt = self.create_system_prompt(available_mcp_tools)
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        try:
            tool_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                top_p=0.95,
                max_tokens=1024,
                stream=False,
                response_format={
                    "type": "structural_tag",
                    "tags": tags,
                    "triggers": ["<function="],
                    "tool_choice": "auto"
                }
            )
            
            tool_response_content = tool_response.choices[0].message.content
            
            # Parse the structural tag response
            try:
                content, tool_calls = xgr.parse_structural_tag(tool_response_content)
                
                # If there are tool calls
                if tool_calls:
                    tool_results_text = []
                    
                    # Add the assistant response to messages
                    if content:
                        messages.append({
                            "role": "assistant",
                            "content": content
                        })
                    
                    # Process each tool call
                    for start_tag, params, end_tag in tool_calls:
                        # Extract function name from start_tag
                        function_name = start_tag.split("=")[1].rstrip(">")
                        
                        tool_results_text.append(f"[Calling tool {function_name} with args {params}]")
                        result = await self.session.call_tool(function_name, params)
                        
                        # Add tool result to messages
                        messages.append({
                            "role": "system",
                            "content": f"Tool {function_name} returned: {result.content}"
                        })
                    
                    # Generate follow-up response
                    follow_up_response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.1,
                        top_p=0.95,
                        max_tokens=1024,
                        stream=False
                    )
                    
                    final_text = follow_up_response.choices[0].message.content
                else:
                    # If no tool calls, just use the initial response
                    final_text = content if content else "No response generated."
                
                return final_text
                
            except Exception as e:
                return f"Error parsing tool response: {str(e)}\nRaw response: {tool_response_content}"
            
        except Exception as e:
            return f"Error processing query: {str(e)}"

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client by XGrammar Started!")
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
        print("Usage: python xgr_stag_client.py <path_to_server_script> <sglang_port>")
        sys.exit(1)
    
    sglang_port = int(sys.argv[2])
    
    client = XgrStagClient(sglang_port)
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())