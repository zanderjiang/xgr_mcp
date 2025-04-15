import asyncio
import json
import sys
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack
from pydantic import BaseModel

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import xgrammar as xgr

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class XGrammarMCPClient:
    def __init__(self, model_name="meta-llama/Llama-3.2-3B-Instruct"):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # XGrammar LLM Engine
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.config = AutoConfig.from_pretrained(self.model_name)
        
        self.tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer, vocab_size=self.config.vocab_size
        )
        self.grammar_compiler = xgr.GrammarCompiler(self.tokenizer_info)
        
        # Initialize with JSON grammar, will be updated with tools
        self.compiled_grammar = self.grammar_compiler.compile_builtin_json_grammar()
        self.xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(self.compiled_grammar)

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
        
        # List available tools
        response = await self.session.list_tools()
        self.tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in self.tools])

    def _setup_grammar_for_tools(self):
        """Set up XGrammar with tool schemas"""
        tags = []
        
        for tool in self.tools:
            # Create a structural tag item for each tool
            tag = xgr.StructuralTagItem(
                begin=f"<function={tool.name}>",
                schema=tool.inputSchema,  # JSON schema from the tool
                end="</function>"
            )
            tags.append(tag)
        
        # Set triggers for tool functions
        triggers = ["<function="]
        
        # Compile the structural tag grammar
        self.compiled_grammar = self.grammar_compiler.compile_structural_tag(tags, triggers)
        self.xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(self.compiled_grammar)
        
        print(f"Grammar updated with {len(tags)} tools")
    
    def _create_system_prompt_with_tools(self) -> str:
        """Create a system prompt that includes tool definitions"""
        tool_descriptions = []
    
        for tool in self.tools:
            # Format the tool schema as a string
            schema_str = json.dumps(tool.inputSchema, indent=2)
        
            # Add tool description to the list
            tool_description = f"""
            Tool: {tool.name}
            Description: {tool.description}
            Schema: {schema_str}
            """
            tool_descriptions.append(tool_description)
    
        # Create the complete system prompt
        system_prompt = f"""You are a helpful assistant with access to the following tools:

{''.join(tool_descriptions)}

To use a tool, respond with:
<function=TOOL_NAME>
{{
  "param1": "value1",
  "param2": "value2"
}}
</function>

First think about whether you need to use a tool. If not, just respond normally.
If you need to use a tool, format your response correctly according to the tool's schema.
        """
        return system_prompt

    async def process_query(self, query: str) -> str:
        """Process a query using XGrammar LLM and available tools"""
        # Build the prompt
        self._setup_grammar_for_tools()

        system_prompt = self._create_system_prompt_with_tools()
        
        # Prepare conversation for Llama
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate response with XGrammar enforcing JSON format
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            logits_processor=[self.xgr_logits_processor],
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Extract only the generated output (not the input prompt)
        generated_ids = output_ids[0][len(inputs[0]):]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Process the response for tool calls
        return await self._process_tool_calls(response_text)
    
    async def _process_tool_calls(self, response_text: str) -> str:
        """Process response text for tool calls and execute them"""
        final_text = []
        current_position = 0
        
        # Find and execute tool calls in the response
        while current_position < len(response_text):
            # Look for start of function call
            function_start = response_text.find("<function=", current_position)
            
            if function_start == -1:
                # No more function calls, add remaining text
                final_text.append(response_text[current_position:])
                break
                
            # Add text before the function call
            if function_start > current_position:
                final_text.append(response_text[current_position:function_start])
            
            # Extract function name
            name_start = function_start + len("<function=")
            name_end = response_text.find(">", name_start)
            function_name = response_text[name_start:name_end]
            
            # Extract function arguments (JSON between > and </function>)
            args_start = name_end + 1
            tag_end = response_text.find("</function>", args_start)
            
            if tag_end == -1:
                # Malformed function call, add text and continue
                final_text.append(response_text[current_position:])
                break
                
            args_json = response_text[args_start:tag_end].strip()
            
            try:
                # Parse arguments and call the tool
                args = json.loads(args_json)
                final_text.append(f"[Calling tool {function_name}]")
                
                # Execute the tool call
                result = await self.session.call_tool(function_name, args)
                
                # Add tool result to final text
                final_text.append(f"[Tool result: {result.content}]")
                
                # Update position to after the function call
                current_position = tag_end + len("</function>")
                
            except json.JSONDecodeError:
                # Handle malformed JSON
                final_text.append(f"[Error: Could not parse arguments for {function_name}]")
                current_position = tag_end + len("</function>")
                
            except Exception as e:
                # Handle other errors
                final_text.append(f"[Error calling {function_name}: {str(e)}]")
                current_position = tag_end + len("</function>")
        
        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nXGrammar MCP Client Started!")
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
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
    
    client = XGrammarMCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())