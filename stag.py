import asyncio
import json
import sys
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack
from pydantic import BaseModel

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import xgrammar as xgr

# Define the schemas for our functions
class GetWeather(BaseModel):
    """Get weather information for a location"""
    location: str
    unit: Optional[str] = "celsius"
    days: Optional[int] = 1

class SearchDatabase(BaseModel):
    """Search a database for information"""
    query: str
    max_results: Optional[int] = 10
    filters: Optional[Dict[str, Any]] = None

# Dummy functions for testing
def get_weather(location: str, unit: str = "celsius", days: int = 1) -> Dict[str, Any]:
    """Dummy function to get weather information"""
    print(f"Called get_weather with: location={location}, unit={unit}, days={days}")
    # Normally this would make an API call to a weather service
    return {
        "location": location,
        "forecast": [
            {"day": 1, "temp": 22, "condition": "Sunny", "unit": unit},
            {"day": 2, "temp": 19, "condition": "Cloudy", "unit": unit}
        ],
        "unit": unit
    }

def search_database(query: str, max_results: int = 10, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Dummy function to search a database"""
    print(f"Called search_database with: query={query}, max_results={max_results}, filters={filters}")
    # Normally this would query a database
    return {
        "query": query,
        "results": [
            {"id": 1, "title": "Sample result 1", "relevance": 0.95},
            {"id": 2, "title": "Sample result 2", "relevance": 0.87}
        ],
        "total_results": 2,
        "max_results": max_results
    }

# Function to parse and execute function calls
def execute_function(func_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the specified function with the given parameters"""
    if func_name == "get_weather":
        return get_weather(**params)
    elif func_name == "search_database":
        return search_database(**params)
    else:
        raise ValueError(f"Unknown function: {func_name}")

def main():
    # Model and tokenizer setup
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=True
    )
    
    # Print available devices for debugging
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    # Load model with bfloat16 for efficiency if supported
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Set up StructuralTag and tokenizer info
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
    
    # Define the structural tags for functions
    tags = [
        xgr.StructuralTagItem(
            begin="<function=get_weather>", 
            schema=GetWeather, 
            end="</function>"
        ),
        xgr.StructuralTagItem(
            begin="<function=search_database>", 
            schema=SearchDatabase, 
            end="</function>"
        ),
    ]
    
    # Define triggers
    triggers = ["<function="]
    
    # Compile grammar
    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammar = compiler.compile_structural_tag(tags, triggers)
    
    # Create matcher
    matcher = xgr.GrammarMatcher(compiled_grammar)
    
    # Determine the device to use (same as the model's device)
    device = model.device
    
    # Allocate token bitmask on CPU as per documentation
    token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)
    
    # Test prompts
    test_prompts = [
        "What's the weather like in New York today?",
        "Find information about quantum computing."
    ]
    
    # Temperature for generation
    temperature = 0.7
    
    for prompt in test_prompts:
        print("\n" + "="*80)
        print(f"PROMPT: {prompt}")
        print("="*80)
        
        # Prepare the system prompt
        system_prompt = (
            "You are a helpful assistant that can call functions to get information. "
            "When you need to call a function, use the <function=func_name> tag followed by the JSON parameters, "
            "and end with </function>."
        )
        
        # Format the input for Llama models
        formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
        
        # Tokenize the input
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
        
        # Reset the matcher for new generation
        matcher.reset()
        
        # Store the generated text
        generated_text = ""
        
        # Set max sequence length
        max_length = 2048
        
        # Generate text token by token
        for _ in range(max_length - len(input_ids[0])):
            # Get model outputs for the next token
            with torch.no_grad():
                outputs = model(input_ids)
                next_token_logits = outputs.logits[0, -1, :].unsqueeze(0)
            
            # Apply grammar constraints if needed
            need_apply = matcher.fill_next_token_bitmask(token_bitmask)
            if need_apply:
                # Create a CUDA version of the bitmask following benchmark pattern
                cuda_bitmask = token_bitmask.to(device=device)
                #print(f"CUDA bitmask dtype: {cuda_bitmask.dtype}")
                xgr.apply_token_bitmask_inplace(next_token_logits, cuda_bitmask)
            
            # Apply temperature and sample
            next_token_logits = next_token_logits / temperature
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Update the matcher with the next token
            matcher.accept_token(next_token)
            
            # Update input_ids with the new token
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=model.device)], dim=-1)
            
            # Decode the token
            next_token_str = tokenizer.decode([next_token])
            generated_text += next_token_str
            """
            # Check for <function> tags to execute functions
            if "</function>" in generated_text:
                # Parse the generated function call
                function_call_match = generated_text.find("<function=")
                function_end_match = generated_text.find("</function>")
                
                if function_call_match != -1 and function_end_match != -1:
                    function_call_text = generated_text[function_call_match:function_end_match + len("</function>")]
                    
                    # Extract function name
                    func_name_start = function_call_text.find("<function=") + len("<function=")
                    func_name_end = function_call_text.find(">")
                    func_name = function_call_text[func_name_start:func_name_end]
                    
                    # Extract parameters (JSON)
                    params_start = function_call_text.find(">") + 1
                    params_end = function_call_text.find("</function>")
                    params_text = function_call_text[params_start:params_end].strip()
                    
                    try:
                        params = json.loads(params_text)
                        
                        # Execute the function
                        print(f"\nExecuting function: {func_name}")
                        print(f"Parameters: {params}")
                        
                        result = execute_function(func_name, params)
                        
                        # Format the result to provide back to the model
                        result_text = f"\nFunction result: {json.dumps(result, indent=2)}\n"
                        
                        # Update the generated text with the result
                        generated_text += result_text
                        
                        # Tokenize the result to add to input_ids
                        result_ids = tokenizer(result_text, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
                        input_ids = torch.cat([input_ids, result_ids], dim=-1)
                        
                        # Reset the matcher
                        matcher.reset()
                        
                    except json.JSONDecodeError:
                        print(f"Error parsing JSON parameters: {params_text}")
            """
            # Check for end of generation
            if next_token == tokenizer.eos_token_id:
                break
        
        # Display the full generated text
        print("\nGENERATED OUTPUT:")
        print(generated_text)

if __name__ == "__main__":
    main()