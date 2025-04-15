from openai import OpenAI
import json
from sglang.utils import wait_for_server, print_highlight, terminate_process
from sglang.test.test_utils import is_in_ci

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

"""Refer to https://docs.sglang.ai/backend/function_calling.html for supported models"""

server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server --model-path meta-llama/Llama-3.2-3B-Instruct --tool-call-parser llama3 --host 0.0.0.0"  # qwen25
)
wait_for_server(f"http://localhost:{port}")