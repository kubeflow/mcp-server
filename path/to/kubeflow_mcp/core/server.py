# complete code
"""
Server API for the Kubeflow MCP Server.
"""
from typing import Optional
from kubeflow_mcp.core.config import Config

class Server:
    def __init__(self, config: Config):
        self.config = config

    def get_hf_home(self) -> str:
        return self.config.get_hf_home()