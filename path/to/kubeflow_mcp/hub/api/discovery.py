# complete code
"""
Discovery API for the Kubeflow MCP Server.
"""
from typing import Optional
from kubeflow_mcp.core/server import Server

class Discovery:
    def __init__(self, server: Server):
        self.server = server

    def get_hf_home(self) -> str:
        return self.server.get_hf_home()