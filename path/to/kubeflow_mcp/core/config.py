# complete code
"""
Core configuration for the Kubeflow MCP Server.
"""
from typing import Optional

class Config:
    def __init__(self, hf_home: Optional[str] = None):
        self.hf_home = hf_home or "/workspace/.hf"

    def get_hf_home(self) -> str:
        return self.hf_home