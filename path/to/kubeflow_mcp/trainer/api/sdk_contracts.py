# complete code
"""
SDK contracts for the Kubeflow MCP Server.
"""
from typing import Optional

class HuggingFaceModelInitializer:
    def __init__(self, hf_home: Optional[str] = None):
        self.hf_home = hf_home or "/workspace/.hf"

class HuggingFaceDatasetInitializer:
    def __init__(self, hf_home: Optional[str] = None):
        self.hf_home = hf_home or "/workspace/.hf"