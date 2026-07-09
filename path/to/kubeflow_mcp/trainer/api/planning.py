# complete code
"""
Planning API for the Kubeflow MCP Server.
"""
from typing import Optional
from kubeflow_mcp.trainer.api/sdk_contracts import HuggingFaceModelInitializer, HuggingFaceDatasetInitializer
from kubeflow_mcp.hub.api.discovery import Discovery

class Planning:
    def __init__(self, discovery: Discovery):
        self.discovery = discovery

    def get_hf_home(self) -> str:
        return self.discovery.get_hf_home()