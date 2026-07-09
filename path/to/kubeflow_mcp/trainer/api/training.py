# complete code
"""
Training API for the Kubeflow MCP Server.
"""
from typing import Optional
from kubeflow_mcp.trainer.api/sdk_contracts import HuggingFaceModelInitializer, HuggingFaceDatasetInitializer
from kubeflow_mcp.trainer.api/planning import Planning

class Training:
    def __init__(self, planning: Planning):
        self.planning = planning

    def fine_tune(self, model_name: str, dataset_name: str):
        # Set HF_HOME environment variable
        hf_home = self.planning.get_hf_home()
        env = {"HF_HOME": hf_home}
        # Initialize model and dataset
        model_initializer = HuggingFaceModelInitializer(hf_home)
        dataset_initializer = HuggingFaceDatasetInitializer(hf_home)
        model = model_initializer(model_name)
        dataset = dataset_initializer(dataset_name)
        # Perform fine-tuning
        # ...