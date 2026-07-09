# complete code
"""
SDK contracts test for the Kubeflow MCP Server.
"""
import unittest
from kubeflow_mcp.trainer.api/sdk_contracts import HuggingFaceModelInitializer, HuggingFaceDatasetInitializer

class TestSDKContracts(unittest.TestCase):
    def test_hf_home(self):
        hf_home = "/workspace/.hf"
        model_initializer = HuggingFaceModelInitializer(hf_home)
        dataset_initializer = HuggingFaceDatasetInitializer(hf_home)
        self.assertEqual(model_initializer.hf_home, hf_home)
        self.assertEqual(dataset_initializer.hf_home, hf_home)