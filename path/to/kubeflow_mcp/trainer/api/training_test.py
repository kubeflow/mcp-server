# complete code
"""
Training API test for the Kubeflow MCP Server.
"""
import unittest
from kubeflow_mcp.trainer.api/training import Training
from kubeflow_mcp.trainer.api/planning import Planning

class TestTrainingAPI(unittest.TestCase):
    def test_fine_tune(self):
        planning = Planning(Discovery(Server(Config())))
        training = Training(planning)
        model_name = "hf://google/gemma-2b"
        dataset_name = "hf://tatsu-lab/alpaca"
        training.fine_tune(model_name, dataset_name)