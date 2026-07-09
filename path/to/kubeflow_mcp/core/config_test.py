# complete code
"""
Core configuration test for the Kubeflow MCP Server.
"""
import unittest
from kubeflow_mcp.core.config import Config

class TestCoreConfig(unittest.TestCase):
    def test_get_hf_home(self):
        config = Config()
        hf_home = config.get_hf_home()
        self.assertEqual(hf_home, "/workspace/.hf")