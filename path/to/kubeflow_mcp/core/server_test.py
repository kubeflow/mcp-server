# complete code
"""
Server API test for the Kubeflow MCP Server.
"""
import unittest
from kubeflow_mcp.core.server import Server
from kubeflow_mcp.core.config import Config

class TestServerAPI(unittest.TestCase):
    def test_get_hf_home(self):
        config = Config()
        server = Server(config)
        hf_home = server.get_hf_home()
        self.assertEqual(hf_home, "/workspace/.hf")