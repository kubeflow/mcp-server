# complete code
"""
Discovery API test for the Kubeflow MCP Server.
"""
import unittest
from kubeflow_mcp.hub.api/discovery import Discovery
from kubeflow_mcp.core/server import Server

class TestDiscoveryAPI(unittest.TestCase):
    def test_get_hf_home(self):
        server = Server(Config())
        discovery = Discovery(server)
        hf_home = discovery.get_hf_home()
        self.assertEqual(hf_home, "/workspace/.hf")