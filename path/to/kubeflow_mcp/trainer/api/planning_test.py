# complete code
"""
Planning API test for the Kubeflow MCP Server.
"""
import unittest
from kubeflow_mcp.trainer.api/planning import Planning
from kubeflow_mcp.hub.api/discovery import Discovery

class TestPlanningAPI(unittest.TestCase):
    def test_get_hf_home(self):
        discovery = Discovery(Server(Config()))
        planning = Planning(discovery)
        hf_home = planning.get_hf_home()
        self.assertEqual(hf_home, "/workspace/.hf")