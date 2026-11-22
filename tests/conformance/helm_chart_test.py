"""Conformance tests for the official Kubeflow MCP Server Helm chart and Kustomize overlays."""

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHART_DIR = REPO_ROOT / "charts" / "kubeflow-mcp"
TEMPLATES_DIR = CHART_DIR / "templates"
OVERLAYS_DIR = REPO_ROOT / "examples" / "kubernetes" / "overlays"


class TestHelmChartMetadata:
    """Validate Chart.yaml specification and conventions."""

    def test_chart_yaml_exists_and_valid(self):
        chart_file = CHART_DIR / "Chart.yaml"
        assert chart_file.is_file(), "Chart.yaml must exist in charts/kubeflow-mcp"

        with open(chart_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data["apiVersion"] == "v2"
        assert data["name"] == "kubeflow-mcp"
        assert "kubeflow" in data["keywords"]
        assert "mcp" in data["keywords"]
        assert data["type"] == "application"
        assert data.get("version") is not None
        assert data.get("appVersion") is not None
        assert data.get("home") == "https://github.com/kubeflow/mcp-server"


class TestHelmChartValues:
    """Validate values.yaml defaults and enterprise requirements."""

    def test_values_yaml_exists_and_valid(self):
        values_file = CHART_DIR / "values.yaml"
        assert values_file.is_file(), "values.yaml must exist in charts/kubeflow-mcp"

        with open(values_file, encoding="utf-8") as f:
            values = yaml.safe_load(f)

        # Horizontal scaling default
        assert values["replicaCount"] >= 2
        assert values["autoscaling"]["enabled"] is True
        assert values["autoscaling"]["minReplicas"] >= 2

        # Transport & server defaults
        server = values["server"]
        assert server["transport"] == "http"
        assert server["port"] == 8000
        assert server["host"] == "0.0.0.0"
        assert "trainer" in server["clients"]
        assert "spark" in server["clients"]
        assert "pipelines" in server["clients"]

        # Health & readiness probes match core server endpoints
        probes = values["probes"]
        assert probes["liveness"]["httpGet"]["path"] == "/health"
        assert probes["readiness"]["httpGet"]["path"] == "/ready"

        # Security context best practices
        sec = values["securityContext"]
        assert sec["runAsNonRoot"] is True
        assert sec["readOnlyRootFilesystem"] is True
        assert sec["allowPrivilegeEscalation"] is False
        assert "ALL" in sec["capabilities"]["drop"]

        # RBAC enabled
        assert values["rbac"]["create"] is True
        assert values["rbac"]["clusterRole"] is True


class TestHelmTemplatesIntegrity:
    """Validate template files, helpers, and syntax."""

    EXPECTED_TEMPLATES = [
        "_helpers.tpl",
        "deployment.yaml",
        "service.yaml",
        "serviceaccount.yaml",
        "clusterrole.yaml",
        "clusterrolebinding.yaml",
        "role.yaml",
        "rolebinding.yaml",
        "configmap.yaml",
        "secret.yaml",
        "ingress.yaml",
        "hpa.yaml",
        "networkpolicy.yaml",
        "servicemonitor.yaml",
        "NOTES.txt",
    ]

    def test_expected_templates_present(self):
        for tmpl in self.EXPECTED_TEMPLATES:
            path = TEMPLATES_DIR / tmpl
            assert path.is_file(), f"Missing expected template: {tmpl}"

    def test_template_braces_balanced(self):
        for tmpl in TEMPLATES_DIR.glob("*.*"):
            content = tmpl.read_text(encoding="utf-8")
            open_count = len(re.findall(r"\{\{", content))
            close_count = len(re.findall(r"\}\}", content))
            assert open_count == close_count, (
                f"Unbalanced braces in {tmpl.name}: {open_count} open vs {close_count} close"
            )

    def test_helpers_define_standard_macros(self):
        helpers_file = TEMPLATES_DIR / "_helpers.tpl"
        content = helpers_file.read_text(encoding="utf-8")
        macros = [
            "kubeflow-mcp.name",
            "kubeflow-mcp.fullname",
            "kubeflow-mcp.chart",
            "kubeflow-mcp.labels",
            "kubeflow-mcp.selectorLabels",
            "kubeflow-mcp.serviceAccountName",
            "kubeflow-mcp.authSecretName",
        ]
        for macro in macros:
            assert f'define "{macro}"' in content, f"Missing macro definition: {macro}"


class TestKustomizeOverlays:
    """Validate Kustomize overlays in examples/kubernetes/overlays/."""

    def test_stateless_http_overlay_valid(self):
        overlay = OVERLAYS_DIR / "stateless-http"
        assert overlay.is_dir()

        kust_file = overlay / "kustomization.yaml"
        assert kust_file.is_file()
        with open(kust_file, encoding="utf-8") as f:
            kust = yaml.safe_load(f)
        assert "../.." in kust["resources"]
        assert "hpa.yaml" in kust["resources"]
        assert "ingress.yaml" in kust["resources"]

        hpa_file = overlay / "hpa.yaml"
        with open(hpa_file, encoding="utf-8") as f:
            hpa = yaml.safe_load(f)
        assert hpa["kind"] == "HorizontalPodAutoscaler"
        assert hpa["spec"]["minReplicas"] >= 2

        ingress_file = overlay / "ingress.yaml"
        with open(ingress_file, encoding="utf-8") as f:
            ing = yaml.safe_load(f)
        assert ing["kind"] == "Ingress"

        patch_file = overlay / "patch-replicas.yaml"
        with open(patch_file, encoding="utf-8") as f:
            patch = yaml.safe_load(f)
        assert patch["kind"] == "Deployment"
        assert patch["spec"]["replicas"] >= 2

    def test_oidc_overlay_valid(self):
        overlay = OVERLAYS_DIR / "oidc"
        assert overlay.is_dir()

        kust_file = overlay / "kustomization.yaml"
        assert kust_file.is_file()
        with open(kust_file, encoding="utf-8") as f:
            kust = yaml.safe_load(f)
        assert "../.." in kust["resources"]

        patch_file = overlay / "patch-oidc.yaml"
        with open(patch_file, encoding="utf-8") as f:
            patch = yaml.safe_load(f)
        assert patch["kind"] == "Deployment"
        env = patch["spec"]["template"]["spec"]["containers"][0]["env"]
        env_names = [e["name"] for e in env]
        assert "KUBEFLOW_MCP_OIDC_ISSUER" in env_names
        assert "KUBEFLOW_MCP_OIDC_AUDIENCE" in env_names
