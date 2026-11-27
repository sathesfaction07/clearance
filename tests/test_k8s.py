"""The Kubernetes manifests agree with the code and with each other.

CI also renders them and applies them to a real cluster (the kind job); these checks catch
drift before that, in milliseconds.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
K8S = ROOT / "k8s"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_cluster_access_file_is_the_same_as_the_one_tested():
    assert _load(K8S / "base" / "access.yaml") == _load(ROOT / "config" / "access.yaml")


def test_only_the_gateway_label_the_probe_uses_may_reach_the_service():
    policy = _load(K8S / "base" / "networkpolicy.yaml")
    probe = _load(K8S / "tests" / "probe-job.yaml")
    allowed = policy["spec"]["ingress"][0]["from"][0]["podSelector"]["matchLabels"]
    assert allowed.items() <= probe["spec"]["template"]["metadata"]["labels"].items()
    assert policy["spec"]["egress"] == [] and "Egress" in policy["spec"]["policyTypes"]


def test_pods_are_hardened_and_cannot_drop_requests_on_rollout():
    deploy = _load(K8S / "base" / "deployment.yaml")
    spec = deploy["spec"]
    pod = spec["template"]["spec"]
    container = pod["containers"][0]
    assert spec["strategy"]["rollingUpdate"]["maxUnavailable"] == 0
    assert container["readinessProbe"]["httpGet"]["path"] == "/readyz"
    assert container["livenessProbe"]["httpGet"]["path"] == "/healthz"
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert pod["automountServiceAccountToken"] is False
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]


def test_the_image_does_not_log_query_strings():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "--no-access-log" in dockerfile
