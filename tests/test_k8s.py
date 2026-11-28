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
    assert "Egress" in policy["spec"]["policyTypes"]
    egress = {
        (str(rule.get("to", [{}])[0].get("podSelector", {}).get("matchLabels")), p["port"])
        for rule in policy["spec"]["egress"]
        for p in rule["ports"]
    }
    assert egress == {
        ("{'k8s-app': 'kube-dns'}", 53),
        ("{'app': 'clearance-db'}", 5432),
    }  # names, and its own database: nothing else


def test_only_the_service_reaches_the_database_and_the_database_reaches_nothing():
    docs = list(yaml.safe_load_all((K8S / "base" / "database.yaml").read_text("utf-8")))
    (policy,) = [d for d in docs if d["kind"] == "NetworkPolicy"]
    assert policy["spec"]["egress"] == []
    assert policy["spec"]["ingress"][0]["from"] == [
        {"podSelector": {"matchLabels": {"app": "clearance"}}}
    ]
    (db,) = [d for d in docs if d["kind"] == "StatefulSet"]
    pod = db["spec"]["template"]["spec"]
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert pod["containers"][0]["securityContext"]["readOnlyRootFilesystem"] is True
    assert pod["containers"][0]["image"].startswith("pgvector/pgvector:0.8")


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


def _gke_patch(kind: str) -> list[dict]:
    kustomization = _load(K8S / "overlays" / "gke" / "kustomization.yaml")
    (patch,) = [p for p in kustomization["patches"] if p["target"]["kind"] == kind]
    return yaml.safe_load(patch["patch"])


def test_gke_verifies_iap_tokens_and_has_the_same_access_rules():
    env = {op["value"]["name"]: op["value"]["value"] for op in _gke_patch("Deployment")[1:]}
    assert env["CLEARANCE_IDENTITY"] == "iap"
    assert env["CLEARANCE_QUERY_EMBEDDER"] == "ollama"
    assert env["CLEARANCE_IAP_AUDIENCE"].startswith("/projects/")
    tested = _load(ROOT / "config" / "access.yaml")
    gke = _load(K8S / "overlays" / "gke" / "access.yaml")
    assert gke["sections"] == tested["sections"]
    assert {f"{u}@example.com": g for u, g in tested["users"].items()} == gke["users"]


def test_gke_admits_only_google_front_ends_and_reaches_only_dns_and_https():
    ops = {op["path"]: op["value"] for op in _gke_patch("NetworkPolicy")}
    (ingress,) = ops["/spec/ingress"]
    assert {b["ipBlock"]["cidr"] for b in ingress["from"]} == {"35.191.0.0/16", "130.211.0.0/22"}
    ports = sorted(p["port"] for rule in ops["/spec/egress"] for p in rule["ports"])
    assert ports == [53, 53, 443, 5432, 11434]
    iap = list(yaml.safe_load_all((K8S / "overlays" / "gke" / "iap.yaml").read_text("utf-8")))
    (backend,) = [d for d in iap if d["kind"] == "BackendConfig"]
    assert backend["spec"]["iap"]["enabled"] is True
    assert backend["spec"]["healthCheck"]["requestPath"] == "/readyz"
