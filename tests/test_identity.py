"""On GKE the service verifies the token Identity-Aware Proxy signs, and refuses any other.

Tokens here are signed with a test key in exactly IAP's format (ES256, IAP's issuer, the
backend service as audience, the user's email). Real IAP is not involved.
"""

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path

import jwt
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import ec
from starlette.testclient import TestClient

from clearance.identity import IAP_HEADER, IAP_ISSUER, IapIdentity, IapKeys
from clearance.service import State, create_app, identity_from_env

FIXTURE = Path(__file__).parent / "fixture"
AUDIENCE = "/projects/123456/global/backendServices/987654"


def _keypair(kid: str):
    private = ec.generate_private_key(ec.SECP256R1())
    jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(private.public_key()))
    return private, {**jwk, "kid": kid, "alg": "ES256", "use": "sig"}


KEY, JWK = _keypair("key-1")
OTHER_KEY, _ = _keypair("key-1")  # same key id, different key: a forger's
NEW_KEY, NEW_JWK = _keypair("key-2")


class Published:
    """IAP's key endpoint, as a fake that counts fetches and can rotate or fail."""

    def __init__(self, *jwks):
        self.jwks, self.calls, self.down = list(jwks), 0, False

    def __call__(self, url):
        self.calls += 1
        if self.down:
            raise OSError("unreachable")
        return {"keys": self.jwks}


def _token(key=KEY, kid="key-1", alg="ES256", **claims):
    now = int(time.time())
    body = {
        "iss": IAP_ISSUER,
        "aud": AUDIENCE,
        "sub": "accounts.google.com:1234",
        "email": "Tom@Example.com",
        "iat": now,
        "exp": now + 600,
        **claims,
    }
    body = {k: v for k, v in body.items() if v is not None}
    return jwt.encode(body, key, algorithm=alg, headers={"kid": kid})


def _iap(published=None):
    return IapIdentity(AUDIENCE, IapKeys(fetch=published or Published(JWK)))


def _user(identity, token):
    return identity.user({IAP_HEADER: token})


def test_a_genuine_iap_token_gives_the_users_email():
    assert _user(_iap(), _token()) == ("tom@example.com", "")


@pytest.mark.parametrize(
    ("token", "reason"),
    [
        (_token(aud="/projects/123456/global/backendServices/111"), "wrong audience"),
        (_token(iss="https://accounts.google.com"), "wrong issuer"),
        (_token(exp=int(time.time()) - 120), "expired"),
        (_token(iat=int(time.time()) + 600, exp=int(time.time()) + 1200), "issued in the future"),
        (_token(key=OTHER_KEY), "bad signature"),
        (_token(kid="key-9"), "unknown key id"),
        (_token(email=None), "invalid token (MissingRequiredClaimError)"),
        (_token(email=" "), "no email"),
        ("not-a-token", "malformed token"),
    ],
)
def test_anything_but_a_genuine_token_is_refused(token, reason):
    assert _user(_iap(), token) == ("", reason)


def test_algorithm_confusion_and_unsigned_tokens_are_refused():
    # A classic forgery: sign with HMAC, using the published public key as the shared secret.
    # (PyJWT refuses to make one, so it is built by hand.)
    def b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    signing_input = (
        b64(json.dumps({"alg": "HS256", "kid": "key-1"}).encode())
        + "."
        + b64(json.dumps({"email": "tom@example.com", "aud": AUDIENCE}).encode())
    )
    secret = json.dumps(JWK).encode()
    mac = hmac.new(secret, signing_input.encode(), hashlib.sha256).digest()
    assert _user(_iap(), signing_input + "." + b64(mac)) == ("", "wrong algorithm")
    unsigned = jwt.encode({"email": "tom@example.com"}, None, algorithm="none")
    assert _user(_iap(), unsigned) == ("", "wrong algorithm")


def test_a_rotated_key_is_picked_up_and_unknown_ids_cannot_hammer_the_endpoint():
    published = Published(JWK)
    identity = _iap(published)
    assert identity.ready() and published.calls == 1
    published.jwks.append(NEW_JWK)  # IAP rotates; our cache is a minute old at most
    identity.keys._tried -= 120
    assert _user(identity, _token(key=NEW_KEY, kid="key-2"))[0] == "tom@example.com"
    calls = published.calls
    for _ in range(20):
        _user(identity, _token(kid="nobody"))
    assert published.calls == calls  # refetched at most once a minute


def test_keys_already_held_survive_a_failed_refresh():
    published = Published(JWK)
    identity = _iap(published)
    identity.ready()
    published.down = True
    identity.keys._fetched -= 7200  # stale: the next request tries to refresh, and fails
    identity.keys._tried -= 120
    assert _user(identity, _token())[0] == "tom@example.com"


def _access_with_emails(tmp_path):
    access = yaml.safe_load((FIXTURE / "access.yaml").read_text(encoding="utf-8"))
    access["users"] = {f"{u}@example.com": g for u, g in access["users"].items()}
    path = tmp_path / "access.yaml"
    path.write_text(yaml.safe_dump(access), encoding="utf-8")
    return path


def test_the_service_in_iap_mode_ignores_the_plain_header(tmp_path, capsys):
    published = Published(JWK)
    state = State(FIXTURE, _access_with_emails(tmp_path), identity=_iap(published))
    with TestClient(create_app(state, background=False)) as api:
        assert api.get("/readyz").status_code == 200
        q = {"q": "Northwind settlement cap"}
        forged = api.get("/search", params=q, headers={"x-authenticated-user": "tom@example.com"})
        assert forged.status_code == 401
        real = api.get("/search", params=q, headers={IAP_HEADER: _token()})
        assert real.status_code == 200
        assert real.json()["results"][0]["path"] == "legal/contracts.md"
        sam = api.get("/search", params=q, headers={IAP_HEADER: _token(email="sam@example.com")})
        assert all(not r["path"].startswith("legal/") for r in sam.json()["results"])
    log = capsys.readouterr().out
    assert '"reason": "no IAP token"' in log
    assert _token()[:20] not in log  # tokens are never logged


def test_not_ready_until_iap_keys_are_loaded(tmp_path):
    published = Published(JWK)
    published.down = True
    state = State(FIXTURE, _access_with_emails(tmp_path), identity=_iap(published))
    with TestClient(create_app(state, background=False)) as api:
        r = api.get("/readyz")
        assert r.status_code == 503 and "identity keys" in r.json()["error"]
        published.down = False
        state.identity.keys._tried -= 120
        assert api.get("/readyz").status_code == 200


def test_identity_configuration_fails_at_start_up(monkeypatch):
    monkeypatch.setenv("CLEARANCE_IDENTITY", "iap")
    monkeypatch.delenv("CLEARANCE_IAP_AUDIENCE", raising=False)
    with pytest.raises(ValueError, match="backendServices"):
        identity_from_env()
    monkeypatch.setenv("CLEARANCE_IAP_AUDIENCE", AUDIENCE)
    assert identity_from_env().mode == "iap"
    monkeypatch.setenv("CLEARANCE_IDENTITY", "trust-me")
    with pytest.raises(ValueError, match="header or iap"):
        identity_from_env()
    monkeypatch.delenv("CLEARANCE_IDENTITY")
    assert identity_from_env().mode == "header"
