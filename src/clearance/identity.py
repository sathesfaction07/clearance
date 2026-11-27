"""Who is asking: from a header the gateway sets, or from Identity-Aware Proxy's signed token.

Two modes, chosen by `CLEARANCE_IDENTITY`:

* `header` (in-cluster gateway, and kind in CI): the gateway sets `x-authenticated-user`, and
  the network policy is what makes that header trustworthy, because only gateway pods can
  reach the service.
* `iap` (GKE behind Identity-Aware Proxy): IAP signs every request it lets through with a JWT
  in `x-goog-iap-jwt-assertion`. The service verifies it, and the user is the token's
  `email` claim. A plain identity header is ignored, so nothing inside or outside the cluster
  can claim to be someone by setting one.

A token is accepted only if all of these hold: algorithm ES256; a key id from IAP's published
keys; a valid signature; issuer `https://cloud.google.com/iap`; audience equal to this
backend service's (`/projects/NUMBER/global/backendServices/ID`), so a token minted for
another IAP-protected service is refused; issued in the past and not expired, with 30 seconds
of clock skew allowed; and an email present. Anything else is a 401, and the reason is logged,
never the token.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import jwt

IAP_HEADER = "x-goog-iap-jwt-assertion"
IAP_ISSUER = "https://cloud.google.com/iap"
IAP_KEYS_URL = "https://www.gstatic.com/iap/verify/public_key-jwk"
LEEWAY = 30


def _fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:  # a fixed https URL
        return json.load(r)


@dataclass
class IapKeys:
    """IAP's public keys by key id. Fetched once, refreshed hourly, and refetched early (at
    most once a minute) when a token names a key id we have not seen, which is how a key
    rotation shows up. If a refresh fails, the keys already held keep working."""

    url: str = IAP_KEYS_URL
    fetch: Callable[[str], dict] = _fetch
    clock: Callable[[], float] = time.time
    max_age: float = 3600
    min_gap: float = 60
    _keys: dict[str, jwt.PyJWK] = field(default_factory=dict)
    _fetched: float = float("-inf")
    _tried: float = float("-inf")
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def refresh(self, force: bool = False) -> bool:
        now = self.clock()
        with self._lock:
            if not force and now - self._tried < self.min_gap:
                return bool(self._keys)
            self._tried = now
            try:
                jwks = self.fetch(self.url)
                keys = {k["kid"]: jwt.PyJWK(k) for k in jwks["keys"] if k.get("kid")}
            except Exception:
                return bool(self._keys)
            if keys:
                self._keys, self._fetched = keys, now
            return bool(self._keys)

    def get(self, kid: str) -> jwt.PyJWK | None:
        if self.clock() - self._fetched > self.max_age:
            self.refresh()
        if kid not in self._keys:
            self.refresh()
        return self._keys.get(kid)

    @property
    def loaded(self) -> bool:
        return bool(self._keys)


class HeaderIdentity:
    mode = "header"

    def __init__(self, header: str = "x-authenticated-user") -> None:
        self.header = header.lower()

    def user(self, headers: Mapping[str, str]) -> tuple[str, str]:
        user = headers.get(self.header, "").strip()
        return (user, "") if user else ("", "no authenticated user")

    def ready(self) -> bool:
        return True


class IapIdentity:
    mode = "iap"

    def __init__(self, audience: str, keys: IapKeys | None = None) -> None:
        if not audience.startswith("/projects/"):
            raise ValueError(
                "CLEARANCE_IAP_AUDIENCE must be /projects/NUMBER/global/backendServices/ID"
            )
        self.audience = audience
        self.keys = keys or IapKeys()

    def ready(self) -> bool:
        """Ready once IAP's keys are held: before that, every request would be refused."""
        return self.keys.loaded or self.keys.refresh()

    def user(self, headers: Mapping[str, str]) -> tuple[str, str]:
        token = headers.get(IAP_HEADER, "").strip()
        if not token:
            return "", "no IAP token"
        try:
            head = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            return "", "malformed token"
        if head.get("alg") != "ES256":
            return "", "wrong algorithm"
        key = self.keys.get(str(head.get("kid", "")))
        if key is None:
            return "", "unknown key id"
        try:
            claims = jwt.decode(
                token,
                key.key,
                algorithms=["ES256"],
                audience=self.audience,
                issuer=IAP_ISSUER,
                leeway=LEEWAY,
                options={"require": ["exp", "iat", "aud", "iss", "sub", "email"]},
            )
        except jwt.ExpiredSignatureError:
            return "", "expired"
        except jwt.ImmatureSignatureError:
            return "", "issued in the future"
        except jwt.InvalidAudienceError:
            return "", "wrong audience"
        except jwt.InvalidIssuerError:
            return "", "wrong issuer"
        except jwt.InvalidSignatureError:
            return "", "bad signature"
        except jwt.PyJWTError as e:
            return "", f"invalid token ({type(e).__name__})"
        email = str(claims.get("email", "")).strip().lower()
        return (email, "") if email else ("", "no email")


Identity = HeaderIdentity | IapIdentity
