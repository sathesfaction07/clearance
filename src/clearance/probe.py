"""Check a running service from the outside: every question, as every user, over HTTP.

Run inside the cluster as a gateway pod (k8s/probe-job.yaml), so it reaches the service the
way real traffic does. For each search it checks that every returned page belongs to a group
the user is in, according to the access file. It also checks that a request with no identity
is refused and an unknown user gets nothing. With `--seconds`, it keeps asking for that long,
which is how CI shows that a rolling restart drops no requests.

Exits 1 on any leak or any failed request. Standard library only, so it runs from the
service image.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from clearance.corpus import Access
from clearance.evaluate import load_questions


def _get(url: str, headers: dict[str, str], timeout: float = 10) -> tuple[int, dict]:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as e:
        return e.code, {}
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return 0, {"error": str(e)}


def probe(
    base: str, access: Access, questions: list, header: str, seconds: float = 0
) -> dict[str, int]:
    out = {"searches": 0, "failed": 0, "leaked_results": 0, "results": 0, "rounds": 0}
    deadline = time.monotonic() + seconds
    while True:
        for user, groups in access.user_groups.items():
            for q in questions:
                url = f"{base}/search?" + urllib.parse.urlencode({"q": q.q, "k": 5})
                status, body = _get(url, {header: user})
                out["searches"] += 1
                if status != 200:
                    out["failed"] += 1
                    continue
                for r in body["results"]:
                    out["results"] += 1
                    section = r["path"].split("/", 1)[0]
                    out["leaked_results"] += access.section_groups.get(section) not in groups
        out["rounds"] += 1
        if time.monotonic() >= deadline:
            break
    query = "?" + urllib.parse.urlencode({"q": questions[0].q})
    out["anonymous_status"] = _get(f"{base}/search{query}", {})[0]
    status, body = _get(f"{base}/search{query}", {header: "nobody-in-the-access-file"})
    out["unknown_user_results"] = len(body.get("results", [])) if status == 200 else -1
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", required=True, help="the service, e.g. http://clearance:8080")
    p.add_argument("--access", required=True, type=Path)
    p.add_argument("--questions", required=True, type=Path)
    p.add_argument("--header", default="x-authenticated-user")
    p.add_argument("--seconds", type=float, default=0, help="keep asking for this long")
    args = p.parse_args(argv)
    result = probe(
        args.url.rstrip("/"),
        Access.load(args.access),
        load_questions(args.questions),
        args.header,
        args.seconds,
    )
    print(json.dumps(result, sort_keys=True))
    ok = (
        result["failed"] == 0
        and result["leaked_results"] == 0
        and result["results"] > 0
        and result["anonymous_status"] == 401
        and result["unknown_user_results"] == 0
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
