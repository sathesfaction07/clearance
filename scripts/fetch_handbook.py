"""Download the six handbook sections Clearance uses into data/raw/handbook/md (git-ignored).

Source: gitlab.com/gitlab-com/content-sites/handbook (public). Only Markdown is kept, under a
short root (Windows path limits), with tar paths checked for absolute paths and traversal.
"""

import io
import posixpath
import sys
import tarfile
import urllib.request
from pathlib import Path

API = "https://gitlab.com/api/v4/projects/42817607/repository/archive.tar.gz?sha=main&path="
SECTIONS = ["people-policies", "total-rewards", "security", "legal", "finance", "support"]
LIMIT = 50 * 1024 * 1024


def main() -> int:
    out = (Path(__file__).resolve().parents[1] / "data" / "raw" / "handbook" / "md").resolve()
    out.mkdir(parents=True, exist_ok=True)
    for section in SECTIONS:
        with urllib.request.urlopen(API + "content/handbook/" + section) as response:
            data = response.read(LIMIT + 1)
        if len(data) > LIMIT:
            raise SystemExit(f"{section}: archive over the 50 MB limit")
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            for m in tf.getmembers():
                name = posixpath.normpath(m.name)
                if not m.isfile() or not name.endswith(".md") or "content/handbook/" not in name:
                    continue
                dest = (out / name.split("content/handbook/", 1)[1]).resolve()
                if out not in dest.parents:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(tf.extractfile(m).read())
        print(f"fetched {section}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
