#!/usr/bin/env python3
"""
Update the pinned OSV-Scanner version and checksum in a workflow file.

Usage:
    python update_scanner_pin.py <workflow_path> <new_version> <new_sha256>

Exit codes:
    0 - both lines updated and verified
    1 - invalid arguments or update failed
"""

import re
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: update_scanner_pin.py <workflow_path> <new_version> <new_sha256>",
            file=sys.stderr,
        )
        sys.exit(1)

    workflow_path = Path(sys.argv[1])
    new_version = sys.argv[2]
    new_sha = sys.argv[3]

    if not re.fullmatch(r"\d+\.\d+\.\d+", new_version):
        print(f"Error: version {new_version!r} is not plain X.Y.Z", file=sys.stderr)
        sys.exit(1)
    if not re.fullmatch(r"[a-f0-9]{64}", new_sha):
        print(f"Error: checksum {new_sha!r} is not a 64-char lowercase sha256", file=sys.stderr)
        sys.exit(1)
    if not workflow_path.exists():
        print(f"Error: {workflow_path} not found", file=sys.stderr)
        sys.exit(1)

    content = workflow_path.read_text()
    content, n_ver = re.subn(r'OSV_VERSION="[^"]*"', f'OSV_VERSION="{new_version}"', content)
    content, n_sha = re.subn(r'EXPECTED_SHA="[^"]*"', f'EXPECTED_SHA="{new_sha}"', content)
    if n_ver != 1 or n_sha != 1:
        print(
            f"Error: expected exactly one OSV_VERSION and one EXPECTED_SHA line, "
            f"replaced {n_ver} and {n_sha}",
            file=sys.stderr,
        )
        sys.exit(1)

    workflow_path.write_text(content)
    print(f"Pinned OSV-Scanner v{new_version} (sha256 {new_sha[:12]}...)")


if __name__ == "__main__":
    main()
