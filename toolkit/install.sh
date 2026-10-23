#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tracker_prefix="${1:?Usage: bash install.sh /absolute/path/to/new-venv}"
python3 -c 'import platform,sys; assert sys.platform == "linux" and platform.machine() in ("aarch64","arm64") and sys.version_info[:2] == (3,8), "Requires DOHC Linux ARM64 system CPython 3.8"'
python3 "$bundle_dir/verify_bundle.py"
if [[ -e "$tracker_prefix" ]]; then
    echo "Refusing to overwrite an existing environment: $tracker_prefix" >&2
    exit 1
fi
python3 -m venv "$tracker_prefix"
"$tracker_prefix/bin/python" -m pip install --no-index --require-hashes --find-links "$bundle_dir/wheelhouse" -r "$bundle_dir/requirements.lock"
"$tracker_prefix/bin/python" -m pip check
"$tracker_prefix/bin/dohc-ir" doctor
echo "Installed in $tracker_prefix. Confirm config/ optics before starting a session."
