#!/usr/bin/env bash
# Stable managed release resolver, protocol version 1.
set -euo pipefail
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
base="${XDG_DATA_HOME:-$HOME/.local/share}/hyprwhspr"
# Only stdlib is needed to read the generation and route recovery commands.
recorded_python=""
if [[ -f "$base/interpreter" ]]; then IFS= read -r recorded_python < "$base/interpreter" || true; fi
for python in "$recorded_python" /usr/bin/python3 /usr/local/bin/python3; do
    if [[ -x "$python" ]] && "$python" -I -B -c 'import sys; sys.exit(not ((3, 11) <= sys.version_info[:2] <= (3, 14)))' 2>/dev/null; then
        exec "$python" -I -B -c '
import json,os,runpy,sys
from pathlib import Path
base=Path(sys.argv.pop(1))
try:
    generation=json.loads((base/"current.json").read_text(encoding="utf-8"))
    os.environ["HYPRWHSPR_RESOLVED_GENERATION"]=json.dumps(generation)
    script=Path(generation["root"])/"lib/src/managed_install.py"
    sys.path.insert(0,str(script.parent))
    sys.argv=[str(script),"launch",*sys.argv[1:]]
    runpy.run_path(str(script),run_name="__main__")
except (OSError,ValueError,KeyError) as exc:
    sys.exit(f"Managed installation unavailable: {exc}. Run bootstrap --repair.")
' "$base" "$@"
    fi
done
echo 'No usable Python remains. Run bootstrap --repair --python PATH.' >&2
exit 1
