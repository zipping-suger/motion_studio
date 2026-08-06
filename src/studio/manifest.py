"""Per-run provenance manifest (runs/<name>/manifest.json)."""

import json
import subprocess
from pathlib import Path
from typing import Optional


def git_state(repo: Path) -> Optional[dict]:
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
        return {"sha": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def update(run_dir: Path, data: dict) -> Path:
    path = run_dir / "manifest.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update(data)
    path.write_text(json.dumps(merged, indent=2, default=str) + "\n")
    return path


def read(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}
