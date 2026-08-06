"""Scene reconstruction -> SPIDER retargeting -> eval for one example.

Every stage is a subprocess of mppi_locoma's own .venv, mirroring its
scripts/run_pipeline.sh; retarget hyperparameters stay single-sourced in
mppi_locoma/config.yml via repo_config.retarget_overrides().
"""

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import manifest, shim
from .config import REPO_ROOT, RUNS_DIR, TASK_SUBTREE, Config

# lines worth echoing from the (very chatty) retargeting log
_RETARGET_ECHO = re.compile(r"Final object|Traceback|Error|error", re.IGNORECASE)
_HEARTBEAT_LINES = 400


def _stream(cmd, cwd: Path, log_path: Path, env: dict | None = None,
            sparse: bool = False) -> int:
    """Run cmd, tee output to log_path; echo all lines, or only interesting
    ones plus a heartbeat when sparse."""
    proc = subprocess.Popen(
        [str(c) for c in cmd], cwd=str(cwd),
        env={**os.environ, **(env or {})},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    with open(log_path, "w") as log:
        for i, line in enumerate(proc.stdout):
            log.write(line)
            if not sparse:
                print(line, end="", flush=True)
            elif _RETARGET_ECHO.search(line):
                print(line, end="", flush=True)
            elif i and i % _HEARTBEAT_LINES == 0:
                print(f"  ... running ({i} log lines, see {log_path.name})",
                      flush=True)
    return proc.wait()


def retarget_overrides(cfg: Config) -> list[str]:
    code = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "from repo_config import retarget_overrides; "
            "print('\\n'.join(retarget_overrides()))")
    out = subprocess.run(
        [str(cfg.mppi_python), "-c", code, str(cfg.mppi_locoma / "src")],
        capture_output=True, text=True, check=True,
    )
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def process_example(cfg: Config, source: Path) -> bool:
    """Shim a Save Example dir OR a bare motion .npz into runs/<name>/ and
    run the full pipeline. Returns True iff the retargeted clip passes the
    LIFT criterion."""
    is_npz = source.is_file()
    name = shim.sanitize(source.stem if is_npz else source.name)
    run_dir = shim.make_run_dir(RUNS_DIR, name)
    if is_npz:
        motion_dir = shim.shim_motion_npz(source, run_dir)
        prompt = seed = None
    else:
        motion_dir = shim.shim_example(source, run_dir)
        meta = json.loads((source / "meta.json").read_text())
        prompt = meta.get("text") or meta.get("texts")
        seed = meta.get("seed")

    manifest.update(run_dir, {
        "name": run_dir.name,
        "source": str(source),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt": prompt,
        "seed": seed,
        "git": {
            "kimodo": manifest.git_state(cfg.kimodo_repo),
            "mppi_locoma": manifest.git_state(cfg.mppi_locoma),
            "kimodo_studio": manifest.git_state(REPO_ROOT),
        },
        "mppi_config": yaml.safe_load((cfg.mppi_locoma / "config.yml").read_text()),
    })
    print(f"run dir: {run_dir}")
    return run_pipeline(cfg, run_dir, motion_dir)


def run_pipeline(cfg: Config, run_dir: Path, motion_dir: Path) -> bool:
    name = run_dir.name
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)
    out_root = run_dir / "outputs"
    task_root = out_root / TASK_SUBTREE

    print(f"=== 1/3 scene reconstruction ({name}) ===")
    rc = _stream(
        [cfg.mppi_python, cfg.mppi_locoma / "scripts/build_trial.py",
         "--motion-dir", motion_dir, "--out-root", out_root],
        cwd=cfg.mppi_locoma, log_path=logs / "build.log",
    )
    task_dirs = sorted(task_root.glob(f"{name}_*")) if task_root.exists() else []
    if rc != 0 or not task_dirs:
        verdict = "error" if rc != 0 else "skipped"
        print(f"reconstruction {verdict}"
              + ("" if rc == 0 else f" (see {logs / 'build.log'})"))
        manifest.update(run_dir, {"verdict": verdict})
        return False
    manifest.update(run_dir, {"tasks": [d.name for d in task_dirs]})

    overrides = retarget_overrides(cfg)
    for task_dir in task_dirs:
        task = task_dir.name
        print(f"=== 2/3 SPIDER retargeting: {task} ===")
        rc = _stream(
            [cfg.mppi_python, cfg.mppi_locoma / "retarget/run_mjwp.py",
             "+override=kimodo_pick", f"task={task}", f"dataset_dir={out_root}",
             *overrides, "show_viewer=false", "+wait_on_finish=false",
             "save_video=false", "+sanity_check_seconds=0.0"],
            cwd=cfg.mppi_locoma, log_path=logs / f"retarget_{task}.log",
            env={"MUJOCO_GL": "egl"}, sparse=True,
        )
        if rc != 0:
            print(f"retargeting failed (see {logs / f'retarget_{task}.log'})")
            manifest.update(run_dir, {"verdict": "error"})
            return False

    print("=== 3/3 evaluation ===")
    out = subprocess.run(
        [str(cfg.mppi_python), str(cfg.mppi_locoma / "scripts/eval_trials.py"),
         "--root", str(task_root), "--filter", f"{name}_"],
        cwd=str(cfg.mppi_locoma), capture_output=True, text=True,
    ).stdout
    print(out)

    rows = [ln for ln in out.splitlines()
            if any(ln.startswith(d.name) for d in task_dirs)]
    lifted = any(" YES" in ln for ln in rows)
    manifest.update(run_dir, {
        "verdict": "LIFT" if lifted else "failed",
        "eval_rows": rows,
    })
    print(f"verdict: {'LIFT' if lifted else 'failed'}   "
          f"(view with: studio view {name})")
    return lifted
