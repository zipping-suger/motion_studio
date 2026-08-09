"""Scene reconstruction -> SBMPC solve -> eval, for one clip.

Reconstruction (studio.recon) runs in-process: numpy + MuJoCo, no GPU.
The solve (studio.solve) needs torch + warp + SPIDER, so it runs as a
subprocess of the solve venv. Evaluation is numpy-only and in-process
again, which is why a verdict never needs the solve runtime.
"""

import json
import os
import re
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import manifest, recon, shim, solve
from .config import REPO_ROOT, RUNS_DIR, Config, task_dirs

# lines worth echoing from the (very chatty) solve log
_SOLVE_ECHO = re.compile(r"Final object|Traceback|Error|error", re.IGNORECASE)
_HEARTBEAT_LINES = 400


def _stream(cmd, cwd: Path, log_path: Path, env: dict | None = None,
            sparse: bool = False) -> int:
    """Run cmd, tee output to log_path; echo all lines, or only interesting
    ones plus a heartbeat when sparse."""
    proc = subprocess.Popen(
        [str(c) for c in cmd], cwd=str(cwd),
        # unbuffered child so echoed lines stream live instead of arriving
        # in block-buffered bursts
        env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    with open(log_path, "w") as log:
        for i, line in enumerate(proc.stdout):
            log.write(line)
            if not sparse:
                print(line, end="", flush=True)
            elif _SOLVE_ECHO.search(line):
                print(line, end="", flush=True)
            elif i and i % _HEARTBEAT_LINES == 0:
                print(f"  ... running ({i} log lines, see {log_path.name})",
                      flush=True)
    return proc.wait()


def _shim_source(cfg: Config, source: Path, run_dir: Path) -> Path:
    """Shim a Save Example dir OR a bare motion .npz into run_dir and
    record provenance in the manifest. Returns the motion dir."""
    if source.is_file():
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
            "kimodo_studio": manifest.git_state(REPO_ROOT),
        },
    })
    return motion_dir


def reconstruct(cfg: Config, run_dir: Path, motion_dir: Path,
                options: dict | None = None) -> list[Path]:
    """Stage 1 only: scene reconstruction into run_dir/outputs.

    studio.recon owns this outright, so it runs in-process — no venv hop,
    no GPU. `options` are forwarded to recon.reconstruct (scene_params,
    allow_held_start, pick, release). Returns task dirs ([] on
    failure/skip; the verdict lands in the manifest)."""
    name = run_dir.name
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / "build.log"

    # task naming keeps build_trial.py's convention: nested (Save Example
    # shim) vs flat (bare motion NPZ)
    npz = next(iter(sorted(motion_dir.glob("*.npz"))))
    nested = (motion_dir.parent / "batch_inputs").is_dir()
    task = (f"{name}_{motion_dir.name.split('_')[-1]}_00" if nested
            else f"{name}_00")

    print(f"=== scene reconstruction ({name}) ===")
    try:
        task_dir, _, line = recon.reconstruct(
            npz, run_dir / "outputs", task, **(options or {}))
        log_path.write_text(line + "\n")
    except Exception:
        task_dir = None
        line = f"{task}: reconstruction ERROR (see {log_path})"
        log_path.write_text(traceback.format_exc())
    print(line)

    if task_dir is None:
        verdict = "skipped" if "SKIPPED" in line else "error"
        manifest.update(run_dir, {"verdict": verdict})
        return []
    tasks = task_dirs(run_dir, prefix=f"{name}_")
    manifest.update(run_dir, {"tasks": [d.name for d in tasks]})
    return tasks


def solve_tasks(cfg: Config, run_dir: Path, tasks: list[Path],
                params: dict | None = None) -> bool:
    """Solve every built trial of a run, then record the verdict. Returns
    True iff a trial passes the LIFT criterion."""
    solve.require(cfg)
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)

    for task_dir in tasks:
        task = task_dir.name
        log_path = logs / f"solve_{task}.log"
        print(f"=== SBMPC solve: {task} ===")
        rc = _stream(
            solve.command(cfg, task, run_dir / "outputs", params),
            cwd=REPO_ROOT, log_path=log_path, env=solve.env(), sparse=True,
        )
        if rc != 0:
            print(f"solve failed (see {log_path})")
            manifest.update(run_dir, {"verdict": "error"})
            return False

    print("=== evaluation ===")
    rows = solve.evaluate.evaluate_tasks(tasks)
    print(solve.evaluate.format_table(rows))
    verdict = solve.evaluate.verdict(rows)
    manifest.update(run_dir, {
        "verdict": verdict,
        "solve_params": params or {},
        "eval_rows": [solve.evaluate.format_row(n, r) for n, r in rows],
    })
    print(f"verdict: {verdict}   (view with: studio view {run_dir.name})")
    return verdict == "LIFT"


def run_pipeline(cfg: Config, run_dir: Path, motion_dir: Path,
                 params: dict | None = None) -> bool:
    tasks = reconstruct(cfg, run_dir, motion_dir)
    if not tasks:
        return False
    return solve_tasks(cfg, run_dir, tasks, params)


def process_example(cfg: Config, source: Path) -> bool:
    """Shim a Save Example dir OR a bare motion .npz into runs/<name>/ and
    run the full pipeline. Returns True iff the retargeted clip passes the
    LIFT criterion."""
    name = shim.sanitize(source.stem if source.is_file() else source.name)
    run_dir = shim.make_run_dir(RUNS_DIR, name)
    motion_dir = _shim_source(cfg, source, run_dir)
    print(f"run dir: {run_dir}")
    return run_pipeline(cfg, run_dir, motion_dir)


def recon_example(cfg: Config, source: Path, name: str | None = None,
                  options: dict | None = None) -> Path | None:
    """Shim + scene reconstruction only, no solve. Unlike process_example
    this REUSES runs/<name>/ so scene params can be iterated on the same
    clip (the solve panel then picks the run up). Returns the run dir on
    success."""
    name = shim.sanitize(name or (source.stem if source.is_file()
                                  else source.name))
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    motion_dir = _shim_source(cfg, source, run_dir)
    if options:
        manifest.update(run_dir, {"recon_options": options})
    print(f"run dir: {run_dir}")
    tasks = reconstruct(cfg, run_dir, motion_dir, options)
    if not tasks:
        return None
    manifest.update(run_dir, {"verdict": "recon"})
    print(f"reconstructed: {', '.join(d.name for d in tasks)}   "
          f"(solve: studio panel)")
    return run_dir
