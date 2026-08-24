"""Benchmark a solve's per-tick wall time without touching run data.

Copies the trial to a scratch dir, runs the solve loop there in the solve
venv, parses the per-tick ``Realtime rate: R ... opt_steps: K`` lines, and
stops the solve after --ticks ticks. Reports the mean realtime rate and the
per-opt-step time (tick time / opt_steps) — the latter stays comparable
across code versions even when the improvement early-exit settles on a
different number of annealing iterations per tick.

Usage:
  # obstacle tasks (under_table / kick, studio.solve.mppi_loop):
  python scripts/bench_solve.py --task-dir runs/<run>/outputs/processed/.../<trial>

  # box_carry / ground_pick / pole (studio.solve.loop):
  python scripts/bench_solve.py --run runs/<run> --task <trial_name>
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from studio import solve                                    # noqa: E402
from studio.config import REPO_ROOT, load_config            # noqa: E402

CTRL_DT = 0.1  # both loops commit one 0.1 s control tick per rate line
TICK_RE = re.compile(r"Realtime rate: ([0-9.]+).*opt_steps: (\d+)")


def run_bench(cmd: list[str], ticks: int, warmup: int) -> int:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", **solve.env()}
    proc = subprocess.Popen([str(c) for c in cmd], cwd=str(REPO_ROOT),
                            env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    rates, opt_steps = [], []
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            m = TICK_RE.search(line)
            if not m:
                continue
            rates.append(float(m.group(1)))
            opt_steps.append(int(m.group(2)))
            if len(rates) >= ticks:
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    if len(rates) <= warmup:
        print(f"\nonly {len(rates)} ticks captured (need > {warmup})")
        return 1

    r, o = rates[warmup:], opt_steps[warmup:]
    tick_times = [CTRL_DT / x for x in r]
    per_opt = [t / k for t, k in zip(tick_times, o)]
    print(f"\nticks measured: {len(r)} (skipped {warmup} warmup)")
    print(f"realtime rate:  mean {statistics.mean(r):.3f}  "
          f"median {statistics.median(r):.3f}  "
          f"stdev {statistics.pstdev(r):.3f}")
    print(f"tick time:      mean {statistics.mean(tick_times):.3f}s")
    print(f"opt steps/tick: mean {statistics.mean(o):.1f}")
    print(f"time/opt-step:  mean {statistics.mean(per_opt) * 1e3:.1f}ms  "
          f"median {statistics.median(per_opt) * 1e3:.1f}ms")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task-dir", type=Path,
                    help="a built obstacle trial dir (mppi_loop)")
    ap.add_argument("--run", type=Path,
                    help="a run dir holding outputs/ (loop)")
    ap.add_argument("--task", help="trial name inside --run")
    ap.add_argument("--ticks", type=int, default=10,
                    help="control ticks to measure (default 10)")
    ap.add_argument("--warmup", type=int, default=2,
                    help="leading ticks to drop from stats (default 2)")
    ap.add_argument("--param", action="append", default=[],
                    metavar="KEY=VALUE", help="forwarded to the solve loop")
    args = ap.parse_args()
    if bool(args.task_dir) == bool(args.run):
        ap.error("pass exactly one of --task-dir or (--run + --task)")
    if args.run and not args.task:
        ap.error("--run needs --task <trial_name>")

    cfg = load_config()
    solve.require(cfg)
    scratch = Path(tempfile.mkdtemp(prefix="bench_solve_"))
    try:
        if args.task_dir:
            trial = scratch / args.task_dir.name
            shutil.copytree(args.task_dir, trial)
            cmd = [cfg.solve_python, "-m", "studio.solve.mppi_loop",
                   "--task-dir", trial]
        else:
            outputs = scratch / "outputs"
            shutil.copytree(args.run / "outputs", outputs)
            cmd = [cfg.solve_python, "-m", "studio.solve.loop",
                   "--task", args.task, "--dataset-dir", outputs]
        for p in args.param:
            cmd += ["--param", p]
        return run_bench(cmd, args.ticks, args.warmup)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
