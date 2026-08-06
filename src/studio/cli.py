"""kimodo_studio CLI — author a clip in Kimodo's viser demo, get a
dynamically feasible G1+object trajectory out of mppi_locoma."""

import argparse
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from . import manifest, pipeline
from .config import REPO_ROOT, RUNS_DIR, TASK_SUBTREE, Config, load_config
from .watch import watch_examples


def _run(cmd, env=None) -> None:
    print("+ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True,
                   env={**os.environ, **(env or {})})


def cmd_setup(cfg: Config, args) -> int:
    venv = cfg.kimodo_python.parent.parent
    if not cfg.kimodo_python.exists():
        _run(["uv", "venv", "--python", "3.11", venv])
    pip = ["uv", "pip", "install", "--python", cfg.kimodo_python]
    _run(pip + ["torch"])
    # base install only: the [demo] extra would pull the viser fork from git,
    # clobbering the editable install of the local clone below
    _run(pip + ["-e", cfg.kimodo_repo],
         env={"SKIP_MOTION_CORRECTION_IN_SETUP": "1"})
    local_viser = cfg.kimodo_repo / "kimodo-viser"
    if local_viser.is_dir():
        _run(pip + ["-e", local_viser])
    else:
        _run(pip + ["viser @ git+https://github.com/nv-tlabs/kimodo-viser.git"])
    print("\nsetup done. Notes:\n"
          f"- first `studio demo` downloads {cfg.model} from HuggingFace and\n"
          "  builds the viser web client (downloads Node 20; a few minutes)\n"
          "- the text encoder runs remotely: sbatch remote/encoder_euler.sh,\n"
          "  then `studio tunnel <node>`")
    return 0


def cmd_tunnel(cfg: Config, args) -> int:
    fwd = f"{cfg.encoder_port}:localhost:{cfg.encoder_port}"
    if args.node:
        cmd = ["ssh", "-N", "-J", cfg.encoder_host, "-L", fwd, args.node]
    else:
        cmd = ["ssh", "-N", "-L", fwd, cfg.encoder_host]
    print("+ " + " ".join(cmd) + "   (Ctrl+C to close)", flush=True)
    os.execvp("ssh", cmd)


def cmd_demo(cfg: Config, args) -> int:
    url = f"http://127.0.0.1:{cfg.encoder_port}/"
    try:
        urllib.request.urlopen(url, timeout=5)
    except urllib.error.HTTPError:
        pass  # any HTTP response means the tunnel is up
    except OSError:
        print(f"text encoder not reachable at {url}\n"
              "Bring it up first:\n"
              f"  1. on {cfg.encoder_host}:  sbatch remote/encoder_euler.sh"
              "   (job log prints the node)\n"
              "  2. here:            studio tunnel <node>\n"
              "then re-run: studio demo", file=sys.stderr)
        return 1
    demo_bin = cfg.kimodo_python.parent / "kimodo_demo"
    if not demo_bin.exists():
        print("kimodo venv missing — run `studio setup` first", file=sys.stderr)
        return 1
    env = dict(os.environ, TEXT_ENCODER_MODE="api", TEXT_ENCODER_URL=url,
               SERVER_PORT=str(cfg.demo_port))
    print(f"demo: http://127.0.0.1:{cfg.demo_port}\n"
          f"Save Example dir (watched by `studio watch`): {cfg.examples_dir}",
          flush=True)
    os.execve(str(demo_bin), [str(demo_bin), "--model", cfg.model], env)


def _resolve_example(cfg: Config, spec: str) -> Path | None:
    for cand in (Path(spec), cfg.examples_dir / spec):
        if cand.is_dir():
            return cand.resolve()
    return None


def cmd_run(cfg: Config, args) -> int:
    example = _resolve_example(cfg, args.example)
    if example is None:
        print(f"no example dir: {args.example} "
              f"(also tried under {cfg.examples_dir})", file=sys.stderr)
        return 1
    missing = [f for f in ("motion.npz", "meta.json")
               if not (example / f).exists()]
    if missing:
        print(f"{example} is not a Save Example dir (missing {missing})",
              file=sys.stderr)
        return 1
    return 0 if pipeline.process_example(cfg, example) else 1


def cmd_watch(cfg: Config, args) -> int:
    watch_examples(cfg, lambda ex: pipeline.process_example(cfg, ex))
    return 0


def _run_tasks(run_dir: Path):
    task_root = run_dir / "outputs" / TASK_SUBTREE
    return task_root, (sorted(d for d in task_root.iterdir() if d.is_dir())
                       if task_root.is_dir() else [])


def cmd_view(cfg: Config, args) -> int:
    run_dir = RUNS_DIR / args.name
    task_root, tasks = _run_tasks(run_dir)
    if not tasks:
        print(f"no built trial under {run_dir}", file=sys.stderr)
        return 1
    cmd = [str(cfg.mppi_python),
           str(cfg.mppi_locoma / "scripts/view_trial.py"), tasks[0].name,
           "--root", str(task_root), "--port", str(args.port)]
    print("+ " + " ".join(cmd), flush=True)
    os.execv(cmd[0], cmd)


def cmd_list(cfg: Config, args) -> int:
    rows = []
    for run_dir in sorted(RUNS_DIR.iterdir()) if RUNS_DIR.is_dir() else []:
        m = manifest.read(run_dir)
        if not m:
            continue
        prompt = m.get("prompt")
        if isinstance(prompt, list):
            prompt = " | ".join(prompt)
        rows.append((m.get("name", run_dir.name),
                     (m.get("created") or "")[:16],
                     m.get("verdict", "pending"),
                     (prompt or "")[:60]))
    if not rows:
        print("no runs yet")
        return 0
    print(f"{'run':24s} {'created':16s} {'verdict':8s} prompt")
    for name, created, verdict, prompt in rows:
        print(f"{name:24s} {created:16s} {verdict:8s} {prompt}")
    return 0


def cmd_promote(cfg: Config, args) -> int:
    run_dir = RUNS_DIR / args.name
    _, tasks = _run_tasks(run_dir)
    if not tasks:
        print(f"no built trial under {run_dir}", file=sys.stderr)
        return 1
    central = cfg.mppi_locoma / "outputs" / TASK_SUBTREE
    clashes = [t.name for t in tasks if (central / t.name).exists()]
    if clashes:
        print(f"refusing to overwrite existing central task(s): {clashes}",
              file=sys.stderr)
        return 1
    for t in tasks:
        shutil.copytree(t, central / t.name)
        print(f"promoted {t.name} -> {central / t.name}")
    print("now visible to mppi_locoma's eval_trials.py / export_dataset.py")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="studio",
        description="Kimodo viser demo -> mppi_locoma feasibility pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup", help="create the kimodo venv").set_defaults(
        func=cmd_setup)
    p = sub.add_parser("tunnel", help="SSH tunnel to the remote text encoder")
    p.add_argument("node", nargs="?", default=None,
                   help="compute node (jumps via the login host); omit to "
                        "tunnel to the login host itself")
    p.set_defaults(func=cmd_tunnel)
    sub.add_parser("demo", help="launch the Kimodo viser demo").set_defaults(
        func=cmd_demo)
    sub.add_parser("watch", help="auto-process new Save Example dirs"
                   ).set_defaults(func=cmd_watch)
    p = sub.add_parser("run", help="process one Save Example dir")
    p.add_argument("example", help="example dir path, or its name under the "
                                   "demo examples dir")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("view", help="viser view of reference vs. result")
    p.add_argument("name", help="run name (see `studio list`)")
    p.add_argument("--port", type=int, default=8081)
    p.set_defaults(func=cmd_view)
    sub.add_parser("list", help="list runs and verdicts").set_defaults(
        func=cmd_list)
    p = sub.add_parser("promote",
                       help="copy a run's trial(s) into mppi_locoma's "
                            "central outputs (paper dataset flow)")
    p.add_argument("name")
    p.set_defaults(func=cmd_promote)

    args = ap.parse_args()
    sys.exit(args.func(load_config(), args))
