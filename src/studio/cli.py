"""kimodo_studio CLI — author a clip in Kimodo's viser demo, get a
dynamically feasible G1+object trajectory out of mppi_locoma."""

import argparse
import os
import shutil
import subprocess
import sys
import time
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
    # accept-new: compute nodes rotate, and the connection already rides
    # through the trusted login host
    base = ["ssh", "-N", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ServerAliveInterval=30", "-o", "ExitOnForwardFailure=yes"]
    if args.node:
        cmd = base + ["-J", cfg.encoder_host, "-L", fwd, args.node]
    else:
        cmd = base + ["-L", fwd, cfg.encoder_host]
    print("+ " + " ".join(cmd) + "   (Ctrl+C to close)", flush=True)
    os.execvp("ssh", cmd)


def _encoder_up(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=5)
        return True
    except urllib.error.HTTPError:
        return True  # any HTTP response means something is serving
    except OSError:
        return False


def cmd_demo(cfg: Config, args) -> int:
    url = f"http://127.0.0.1:{cfg.encoder_port}/"
    demo_bin = cfg.kimodo_python.parent / "kimodo_demo"
    if not demo_bin.exists():
        print("kimodo venv missing — run `studio setup` first", file=sys.stderr)
        return 1
    env = dict(os.environ, TEXT_ENCODER_MODE="api", TEXT_ENCODER_URL=url,
               SERVER_PORT=str(cfg.demo_port))
    server = None

    if args.offline:
        if _encoder_up(url):
            print(f"port {cfg.encoder_port} is already serving — the tunnel "
                  "is up, run plain `studio demo` instead", file=sys.stderr)
            return 1
        emb_dir = REPO_ROOT / "offline_cache/embeddings"
        snapshots = sorted((REPO_ROOT / "offline_cache/snapshots").glob("*.npz"))
        server = subprocess.Popen(
            [str(cfg.kimodo_python), str(REPO_ROOT / "scripts/offline_encoder.py"),
             "--cache-dir", str(emb_dir), "--model", cfg.model_short,
             "--port", str(cfg.encoder_port),
             *(a for s in snapshots for a in ("--seed", str(s)))])
        for _ in range(60):
            if _encoder_up(url):
                break
            if server.poll() is not None:
                print("offline encoder server died at startup", file=sys.stderr)
                return 1
            time.sleep(1)
        else:
            server.terminate()
            print("offline encoder server did not come up", file=sys.stderr)
            return 1
        # isolated cache: offline zero-embeddings must never pollute the
        # real demo cache used by tunnel sessions
        env["kimodo_EMBED_CACHE_DIR"] = str(emb_dir)
        print(f"offline mode: known prompts from {len(snapshots)} snapshot(s) "
              "+ demo cache; NEW prompts will be IGNORED by generation "
              "(watch the server warnings)", flush=True)
    elif not _encoder_up(url):
        print(f"text encoder not reachable at {url}\n"
              "Bring it up first:\n"
              f"  1. on {cfg.encoder_host}:  sbatch remote/encoder_euler.sh"
              "   (job log prints the node)\n"
              "  2. here:            studio tunnel <node>\n"
              "then re-run: studio demo  (or use `studio demo --offline` "
              "for already-encoded prompts)", file=sys.stderr)
        return 1

    print(f"demo: http://127.0.0.1:{cfg.demo_port}\n"
          f"Save Example dir (watched by `studio watch`): {cfg.examples_dir}",
          flush=True)
    try:
        # the add-on wraps the stock demo: same UI + a scene-recon folder
        return subprocess.run(
            [str(cfg.kimodo_python),
             str(REPO_ROOT / "scripts/demo_scene_addon.py"),
             "--model", cfg.model], env=env).returncode
    finally:
        if server is not None:
            server.terminate()


def _resolve_example(cfg: Config, spec: str) -> Path | None:
    for cand in (Path(spec), cfg.examples_dir / spec):
        if cand.is_dir():
            return cand.resolve()
    return None


def cmd_run(cfg: Config, args) -> int:
    src = Path(args.example)
    if src.is_file() and src.suffix == ".npz":
        return 0 if pipeline.process_example(cfg, src.resolve()) else 1
    example = _resolve_example(cfg, args.example)
    if example is None:
        print(f"no example dir or .npz: {args.example} "
              f"(also tried under {cfg.examples_dir})", file=sys.stderr)
        return 1
    missing = [f for f in ("motion.npz", "meta.json")
               if not (example / f).exists()]
    if missing:
        print(f"{example} is not a Save Example dir (missing {missing})",
              file=sys.stderr)
        return 1
    return 0 if pipeline.process_example(cfg, example) else 1


def cmd_recon(cfg: Config, args) -> int:
    src = Path(args.source)
    if not (src.is_file() and src.suffix == ".npz"):
        resolved = _resolve_example(cfg, args.source)
        if resolved is None:
            print(f"no example dir or .npz: {args.source} "
                  f"(also tried under {cfg.examples_dir})", file=sys.stderr)
            return 1
        src = resolved
    flags = [f for f in args.build_flags if f != "--"]
    run_dir = pipeline.recon_example(cfg, src.resolve(), name=args.name,
                                     scene_flags=flags)
    return 0 if run_dir is not None else 1


def cmd_watch(cfg: Config, args) -> int:
    watch_examples(cfg, lambda ex: pipeline.process_example(cfg, ex))
    return 0


def _run_tasks(run_dir: Path):
    task_root = run_dir / "outputs" / TASK_SUBTREE
    return task_root, (sorted(d for d in task_root.iterdir() if d.is_dir())
                       if task_root.is_dir() else [])


def cmd_panel(cfg: Config, args) -> int:
    cmd = [str(cfg.mppi_python), str(REPO_ROOT / "scripts/panel_app.py"),
           "--port", str(args.port)]
    print("+ " + " ".join(cmd), flush=True)
    os.execv(cmd[0], cmd)


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
    p = sub.add_parser("demo", help="launch the Kimodo viser demo")
    p.add_argument("--offline", action="store_true",
                   help="no cluster: serve embeddings from snapshots + demo "
                        "cache; new prompts generate unconditioned motion")
    p.set_defaults(func=cmd_demo)
    sub.add_parser("watch", help="auto-process new Save Example dirs"
                   ).set_defaults(func=cmd_watch)
    p = sub.add_parser("run", help="process a Save Example dir or a bare "
                                   "motion .npz (demo's Save Motion)")
    p.add_argument("example", help="example dir path / its name under the "
                                   "demo examples dir / a motion .npz file")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser(
        "recon", help="scene reconstruction only (no solve): shim into "
                      "runs/<name>/ + build_trial; reuses the run dir so "
                      "scene params can be iterated")
    p.add_argument("source", help="motion .npz / example dir")
    p.add_argument("--name", default=None,
                   help="run name (default: derived from the source)")
    # unrecognized flags (e.g. --box-mass 0.5) pass through to build_trial.py
    p.set_defaults(func=cmd_recon, passthrough=True)
    p = sub.add_parser("panel", help="interactive viser panel: reconstruct + "
                                     "solve with tunable hyperparams")
    p.add_argument("--port", type=int, default=8082)
    p.set_defaults(func=cmd_panel)
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

    args, extra = ap.parse_known_args()
    if getattr(args, "passthrough", False):
        args.build_flags = [f for f in extra if f != "--"]
    elif extra:
        ap.error(f"unrecognized arguments: {' '.join(extra)}")
    sys.exit(args.func(load_config(), args))
