"""kimodo_studio CLI — author a clip in Kimodo's viser demo, reconstruct
the scene it implies, and solve for a dynamically feasible G1+object
trajectory."""

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import manifest, pipeline, recon, solve
from .config import (DEFAULT_KIMODO_VISER, REPO_ROOT, RUNS_DIR,
                     SCENE_DEFAULTS, SOLVE_DEFAULTS, SOLVE_INT_KEYS, Config,
                     load_config, task_dirs)
from .watch import watch_examples


def _run(cmd, env=None) -> None:
    print("+ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True,
                   env={**os.environ, **(env or {})})


def _spider(cfg: Config, explicit: str | None) -> Path:
    """A SPIDER checkout (or wheel). Needed only at setup: the G1 assets
    are Unitree-licensed and SPIDER is CC-BY-NC, so this repo redistributes
    neither — it copies what it needs from a checkout (by default the
    extern/spider submodule, initialized on demand)."""
    for candidate in (explicit, os.environ.get("SPIDER_ROOT"), cfg.spider):
        if candidate:
            return Path(candidate).expanduser()
    raise SystemExit(
        "no SPIDER install configured. Set paths.spider in config.yml, or:\n"
        "  studio setup --spider <checkout or .whl>")


def _checked_out(repo: Path) -> bool:
    # a submodule checkout has a .git *file*; an uninitialized one is an
    # empty dir
    return (repo / ".git").exists()


def _init_submodules(*repos: Path) -> None:
    """Fresh clones made without --recurse-submodules leave extern/ empty;
    fetch the pinned checkouts before a stage needs them. Paths outside
    extern/ (config.yml overrides, wheels) are left alone."""
    extern = REPO_ROOT / "extern"
    need = sorted({str(r) for r in repos
                   if r is not None and extern in r.parents
                   and not _checked_out(r)})
    if need:
        _run(["git", "-C", REPO_ROOT, "submodule", "update", "--init",
              "--depth", "1", *need])


def _setup_assets(cfg: Config, spider: Path) -> None:
    print("== G1 assets ==")
    n = recon.assets.install(spider)
    print(f"installed {n} file(s) into {recon.assets.G1_DIR}")


def _setup_kimodo(cfg: Config) -> None:
    print("\n== kimodo venv (the demo runtime) ==")
    venv = cfg.kimodo_python.parent.parent
    if not cfg.kimodo_python.exists():
        _run(["uv", "venv", "--python", "3.11", venv])
    pip = ["uv", "pip", "install", "--python", cfg.kimodo_python]
    _run(pip + ["torch"])
    # base install only: the [demo] extra would pull the viser fork from git,
    # clobbering the editable install of the local clone below
    _run(pip + ["-e", cfg.kimodo_repo],
         env={"SKIP_MOTION_CORRECTION_IN_SETUP": "1"})
    # the demo needs the kimodo-viser fork; prefer a local checkout (a
    # kimodo checkout with the fork cloned inside it, else the extern/
    # submodule) so it installs editable and pinned
    for local_viser in (cfg.kimodo_repo / "kimodo-viser", DEFAULT_KIMODO_VISER):
        if _checked_out(local_viser):
            _run(pip + ["-e", local_viser])
            break
    else:
        _run(pip + ["viser @ git+https://github.com/nv-tlabs/kimodo-viser.git"])


def _setup_solve(cfg: Config, spider: Path) -> None:
    """The solve runtime: SPIDER pins its own torch, warp and mujoco-warp,
    which is exactly why it gets its own venv instead of studio's."""
    print("\n== solve venv (torch + warp + SPIDER) ==")
    venv = cfg.solve_python.parent.parent
    if not cfg.solve_python.exists():
        # SPIDER requires >= 3.12
        _run(["uv", "venv", "--python", "3.12", venv])
    _run(["uv", "pip", "install", "--python", cfg.solve_python,
          # warp-lang / mujoco-warp are served from NVIDIA's index; uv
          # sources are not transitive, so the index is named again here
          "--extra-index-url", "https://pypi.nvidia.com/",
          "--index-strategy", "unsafe-best-match",
          str(spider),
          # the panel draws URDF + convex hulls on top of the solve venv
          "trimesh", "yourdfpy"])


def cmd_setup(cfg: Config, args) -> int:
    stages = [s for s in ("assets", "kimodo", "solve") if getattr(args, s)]
    stages = stages or ["assets", "kimodo", "solve"]
    spider = _spider(cfg, args.spider) if {"assets", "solve"} & set(stages) \
        else None

    wanted = [spider]
    if "kimodo" in stages:
        wanted.append(cfg.kimodo_repo)
        if not (cfg.kimodo_repo / "kimodo-viser").is_dir():
            wanted.append(DEFAULT_KIMODO_VISER)
    _init_submodules(*wanted)

    if "assets" in stages:
        _setup_assets(cfg, spider)
    if "kimodo" in stages:
        _setup_kimodo(cfg)
    if "solve" in stages:
        _setup_solve(cfg, spider)

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


def _recon_options(args) -> dict:
    """CLI flags -> recon.reconstruct kwargs, carrying only what was set
    so unspecified params keep their SCENE_DEFAULTS values."""
    scene_params = {k: getattr(args, k) for k in SCENE_DEFAULTS
                    if getattr(args, k, None) is not None}
    options = {"allow_held_start": args.allow_held_start}
    if scene_params:
        options["scene_params"] = scene_params
    if args.pick_frame is not None:
        options["pick"] = args.pick_frame
    if args.release_frame is not None:
        options["release"] = args.release_frame
    return options


def cmd_recon(cfg: Config, args) -> int:
    src = Path(args.source)
    if not (src.is_file() and src.suffix == ".npz"):
        resolved = _resolve_example(cfg, args.source)
        if resolved is None:
            print(f"no example dir or .npz: {args.source} "
                  f"(also tried under {cfg.examples_dir})", file=sys.stderr)
            return 1
        src = resolved
    run_dir = pipeline.recon_example(cfg, src.resolve(), name=args.name,
                                     options=_recon_options(args))
    return 0 if run_dir is not None else 1


def cmd_watch(cfg: Config, args) -> int:
    watch_examples(cfg, lambda ex: pipeline.process_example(cfg, ex))
    return 0


def cmd_panel(cfg: Config, args) -> int:
    solve.require(cfg)
    cmd = [str(cfg.solve_python), str(REPO_ROOT / "scripts/panel_app.py"),
           "--port", str(args.port)]
    if getattr(args, "name", None):
        cmd += ["--run", args.name]
    print("+ " + " ".join(cmd), flush=True)
    os.execve(cmd[0], cmd, {**os.environ, **solve.env()})


def cmd_solve(cfg: Config, args) -> int:
    run_dir = RUNS_DIR / args.name
    tasks = task_dirs(run_dir)
    if not tasks:
        print(f"no built trial under {run_dir} — reconstruct it first "
              f"(studio recon ...)", file=sys.stderr)
        return 1
    params = {k: getattr(args, k) for k in SOLVE_DEFAULTS
              if getattr(args, k, None) is not None}
    return 0 if pipeline.solve_tasks(cfg, run_dir, tasks, params) else 1


def cmd_view(cfg: Config, args) -> int:
    """Reference vs. result. The solve panel already draws exactly this,
    so `view` opens it on an existing run rather than shipping a second
    viewer."""
    run_dir = RUNS_DIR / args.name
    if not task_dirs(run_dir):
        print(f"no built trial under {run_dir}", file=sys.stderr)
        return 1
    return cmd_panel(cfg, args)


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
    tasks = task_dirs(run_dir)
    if not tasks:
        print(f"no built trial under {run_dir}", file=sys.stderr)
        return 1
    central = Path(args.to) if args.to else cfg.dataset_outputs
    if central is None:
        print("no destination: pass --to <dir>, or set paths.dataset_outputs "
              "in config.yml", file=sys.stderr)
        return 1
    central.mkdir(parents=True, exist_ok=True)
    clashes = [t.name for t in tasks if (central / t.name).exists()]
    if clashes:
        print(f"refusing to overwrite existing central task(s): {clashes}",
              file=sys.stderr)
        return 1
    for t in tasks:
        shutil.copytree(t, central / t.name)
        print(f"promoted {t.name} -> {central / t.name}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="studio",
        description="Kimodo viser demo -> scene recon -> SBMPC feasibility")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("setup", help="G1 assets + the kimodo and solve venvs "
                                     "(all three unless a stage is named)")
    p.add_argument("--assets", action="store_true", help="G1 meshes + URDF")
    p.add_argument("--kimodo", action="store_true", help="the demo runtime")
    p.add_argument("--solve", action="store_true",
                   help="the solve runtime: torch + warp + SPIDER")
    p.add_argument("--spider", default=None,
                   help="SPIDER checkout or .whl (default: paths.spider in "
                        "config.yml, or $SPIDER_ROOT)")
    p.set_defaults(func=cmd_setup)
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
                      "runs/<name>/ + reconstruct; reuses the run dir so "
                      "scene params can be iterated")
    p.add_argument("source", help="motion .npz / example dir")
    p.add_argument("--name", default=None,
                   help="run name (default: derived from the source)")
    for key, val in SCENE_DEFAULTS.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(val, str):
            p.add_argument(flag, choices=("capsule", "mesh"), default=None,
                           help=f"scene param (default: {val})")
        else:
            p.add_argument(flag, type=float, default=None,
                           help=f"scene param (default: {val})")
    p.add_argument("--allow-held-start", action="store_true",
                   help="let the detector accept a clip that begins mid-hold")
    p.add_argument("--pick-frame", type=int, default=None,
                   help="force the contact window start")
    p.add_argument("--release-frame", type=int, default=None,
                   help="force the contact window end")
    p.set_defaults(func=cmd_recon)
    p = sub.add_parser("solve", help="SBMPC solve for an already "
                                     "reconstructed run")
    p.add_argument("name", help="run name (see `studio list`)")
    for key, val in SOLVE_DEFAULTS.items():
        p.add_argument("--" + key.replace("_", "-"),
                       type=int if key in SOLVE_INT_KEYS else float,
                       default=None, help=f"solve param (default: {val})")
    p.set_defaults(func=cmd_solve)
    p = sub.add_parser("panel", help="interactive viser panel: reconstruct + "
                                     "solve with tunable hyperparams")
    p.add_argument("--port", type=int, default=8082)
    p.add_argument("--run", dest="name", default=None,
                   help="open on an existing run instead of a raw clip")
    p.set_defaults(func=cmd_panel)
    p = sub.add_parser("view", help="reference vs. result (opens the panel "
                                    "on that run)")
    p.add_argument("name", help="run name (see `studio list`)")
    p.add_argument("--port", type=int, default=8082)
    p.set_defaults(func=cmd_view)
    sub.add_parser("list", help="list runs and verdicts").set_defaults(
        func=cmd_list)
    p = sub.add_parser("promote",
                       help="copy a run's trial(s) into a central dataset dir")
    p.add_argument("name")
    p.add_argument("--to", default=None,
                   help="destination (default: paths.dataset_outputs)")
    p.set_defaults(func=cmd_promote)

    args = ap.parse_args()
    sys.exit(args.func(load_config(), args))
