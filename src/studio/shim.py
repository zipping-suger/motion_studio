"""Reshape a demo "Save Example" dir into mppi_locoma's nested batch layout.

Save Example writes a flat {motion.npz, constraints.json, meta.json};
build_trial.py wants <batch>/motion_XXXX/*.npz plus
<batch>/batch_inputs/motion_XXXX/{constraints.json, meta.json}, and names
the task <batch>_<XXXX>_<clip>. The run dir acts as the batch, so run
"foo" yields task foo_0000_00.
"""

import re
import shutil
from pathlib import Path

import numpy as np


def sanitize(name: str) -> str:
    # Run name becomes a hydra task= override and a dir name; lowercase to
    # match build_trial's task naming so task == <name>_0000_00 exactly.
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_").lower()
    return clean or "clip"


def make_run_dir(runs_root: Path, name: str) -> Path:
    run_dir = runs_root / name
    n = 2
    while run_dir.exists():
        run_dir = runs_root / f"{name}-{n}"
        n += 1
    run_dir.mkdir(parents=True)
    return run_dir


def _copy_motion_npz(src: Path, dst: Path) -> None:
    """Copy a motion NPZ, normalizing a demo-save quirk: the demo writes
    foot_contacts as float probabilities, but mppi_locoma's contact graph
    needs bools (batch_generate's output dtype)."""
    data = dict(np.load(src))
    contacts = data.get("foot_contacts")
    if contacts is not None and contacts.dtype != np.bool_:
        data["foot_contacts"] = contacts > 0.5
        np.savez(dst, **data)
    else:
        shutil.copy2(src, dst)


def shim_example(example_dir: Path, run_dir: Path) -> Path:
    motion_dir = run_dir / "motion_0000"
    motion_dir.mkdir()
    _copy_motion_npz(example_dir / "motion.npz",
                     motion_dir / "motion_0000_00.npz")
    inputs = run_dir / "batch_inputs/motion_0000"
    inputs.mkdir(parents=True)
    shutil.copy2(example_dir / "meta.json", inputs / "meta.json")
    constraints = example_dir / "constraints.json"
    if constraints.exists():
        shutil.copy2(constraints, inputs / "constraints.json")
    return motion_dir


def shim_motion_npz(npz_path: Path, run_dir: Path) -> Path:
    """Bare motion NPZ (demo's "Save Motion") -> flat batch layout.

    No batch_inputs dir, so build_trial takes its flat naming path
    (<dir>_<clip>); grasp detection runs purely from kinematics."""
    motion_dir = run_dir / run_dir.name
    motion_dir.mkdir()
    _copy_motion_npz(npz_path, motion_dir / f"{run_dir.name}_00.npz")
    return motion_dir
