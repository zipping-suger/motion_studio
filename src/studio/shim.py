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


def shim_example(example_dir: Path, run_dir: Path) -> Path:
    motion_dir = run_dir / "motion_0000"
    motion_dir.mkdir()
    shutil.copy2(example_dir / "motion.npz", motion_dir / "motion_0000_00.npz")
    inputs = run_dir / "batch_inputs/motion_0000"
    inputs.mkdir(parents=True)
    shutil.copy2(example_dir / "meta.json", inputs / "meta.json")
    constraints = example_dir / "constraints.json"
    if constraints.exists():
        shutil.copy2(constraints, inputs / "constraints.json")
    return motion_dir
