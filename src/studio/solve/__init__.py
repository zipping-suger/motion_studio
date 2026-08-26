"""The SBMPC solve: studio's rewards over SPIDER's sampling optimizer.

A solve is a function of one trial directory and a flat params dict —
`scene.xml`, `task_info.json` and `0/trajectory_kinematic.npz` in, the
solved `0/trajectory_mjwp.npz` out (see `spec.SolveScene` for what is
read from task_info, and `defaults` for the params). Nothing in here
knows a task by name: what a task changes about the solve arrives as
data in the trial (grips, faces, supports, symmetry) or as params.

Split by what each half imports: `evaluate.py`, `spec.py` and
`defaults.py` are numpy-only and run in studio's own venv, so verdicts
never need the solve runtime; `loop.py`, `rewards.py`, `rollout.py`,
`quat.py` and `spider_cfg.py` need torch/warp/SPIDER and run in the
solve venv as a subprocess — launched by `studio.runner`. This package
deliberately imports nothing from `studio` outside itself, so the solve
venv only needs `src/` on its path.
"""
