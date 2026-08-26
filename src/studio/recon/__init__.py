"""Scene reconstruction: a robot-only clip -> a physics trial.

Detect when and how the robot interacts with an object, decide what the
object is and where it goes, pose the hands on it, and emit a trial the
solve can run. The pipeline every task runs through is `run.py`; the
records it passes around and the one trial writer are `spec.py`; a
task is a `run.ReconTask` (box_carry.py, pick.py, pole.py, chair.py)
composed from the shared kernels — `mjcf` (scene template surgery),
`robot` (the robot-only model), `objects` (the mesh/shape roster),
`grasp` / `graph` (bimanual grasp detection and SceneBot's interaction
graph), `pick` (grasp calibration, wrist IK), `loader` / `layout` (the
Kimodo clip format and the qpos layouts).

Needs numpy + mujoco and the G1 assets; no solver and no GPU, so it
runs in studio's own venv. `objects` is MuJoCo-free so the GUIs can
read the rosters. This package's __init__ imports nothing: importing a
submodule never drags MuJoCo into a venv that lacks it.
"""
