"""Per-run provenance manifests."""

import json

from studio import manifest


def test_read_missing_manifest_is_empty(tmp_path):
    assert manifest.read(tmp_path) == {}


def test_update_creates_then_merges(tmp_path):
    manifest.update(tmp_path, {"name": "clip", "verdict": "pending"})
    manifest.update(tmp_path, {"verdict": "LIFT", "tasks": ["clip_00"]})

    assert manifest.read(tmp_path) == {
        "name": "clip", "verdict": "LIFT", "tasks": ["clip_00"]}


def test_update_serializes_paths(tmp_path):
    """Provenance carries Paths; json needs default=str to not choke."""
    manifest.update(tmp_path, {"source": tmp_path / "motion.npz"})
    assert manifest.read(tmp_path)["source"] == str(tmp_path / "motion.npz")


def test_manifest_is_human_readable(tmp_path):
    manifest.update(tmp_path, {"name": "clip"})
    text = (tmp_path / "manifest.json").read_text()
    assert text.endswith("\n")
    assert "\n  " in text          # indented, not one long line
    assert json.loads(text) == {"name": "clip"}


def _git(tmp_path, *args):
    import subprocess
    subprocess.run(["git", "-C", str(tmp_path), *args], check=True,
                   capture_output=True)


def test_git_state_records_sha_and_dirtiness(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "f.txt").write_text("hi")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "first")

    clean = manifest.git_state(tmp_path)
    assert clean is not None
    assert len(clean["sha"]) == 40 and clean["dirty"] is False

    (tmp_path / "f.txt").write_text("changed")
    assert manifest.git_state(tmp_path)["dirty"] is True


def test_git_state_of_a_non_repo_is_none(tmp_path):
    assert manifest.git_state(tmp_path) is None


def test_git_state_before_the_first_commit_is_none(tmp_path):
    """An unborn HEAD has no SHA to record — provenance is simply absent
    rather than half-filled."""
    _git(tmp_path, "init", "-q")
    assert manifest.git_state(tmp_path) is None
