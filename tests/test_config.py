"""Config loading and the run-dir path math."""

import pytest

from studio import config


def _config_yaml(kimodo="/a/kimodo", spider=None, model="Kimodo-G1-RP-v1",
                 kimodo_python=None, solve_python=None):
    lines = ["paths:", f"  kimodo_repo: {kimodo}"]
    for key, value in (("spider", spider), ("kimodo_python", kimodo_python),
                       ("solve_python", solve_python)):
        if value:
            lines.append(f"  {key}: {value}")
    lines += ["demo:", f"  model: {model}", "  port: 7860",
              "encoder:", "  remote_host: euler", "  port: 9550"]
    return "\n".join(lines) + "\n"


@pytest.fixture
def write_config(tmp_path, monkeypatch):
    def _write(**kwargs):
        path = tmp_path / "config.yml"
        path.write_text(_config_yaml(**kwargs))
        monkeypatch.setenv("STUDIO_CONFIG", str(path))
        return path
    return _write


def test_loads_from_studio_config_env(write_config):
    write_config(kimodo="/a/kimodo", spider="/b/spider")
    cfg = config.load_config()
    assert cfg.kimodo_repo.as_posix() == "/a/kimodo"
    assert cfg.spider.as_posix() == "/b/spider"
    assert cfg.demo_port == 7860
    assert cfg.encoder_host == "euler"
    assert cfg.encoder_port == 9550


def test_paths_expand_tilde_and_vars(write_config, monkeypatch):
    monkeypatch.setenv("SPIDER_HOME", "/opt/spider")
    monkeypatch.setenv("HOME", "/home/someone")
    write_config(kimodo="~/src/kimodo", spider="$SPIDER_HOME/checkout")
    cfg = config.load_config()
    assert cfg.kimodo_repo.as_posix() == "/home/someone/src/kimodo"
    assert cfg.spider.as_posix() == "/opt/spider/checkout"


def test_optional_paths_have_defaults(write_config):
    """spider defaults to the extern/ submodule; dataset_outputs (promote
    only) has no sensible default."""
    write_config()
    cfg = config.load_config()
    assert cfg.spider == config.DEFAULT_SPIDER
    assert cfg.dataset_outputs is None


def test_kimodo_python_defaults_to_the_setup_location(write_config):
    write_config()
    assert config.load_config().kimodo_python == config.DEFAULT_KIMODO_PYTHON


def test_kimodo_python_can_be_overridden(write_config):
    write_config(kimodo_python="/venvs/kimodo/bin/python")
    cfg = config.load_config()
    assert cfg.kimodo_python.as_posix() == "/venvs/kimodo/bin/python"


def test_missing_config_yields_submodule_defaults(tmp_path, monkeypatch):
    """A fresh clone has no config.yml; everything defaults, with the
    external repos pointing at the pinned extern/ submodules."""
    monkeypatch.setenv("STUDIO_CONFIG", str(tmp_path / "nope.yml"))
    cfg = config.load_config()
    assert cfg.kimodo_repo == config.DEFAULT_KIMODO_REPO
    assert cfg.spider == config.DEFAULT_SPIDER
    assert cfg.model == "Kimodo-G1-RP-v1"
    assert cfg.demo_port == 7860
    assert cfg.encoder_host == "euler"
    assert cfg.encoder_port == 9550


def test_empty_config_file_is_valid(tmp_path, monkeypatch):
    path = tmp_path / "config.yml"
    path.write_text("# overrides only\n")
    monkeypatch.setenv("STUDIO_CONFIG", str(path))
    assert config.load_config().kimodo_repo == config.DEFAULT_KIMODO_REPO


def test_solve_python_defaults_to_the_setup_location(write_config):
    write_config()
    assert config.load_config().solve_python == config.DEFAULT_SOLVE_PYTHON


def test_solve_python_can_be_overridden(write_config):
    write_config(solve_python="/venvs/solve/bin/python")
    assert config.load_config().solve_python.as_posix() == (
        "/venvs/solve/bin/python")


@pytest.mark.parametrize("model,short", [
    ("Kimodo-G1-RP-v1", "kimodo-g1-rp"),
    ("Kimodo-G1-RP-v2", "kimodo-g1-rp-v2"),   # only -v1 is stripped
])
def test_model_short_is_the_registry_key(write_config, model, short):
    write_config(model=model)
    assert config.load_config().model_short == short


def test_examples_dir_matches_the_demos_hardcoded_save_path(write_config):
    write_config(kimodo="/a/kimodo")
    assert config.load_config().examples_dir.as_posix() == (
        "/a/kimodo/kimodo/assets/demo/examples/kimodo-g1-rp")


# ------------------------------------------------------------- runs --

def test_task_root_mirrors_the_mppi_subtree(tmp_path):
    assert config.task_root(tmp_path / "run") == (
        tmp_path / "run/outputs" / config.TASK_SUBTREE)


def test_task_dirs_is_empty_when_never_reconstructed(tmp_path):
    assert config.task_dirs(tmp_path / "run") == []


def test_task_dirs_finds_and_filters_trials(tmp_path):
    root = config.task_root(tmp_path / "run")
    root.mkdir(parents=True)
    for name in ("clip_00", "clip_01", "other_00"):
        (root / name).mkdir()
    (root / "stray.json").write_text("{}")   # files are not trials

    assert [d.name for d in config.task_dirs(tmp_path / "run")] == [
        "clip_00", "clip_01", "other_00"]
    assert [d.name for d in config.task_dirs(tmp_path / "run", "clip")] == [
        "clip_00", "clip_01"]
