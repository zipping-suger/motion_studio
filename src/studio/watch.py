"""Watch the demo's Save Example dir and process new examples sequentially."""

import time
from pathlib import Path

from .config import Config


def _complete(example_dir: Path, settle: float) -> bool:
    # meta.json is written last by the demo's save handler; the mtime settle
    # guards against catching a save mid-write.
    if not (example_dir / "meta.json").exists():
        return False
    newest = max(f.stat().st_mtime for f in example_dir.iterdir())
    return time.time() - newest >= settle


def watch_examples(cfg: Config, process, poll: float = 2.0,
                   settle: float = 3.0) -> None:
    examples = cfg.examples_dir
    seen = ({p.name for p in examples.iterdir() if p.is_dir()}
            if examples.is_dir() else set())
    print(f"watching {examples}\n"
          f"({len(seen)} existing example(s) ignored; save a new example in "
          f"the demo to trigger the pipeline; Ctrl+C to stop)")
    while True:
        time.sleep(poll)
        if not examples.is_dir():
            continue
        for p in sorted(examples.iterdir()):
            if not p.is_dir() or p.name in seen or not _complete(p, settle):
                continue
            seen.add(p.name)
            print(f"\n### new example: {p.name}")
            try:
                process(p)
            except Exception as e:  # keep watching after a bad clip
                print(f"ERROR processing {p.name}: {e}")
            print("\nwatching for the next example ...")
