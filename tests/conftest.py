"""Test fixtures, including a tiny synthetic LeRobot-v3 LIBERO-shaped dataset built on disk.

The synthetic dataset mirrors the real one's quirks:
  * meta/episodes ``data/file_index`` is deliberately WRONG (all 0), like the real Hub repo
  * one episode is split across two shards, which the loader must handle
  * every image's pixels encode the frame's global index, so tests can check that each
    sample got the right image
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

CAMS = ("observation.images.image", "observation.images.image2")
IMG = 16
TASKS = ["pick up the red block", "open the drawer", "put the bowl on the plate"]
# (episode_index, length, task_index); 7 episodes, 3 tasks
EPISODES = [(0, 5, 0), (1, 7, 1), (2, 3, 2), (3, 6, 0), (4, 4, 1), (5, 8, 2), (6, 5, 0)]
# Shard boundaries in global-index space; shard 1 starts mid-episode 3 (split episode).
SHARD_ROWS = [(0, 18), (18, 29), (29, 38)]


def pytest_addoption(parser):
    parser.addoption("--hub", action="store_true", help="run tests that download from the HF Hub")


def pytest_configure(config):
    config.addinivalue_line("markers", "hub: downloads real data from the HF Hub (opt in with --hub)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--hub") or os.environ.get("NANOVLA_HUB_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="needs --hub (downloads ~100 MB from the HF Hub)")
    for item in items:
        if "hub" in item.keywords:
            item.add_marker(skip)


def frame_value(global_index: int, cam: int) -> int:
    return (global_index * 7 + cam * 101) % 256


def _png(value: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((IMG, IMG, 3), value, np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def make_rows() -> dict[str, list]:
    rows: dict[str, list] = {k: [] for k in ("index", "episode_index", "frame_index", "task_index", "timestamp")}
    rows["observation.state"], rows["action"] = [], []
    g = 0
    for ep, length, task in EPISODES:
        for t in range(length):
            rows["index"].append(g)
            rows["episode_index"].append(ep)
            rows["frame_index"].append(t)
            rows["task_index"].append(task)
            rows["timestamp"].append(t / 10)
            # Distinct, decodable values: action[0] = global index, action[1] = episode, etc.
            rows["action"].append([float(g), float(ep), float(t), 0.1, -0.2, 0.3, 1.0 if t % 2 else -1.0])
            rows["observation.state"].append([float(g) / 10] + [float(ep)] * 7)
            g += 1
    return rows


def build_fake_libero(root: Path) -> dict:
    rows = make_rows()
    total = len(rows["index"])
    assert SHARD_ROWS[-1][1] == total
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)

    for shard, (a, b) in enumerate(SHARD_ROWS):
        cols = {k: v[a:b] for k, v in rows.items()}
        for c, cam in enumerate(CAMS):
            cols[cam] = [{"bytes": _png(frame_value(g, c)), "path": f"frame_{g:06d}.png"} for g in cols["index"]]
        pq.write_table(pa.table(cols), root / f"data/chunk-000/file-{shard:03d}.parquet")

    feats = {cam: {"dtype": "image", "shape": [IMG, IMG, 3]} for cam in CAMS}
    feats["observation.state"] = {"dtype": "float32", "shape": [8]}
    feats["action"] = {"dtype": "float32", "shape": [7]}
    info = {"codebase_version": "v3.0", "fps": 10.0, "total_episodes": len(EPISODES), "total_frames": total, "features": feats}
    (root / "meta/info.json").write_text(json.dumps(info))

    st, ac = np.asarray(rows["observation.state"]), np.asarray(rows["action"])
    stats = {
        k: {
            "mean": v.mean(0).tolist(),
            "std": v.std(0).tolist(),
            "q01": np.quantile(v, 0.01, axis=0).tolist(),
            "q99": np.quantile(v, 0.99, axis=0).tolist(),
        }
        for k, v in (("observation.state", st), ("action", ac))
    }
    (root / "meta/stats.json").write_text(json.dumps(stats))

    # Same layout as the real repo: task strings are the pandas index (__index_level_0__).
    pq.write_table(
        pa.table({"task_index": list(range(len(TASKS))), "__index_level_0__": TASKS}),
        root / "meta/tasks.parquet",
    )
    starts = np.cumsum([0] + [length for _, length, _ in EPISODES])
    pq.write_table(
        pa.table(
            {
                "episode_index": [e for e, _, _ in EPISODES],
                "data/chunk_index": [0] * len(EPISODES),
                "data/file_index": [0] * len(EPISODES),  # wrong on purpose, like the Hub repo
                "dataset_from_index": starts[:-1].tolist(),
                "dataset_to_index": starts[1:].tolist(),
                "tasks": [[TASKS[t]] for _, _, t in EPISODES],
                "length": [length for _, length, _ in EPISODES],
            }
        ),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    return rows


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    root = tmp_path / "libero"
    build_fake_libero(root)
    return root
