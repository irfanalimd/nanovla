from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from conftest import CAMS, EPISODES, IMG, TASKS, frame_value
from nanovla.data import LiberoDataset, LiberoMeta, Normalizer, train_val_episodes
from nanovla.data.libero import _SHIPPED_INDEX, REPO_ID, REVISION

H = 4


def make(root, **kw) -> LiberoDataset:
    kw.setdefault("action_horizon", H)
    kw.setdefault("normalize", None)
    return LiberoDataset(root, download=False, progress=False, **kw)


# ---------------------------------------------------------------- synthetic (offline)


def test_length_and_shapes(fake_root):
    ds = make(fake_root)
    assert len(ds) == sum(length for _, length, _ in EPISODES)
    assert ds.num_episodes == len(EPISODES)
    s = ds[0]
    assert s["images"].shape == (2, 3, IMG, IMG) and s["images"].dtype == torch.uint8
    assert s["state"].shape == (8,) and s["state"].dtype == torch.float32
    assert s["actions"].shape == (H, 7) and s["actions"].dtype == torch.float32
    assert s["action_is_pad"].shape == (H,) and s["action_is_pad"].dtype == torch.bool
    assert isinstance(s["task"], str)


def test_every_frame_matches_its_source_row(fake_root):
    """Images, state, task and action chunk must all come from the right global row,
    including for episodes split across shards (episodes 3 and 5)."""
    ds = make(fake_root)
    ep_bounds, g = {}, 0
    for ep, length, task in EPISODES:
        ep_bounds[ep] = (g, g + length, task)
        g += length
    for i in range(len(ds)):
        s = ds[i]
        gi, ep = int(s["index"]), int(s["episode_index"])
        lo, hi, task = ep_bounds[ep]
        assert lo <= gi < hi and int(s["frame_index"]) == gi - lo
        assert s["task"] == TASKS[task] and int(s["task_index"]) == task
        for c in range(len(CAMS)):
            assert (s["images"][c] == frame_value(gi, c)).all(), (i, c)
        assert s["state"][0].item() == pytest.approx(gi / 10)
        # action[0] holds the global index; the chunk must stay inside the episode.
        want = np.minimum(gi + np.arange(H), hi - 1)
        np.testing.assert_array_equal(s["actions"][:, 0].numpy(), want)
        np.testing.assert_array_equal(s["actions"][:, 1].numpy(), ep)
        np.testing.assert_array_equal(s["action_is_pad"].numpy(), gi + np.arange(H) >= hi)


def test_last_frame_of_episode_is_padded(fake_root):
    ds = make(fake_root, episodes=[2])  # 3-frame episode
    last = ds[len(ds) - 1]
    assert last["action_is_pad"].tolist() == [False, True, True, True]
    assert (last["actions"] == last["actions"][0]).all()  # repeats the final action


def test_episode_and_task_filtering(fake_root):
    ds = make(fake_root, episodes=[3, 5])  # both split across shards
    assert set(ds.episodes.tolist()) == {3, 5} and len(ds) == 6 + 8
    assert {int(ds[i]["episode_index"]) for i in range(len(ds))} == {3, 5}

    ds = make(fake_root, task_indices=[0])
    want = [e for e, _, t in EPISODES if t == 0]
    assert ds.episodes.tolist() == want
    assert all(ds[i]["task"] == TASKS[0] for i in range(len(ds)))

    with pytest.raises(ValueError):
        make(fake_root, episodes=[999])


def test_only_needed_shards_are_converted(fake_root):
    make(fake_root, episodes=[0])  # lives entirely in shard 0
    arrows = sorted(p.name for p in (fake_root / "arrow").glob("**/*.arrow"))
    assert arrows == ["file-000.arrow"]


def test_shard_index_ignores_wrong_episode_metadata(fake_root):
    make(fake_root)
    idx = json.loads((fake_root / "shard_index.json").read_text())
    assert [s["episode_index"] for s in idx["shards"]] == [[0, 3], [3, 5], [5, 6]]


def test_keep_raw_false_then_reload_offline(fake_root):
    ds = make(fake_root, keep_raw=False)
    assert not list((fake_root / "data").glob("**/*.parquet"))
    ds2 = make(fake_root)  # served entirely from the Arrow cache + cached shard index
    assert len(ds2) == len(ds)
    assert torch.equal(ds2[10]["images"], ds[10]["images"])


def test_cameras_and_resize(fake_root):
    ds = make(fake_root, cameras=[CAMS[1]], image_size=8)
    s = ds[3]
    assert s["images"].shape == (1, 3, 8, 8)
    assert (s["images"] == frame_value(int(s["index"]), 1)).all()  # constant image survives resize
    with pytest.raises(KeyError):
        make(fake_root, cameras=["observation.images.nope"])


def test_normalization_roundtrip(fake_root):
    raw = make(fake_root)
    for mode in ("quantile", "mean_std"):
        ds = make(fake_root, normalize=mode)
        s, r = ds[7], raw[7]
        assert not torch.allclose(s["actions"], r["actions"])
        torch.testing.assert_close(ds.normalizer.unnormalize(s["actions"], "action"), r["actions"])
        torch.testing.assert_close(ds.normalizer.unnormalize(s["state"], "observation.state"), r["state"])
        n2 = Normalizer.from_state_dict(json.loads(json.dumps(ds.normalizer.state_dict())))
        np.testing.assert_allclose(n2.unnormalize(s["actions"].numpy(), "action"), r["actions"].numpy(), rtol=1e-5)


@pytest.mark.parametrize("ctx", ["fork", "spawn"])
def test_dataloader_multiprocess(fake_root, ctx):
    ds = make(fake_root)
    dl = DataLoader(ds, batch_size=5, shuffle=True, num_workers=2, multiprocessing_context=ctx)
    seen = []
    for b in dl:
        assert b["images"].shape[1:] == (2, 3, IMG, IMG)
        assert b["actions"].shape[1:] == (H, 7)
        assert isinstance(b["task"], list) and len(b["task"]) == len(b["index"])
        seen += b["index"].tolist()
    assert sorted(seen) == list(range(len(ds)))


def test_train_val_split_is_stratified(fake_root):
    meta = LiberoMeta.load(fake_root)
    train, val = train_val_episodes(meta, val_fraction=0.3, seed=0)
    assert not set(train) & set(val)
    assert sorted(train + val) == [e for e, _, _ in EPISODES]
    task_of = dict(zip(meta.episode_index.tolist(), meta.episode_task.tolist()))
    assert {task_of[e] for e in val} == {0, 1, 2}  # every task represented in val
    assert train_val_episodes(meta, 0.3, seed=0) == (train, val)  # deterministic


# ---------------------------------------------------------------- real Hub data (opt in)


def test_shipped_index_is_consistent():
    idx = json.loads(_SHIPPED_INDEX.read_text())
    assert idx["repo_id"] == REPO_ID and idx["revision"] == REVISION
    shards = idx["shards"]
    assert shards[0]["index"][0] == 0
    for a, b in zip(shards, shards[1:]):
        assert a["index"][1] + 1 == b["index"][0]
    assert sum(s["num_rows"] for s in shards) == shards[-1]["index"][1] + 1 == 273465


@pytest.mark.hub
def test_real_libero_first_shard(tmp_path_factory):
    root = tmp_path_factory.getbasetemp().parent / "nanovla-hub-cache"  # reused across runs
    ds = LiberoDataset(root, episodes=[0, 1, 2], action_horizon=10, progress=False)
    assert len(ds) == 843 and ds.state_dim == 8 and ds.action_dim == 7
    s = ds[0]
    assert s["images"].shape == (2, 3, 256, 256)
    assert s["task"].startswith("put the white mug on the left plate")
    assert torch.isfinite(s["actions"]).all() and torch.isfinite(s["state"]).all()
    b = next(iter(DataLoader(ds, batch_size=8, shuffle=True, num_workers=2)))
    assert b["images"].shape == (8, 2, 3, 256, 256)
