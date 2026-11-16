"""CPU smoke test for the LIBERO loader on real data from HuggingFaceVLA/libero.

By default this downloads only the first shard (~100 MB, episodes 0-2), checks samples,
iterates a multi-worker DataLoader, and reports throughput. Runs on CPU only.

    python scripts/smoke_test_data.py
    python scripts/smoke_test_data.py --episodes 0-49 --num-workers 4 --steps 50
    python scripts/smoke_test_data.py --preview preview.png    # save a grid of frames
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from nanovla.data import LiberoDataset  # noqa: E402


def parse_episodes(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        else:
            out.append(int(part))
    return out


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("NANOVLA_DATA", "data/libero"))
    ap.add_argument("--episodes", default="0-2", help='e.g. "0-2" or "0,5,10-20"; "all" for everything')
    ap.add_argument("--action-horizon", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20, help="DataLoader batches to time")
    ap.add_argument("--preview", default=None, help="save a PNG grid of a batch here")
    args = ap.parse_args()

    torch.set_num_threads(1)
    episodes = None if args.episodes == "all" else parse_episodes(args.episodes)

    print("== build dataset")
    t0 = time.perf_counter()
    ds = LiberoDataset(
        args.root, episodes=episodes, action_horizon=args.action_horizon, image_size=args.image_size
    )
    print(f"  {ds}\n  built in {time.perf_counter() - t0:.1f}s")

    print("== single sample")
    s = ds[0]
    for k, v in s.items():
        print(f"  {k:14s} {tuple(v.shape) if torch.is_tensor(v) else repr(v)[:70]}"
              f"{'  ' + str(v.dtype) if torch.is_tensor(v) else ''}")
    side = args.image_size or 256
    check(s["images"].shape == (len(ds.cameras), 3, side, side) and s["images"].dtype == torch.uint8, "image shape/dtype")
    check(s["state"].shape == (ds.state_dim,) and s["actions"].shape == (args.action_horizon, ds.action_dim), "state/action shapes")
    check(bool(torch.isfinite(s["actions"]).all() and torch.isfinite(s["state"]).all()), "finite state/actions")
    check(len(s["task"]) > 0, f"task string: {s['task']!r}")

    # Action chunks at the end of an episode must be padded, and nowhere else.
    ep_len = int((ds._episode_index == ds.episodes[0]).sum())
    last = ds[ep_len - 1]
    check(bool(last["action_is_pad"][1:].all() and not last["action_is_pad"][0]), "last frame of episode: chunk padded")
    check(not bool(ds[0]["action_is_pad"].any()) or ep_len < args.action_horizon, "first frame: no padding")

    # Quantile normalization maps the dataset-wide q01..q99 to [-1, 1]. A small subset of
    # episodes won't fill that range exactly (and gripper +1 maps to ~1.08 because q99 is
    # 0.92), so only check for gross errors such as wrong stats or misaligned columns.
    acts = torch.stack([ds[i]["actions"][0] for i in np.linspace(0, len(ds) - 1, 200).astype(int)])
    states = torch.stack([ds[i]["state"] for i in np.linspace(0, len(ds) - 1, 200).astype(int)])
    frac = (acts.abs() <= 1.1).float().mean().item()
    check(acts.abs().max() < 5 and states.abs().max() < 10, f"normalized ranges sane ({frac:.0%} of actions in [-1.1, 1.1])")
    raw = ds.normalizer.unnormalize(acts, "action")
    check(bool(((raw[:, -1].abs() - 1).abs() < 1e-3).all()), "gripper action is in {-1, +1} after unnormalize")

    print(f"== DataLoader: batch_size={args.batch_size} workers={args.num_workers}")
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0, drop_last=True,
    )
    it = iter(dl)
    batch = next(it)  # warm-up (worker start-up, page cache)
    t0, n = time.perf_counter(), 0
    for _ in range(args.steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl)
            b = next(it)
        n += len(b["index"])
    dt = time.perf_counter() - t0
    check(batch["images"].shape[0] == args.batch_size, f"batch images {tuple(batch['images'].shape)}")
    check(isinstance(batch["task"], list) and len(batch["task"]) == args.batch_size, "batch tasks collate to list[str]")
    print(f"  throughput: {n / dt:.0f} samples/s ({dt / args.steps * 1e3:.0f} ms/batch)")

    if args.preview:
        from PIL import Image

        imgs = batch["images"][:8]  # [B, cams, 3, H, W] -> rows = cameras, cols = samples
        grid = imgs.permute(1, 3, 0, 4, 2).reshape(imgs.shape[1] * imgs.shape[3], -1, 3)
        Image.fromarray(grid.numpy()).save(args.preview)
        print(f"  wrote {args.preview}")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
