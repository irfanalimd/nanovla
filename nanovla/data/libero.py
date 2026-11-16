"""Map-style PyTorch dataset for the LeRobot-v3 LIBERO dataset on the HF Hub.

Dataset: https://huggingface.co/datasets/HuggingFaceVLA/libero
  - 1693 episodes, 273,465 frames, 40 tasks (the 4 LIBERO suites x 10 tasks), 10 fps, Franka Panda
  - two 256x256 RGB cameras stored *inline* in parquet as PNG bytes (no videos):
      observation.images.image   (third-person / agentview)
      observation.images.image2  (wrist)
  - observation.state: float32[8], action: float32[7] (6-DoF delta EEF + gripper in {-1, 1})
  - 377 parquet shards of ~100 MB each, ~35 GB in total

Two details of this dataset shape the loader:

1. ``meta/episodes`` claims the data lives in 69 files (``data/file_index`` 0..68), but the
   Hub actually holds 377 shards: the data was re-sharded without updating the metadata.
   So we do NOT trust ``data/file_index``. Instead we build a shard index from each parquet
   footer's min/max statistics of ``index`` / ``episode_index``, which can be read remotely
   with HTTP range requests without downloading the shards. A precomputed index for the
   pinned revision ships in ``assets/`` so this costs nothing in the common case.

2. Parquet gives no cheap random row access (each shard is a single ~100 MB row group).
   On first use each shard is downloaded and rewritten once as an uncompressed Arrow IPC
   file. Those files are memory-mapped, which gives O(1) zero-copy access to any row's
   PNG bytes. Low-dimensional columns are loaded into RAM (about 16 MB for the full dataset).

Samples are dicts:
    images         uint8   [n_cams, 3, H, W]   (normalize on the GPU, not here)
    state          float32 [8]
    actions        float32 [action_horizon, 7]  action chunk a_t .. a_{t+H-1}
    action_is_pad  bool    [action_horizon]     True where the chunk ran past the episode end
    task           str                          language instruction
    task_index, episode_index, frame_index, index   int64 scalars
"""

from __future__ import annotations

import io
import json
import os
import threading
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

from nanovla.data.normalize import Normalizer

REPO_ID = "HuggingFaceVLA/libero"
# Pinned for reproducibility and because the shipped shard index is only valid for this revision.
REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"

CAMERA_KEYS: tuple[str, ...] = ("observation.images.image", "observation.images.image2")
STATE_KEY = "observation.state"
ACTION_KEY = "action"
_LOWDIM_SCALARS = ("index", "episode_index", "frame_index", "task_index")

_ASSETS = Path(__file__).parent / "assets"
_SHIPPED_INDEX = _ASSETS / "libero_shard_index.json"
_META_FILES = ("meta/info.json", "meta/stats.json", "meta/tasks.parquet")
_EPISODES_META = "meta/episodes/chunk-000/file-000.parquet"
_ARROW_DIR = "arrow"


# --------------------------------------------------------------------------------------
# Shard index: which global rows / episodes live in which parquet file
# --------------------------------------------------------------------------------------


def _shard_entry(path: str, fileobj: Any) -> dict:
    """Read min/max of ``index`` and ``episode_index`` for one parquet file from its footer."""
    pf = pq.ParquetFile(fileobj)
    md = pf.metadata
    names = [md.schema.column(j).path for j in range(md.num_columns)]
    ranges: dict[str, list[int]] = {}
    for col in ("index", "episode_index"):
        lo, hi = None, None
        j = names.index(col)
        for rg in range(md.num_row_groups):
            s = md.row_group(rg).column(j).statistics
            if s is None or not s.has_min_max:
                lo = None
                break
            lo = s.min if lo is None else min(lo, s.min)
            hi = s.max if hi is None else max(hi, s.max)
        if lo is None:  # no footer stats: fall back to reading just this column
            arr = pf.read(columns=[col]).column(0)
            lo, hi = pc.min(arr).as_py(), pc.max(arr).as_py()
        ranges[col] = [int(lo), int(hi)]
    return {"path": path, "num_rows": md.num_rows, **ranges}


def _finalize_index(shards: list[dict], repo_id: str | None, revision: str | None) -> dict:
    shards = sorted(shards, key=lambda s: s["index"][0])
    for a, b in zip(shards, shards[1:]):
        if a["index"][1] >= b["index"][0]:
            raise ValueError(f"Overlapping global index ranges: {a['path']} and {b['path']}")
    return {"repo_id": repo_id, "revision": revision, "shards": shards}


def build_shard_index_remote(
    repo_id: str = REPO_ID, revision: str = REVISION, max_workers: int = 16
) -> dict:
    """Build the shard index by reading parquet footers from the Hub (range requests only)."""
    from huggingface_hub import HfApi, HfFileSystem

    files = [
        f
        for f in HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)
        if f.startswith("data/") and f.endswith(".parquet")
    ]
    fs = HfFileSystem()

    def one(path: str) -> dict:
        with fs.open(f"datasets/{repo_id}@{revision}/{path}", "rb", block_size=1 << 16) as f:
            return _shard_entry(path, f)

    with ThreadPoolExecutor(max_workers) as ex:
        shards = list(ex.map(one, files))
    return _finalize_index(shards, repo_id, revision)


def build_shard_index_local(root: str | Path) -> dict:
    """Build the shard index from raw parquet files already present under ``root/data``."""
    root = Path(root)
    shards = []
    for p in sorted((root / "data").glob("**/*.parquet")):
        with open(p, "rb") as f:
            shards.append(_shard_entry(p.relative_to(root).as_posix(), f))
    if not shards:
        raise FileNotFoundError(f"No parquet shards under {root / 'data'} to index")
    return _finalize_index(shards, None, None)


def load_shard_index(
    root: Path, repo_id: str | None, revision: str | None, download: bool
) -> dict:
    """Resolve the shard index and cache it at ``root/shard_index.json``.

    Online: cached copy if it matches (repo_id, revision), else the shipped asset if it
    matches, else read footers from the Hub. Offline: cached copy, else index the raw
    parquet files present under ``root``.
    """
    cached = root / "shard_index.json"
    if cached.exists():
        idx = json.loads(cached.read_text())
        if not download or (idx["repo_id"] == repo_id and idx["revision"] == revision):
            return idx
    if not download:
        idx = build_shard_index_local(root)
    else:
        idx = None
        if _SHIPPED_INDEX.exists():
            shipped = json.loads(_SHIPPED_INDEX.read_text())
            if shipped["repo_id"] == repo_id and shipped["revision"] == revision:
                idx = shipped
        if idx is None:
            idx = build_shard_index_remote(repo_id, revision)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(cached, json.dumps(idx))
    return idx


# --------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------


@dataclass
class LiberoMeta:
    info: dict
    stats: dict
    tasks: list[str]  # task_index -> instruction
    episode_index: np.ndarray  # [E]
    episode_length: np.ndarray  # [E]
    episode_task: np.ndarray  # [E] task_index of each episode

    @classmethod
    def load(cls, root: Path) -> LiberoMeta:
        info = json.loads((root / "meta/info.json").read_text())
        stats = json.loads((root / "meta/stats.json").read_text())
        t = pq.read_table(root / "meta/tasks.parquet").to_pydict()
        # LeRobot stores the task string as the (pandas) index of tasks.parquet.
        name_col = "task" if "task" in t else next(k for k in t if k != "task_index")
        tasks: list[str | None] = [None] * (max(t["task_index"]) + 1)
        for i, name in zip(t["task_index"], t[name_col]):
            tasks[i] = name
        task_to_idx = {name: i for i, name in enumerate(tasks)}

        e = pq.read_table(root / _EPISODES_META, columns=["episode_index", "length", "tasks"])
        e = e.to_pydict()
        order = np.argsort(e["episode_index"])
        episode_task = np.array([task_to_idx[e["tasks"][i][0]] for i in order], dtype=np.int64)
        return cls(
            info=info,
            stats=stats,
            tasks=tasks,  # type: ignore[arg-type]
            episode_index=np.asarray(e["episode_index"], dtype=np.int64)[order],
            episode_length=np.asarray(e["length"], dtype=np.int64)[order],
            episode_task=episode_task,
        )

    def episodes_for_tasks(self, task_indices: Iterable[int]) -> np.ndarray:
        return self.episode_index[np.isin(self.episode_task, list(task_indices))]


def train_val_episodes(
    meta: LiberoMeta,
    val_fraction: float = 0.05,
    seed: int = 0,
    episodes: Iterable[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Split episodes into train/val, stratified by task so every task appears in val."""
    rng = np.random.default_rng(seed)
    pool = meta.episode_index if episodes is None else np.asarray(sorted(set(episodes)))
    task_of = dict(zip(meta.episode_index.tolist(), meta.episode_task.tolist()))
    train, val = [], []
    for task in sorted({task_of[int(e)] for e in pool}):
        eps = np.array([e for e in pool if task_of[int(e)] == task])
        rng.shuffle(eps)
        n_val = int(round(len(eps) * val_fraction))
        if val_fraction > 0 and len(eps) > 1:
            n_val = max(1, n_val)
        val += eps[:n_val].tolist()
        train += eps[n_val:].tolist()
    return sorted(train), sorted(val)


# --------------------------------------------------------------------------------------
# Download + parquet -> Arrow IPC conversion
# --------------------------------------------------------------------------------------


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def _arrow_path(root: Path, shard_path: str) -> Path:
    return root / _ARROW_DIR / Path(shard_path).with_suffix(".arrow")


def _convert_to_arrow(src: Path, dst: Path) -> None:
    table = pq.read_table(src).combine_chunks()
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(f".arrow.tmp{os.getpid()}.{threading.get_ident()}")
    with pa.OSFile(str(tmp), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    os.replace(tmp, dst)  # atomic: an interrupted run never leaves a half-written cache file


def _ensure_meta(root: Path, repo_id: str, revision: str, download: bool) -> None:
    needed = [*_META_FILES, _EPISODES_META]
    missing = [f for f in needed if not (root / f).exists()]
    if not missing:
        return
    if not download:
        raise FileNotFoundError(f"Missing metadata under {root}: {missing} (download=False)")
    from huggingface_hub import hf_hub_download

    for f in missing:
        hf_hub_download(repo_id, f, repo_type="dataset", revision=revision, local_dir=root)


def prepare_shards(
    root: Path,
    shard_paths: Sequence[str],
    repo_id: str,
    revision: str,
    download: bool = True,
    keep_raw: bool = True,
    num_workers: int = 8,
    progress: bool = True,
) -> list[Path]:
    """Make sure every shard exists as an Arrow IPC file under ``root/arrow``; return the paths."""

    def one(shard_path: str) -> Path:
        dst = _arrow_path(root, shard_path)
        if dst.exists():
            return dst
        src = root / shard_path
        if not src.exists():
            if not download:
                raise FileNotFoundError(f"{src} not found and download=False")
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id, shard_path, repo_type="dataset", revision=revision, local_dir=root
            )
        _convert_to_arrow(src, dst)
        if not keep_raw:
            src.unlink()
        return dst

    todo = [p for p in shard_paths if not _arrow_path(root, p).exists()]
    if not todo:
        return [_arrow_path(root, p) for p in shard_paths]
    with ThreadPoolExecutor(max(1, min(num_workers, len(todo)))) as ex:
        it = ex.map(one, todo)
        if progress:
            from tqdm.auto import tqdm

            it = tqdm(it, total=len(todo), desc="LIBERO shards (download+convert)")
        list(it)
    return [_arrow_path(root, p) for p in shard_paths]


def _read_lowdim(path: Path) -> dict[str, np.ndarray]:
    with pa.memory_map(str(path), "r") as mm:
        table = pa.ipc.open_file(mm).read_all()
        out = {k: table.column(k).to_numpy() for k in _LOWDIM_SCALARS}
        for k in (STATE_KEY, ACTION_KEY):
            col = table.column(k).combine_chunks()
            lengths = pc.list_value_length(col).to_numpy(zero_copy_only=False)
            if len(lengths) and not (lengths == lengths[0]).all():
                raise ValueError(f"{path}: ragged {k} column")
            d = int(lengths[0]) if len(lengths) else 0
            out[k] = col.flatten().to_numpy().astype(np.float32).reshape(-1, d)
    return out


def _decode(png: bytes, size: int | None) -> torch.Tensor:
    """PNG/JPEG bytes -> uint8 [3, H, W], optionally resized (bilinear, antialiased)."""
    img = Image.open(io.BytesIO(png)).convert("RGB")
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1)


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------


class LiberoDataset(Dataset):
    """Frame-level LIBERO dataset that returns an action chunk for every frame.

    Args:
        root: local directory for metadata, raw parquet and the Arrow cache.
        episodes: episode indices to include (default: all 1693). Only the shards that
            contain these episodes are downloaded.
        task_indices: keep only episodes of these tasks (intersected with ``episodes``).
        action_horizon: length H of the returned action chunk. Steps past the episode end
            repeat the last action and are flagged in ``action_is_pad``.
        cameras: which camera keys to return, stacked in this order.
        image_size: if set, resize images to (image_size, image_size) with antialiasing.
        normalize: "quantile" (q01/q99 -> [-1, 1]), "mean_std", or None. Applies to
            state and actions. Uses whole-dataset stats from meta/stats.json.
        download: fetch missing files from the Hub. If False, ``root`` must already hold
            everything (raw parquet or Arrow cache).
        keep_raw: keep raw parquet after converting to Arrow. Set to False to halve disk use.
    """

    def __init__(
        self,
        root: str | os.PathLike = "data/libero",
        episodes: Iterable[int] | None = None,
        task_indices: Iterable[int] | None = None,
        action_horizon: int = 16,
        cameras: Sequence[str] = CAMERA_KEYS,
        image_size: int | None = None,
        normalize: str | None = "quantile",
        repo_id: str = REPO_ID,
        revision: str = REVISION,
        download: bool = True,
        keep_raw: bool = True,
        num_download_workers: int = 8,
        progress: bool = True,
    ):
        if action_horizon < 1:
            raise ValueError("action_horizon must be >= 1")
        self.root = Path(root)
        self.action_horizon = action_horizon
        self.cameras = tuple(cameras)
        self.image_size = image_size

        _ensure_meta(self.root, repo_id, revision, download)
        self.meta = LiberoMeta.load(self.root)
        self.fps = float(self.meta.info["fps"])
        unknown = set(self.cameras) - set(self.meta.info["features"])
        if unknown:
            raise KeyError(f"Unknown camera keys {unknown}")
        self.normalizer = Normalizer.from_stats(self.meta.stats, normalize) if normalize else None

        # ---- which episodes, which shards
        selected = set(self.meta.episode_index.tolist())
        if episodes is not None:
            requested = {int(e) for e in episodes}
            missing = requested - selected
            if missing:
                raise ValueError(f"Episodes not in dataset: {sorted(missing)[:10]}")
            selected = requested
        if task_indices is not None:
            selected &= set(self.meta.episodes_for_tasks(task_indices).tolist())
        if not selected:
            raise ValueError("No episodes selected")
        self.episodes = np.array(sorted(selected), dtype=np.int64)

        index = load_shard_index(self.root, repo_id, revision, download)
        shards = [  # every shard that holds at least one selected episode (even partially)
            s
            for s in index["shards"]
            if np.any((self.episodes >= s["episode_index"][0]) & (self.episodes <= s["episode_index"][1]))
        ]
        self._shard_files = [
            str(p)
            for p in prepare_shards(
                self.root,
                [s["path"] for s in shards],
                repo_id,
                revision,
                download=download,
                keep_raw=keep_raw,
                num_workers=num_download_workers,
                progress=progress,
            )
        ]

        # ---- low-dim columns for the selected episodes, in (episode, frame) order
        parts: dict[str, list[np.ndarray]] = {}
        for shard_id, path in enumerate(self._shard_files):
            cols = _read_lowdim(Path(path))
            keep = np.isin(cols["episode_index"], self.episodes)
            cols["shard_id"] = np.full(len(keep), shard_id, dtype=np.int32)
            cols["row"] = np.arange(len(keep), dtype=np.int64)
            for k, v in cols.items():
                parts.setdefault(k, []).append(v[keep])
        cat = {k: np.concatenate(v) for k, v in parts.items()}
        order = np.lexsort((cat["frame_index"], cat["episode_index"]))
        cat = {k: v[order] for k, v in cat.items()}

        self._state = cat[STATE_KEY]
        self._actions = cat[ACTION_KEY]
        self._episode_index = cat["episode_index"]
        self._frame_index = cat["frame_index"]
        self._task_index = cat["task_index"]
        self._global_index = cat["index"]
        self._shard_id = cat["shard_id"]
        self._row = cat["row"]
        self._check_and_index_episodes()

        self._tables: dict[int, pa.Table] | None = None  # opened lazily, per worker process

    def _check_and_index_episodes(self) -> None:
        ep = self._episode_index
        starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
        ends = np.r_[starts[1:], len(ep)]
        found = ep[starts]
        if not np.array_equal(found, self.episodes):
            missing = sorted(set(self.episodes.tolist()) - set(found.tolist()))
            raise RuntimeError(f"Episodes missing from downloaded shards: {missing[:10]}")
        meta_len = dict(zip(self.meta.episode_index.tolist(), self.meta.episode_length.tolist()))
        meta_task = dict(zip(self.meta.episode_index.tolist(), self.meta.episode_task.tolist()))
        for e, s, t in zip(found, starts, ends):
            if t - s != meta_len[int(e)]:
                raise RuntimeError(f"Episode {e}: {t - s} frames on disk, metadata says {meta_len[int(e)]}")
            if not np.array_equal(self._frame_index[s:t], np.arange(t - s)):
                raise RuntimeError(f"Episode {e}: frame_index is not 0..{t - s - 1}")
            if not (self._task_index[s:t] == meta_task[int(e)]).all():
                raise RuntimeError(f"Episode {e}: frame task_index disagrees with meta/episodes")
        lengths = ends - starts
        self._ep_start = np.repeat(starts, lengths)
        self._ep_end = np.repeat(ends, lengths)

    # ---- pickling: worker processes (spawn/forkserver) must reopen the memory maps
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_tables"] = None
        return state

    def _table(self, shard_id: int) -> pa.Table:
        if self._tables is None:
            self._tables = {}
        t = self._tables.get(shard_id)
        if t is None:
            mm = pa.memory_map(self._shard_files[shard_id], "r")
            t = pa.ipc.open_file(mm).read_all()  # zero-copy view over the mmap
            self._tables[shard_id] = t
        return t

    def __len__(self) -> int:
        return len(self._state)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def state_dim(self) -> int:
        return self._state.shape[1]

    @property
    def action_dim(self) -> int:
        return self._actions.shape[1]

    def load_images(self, i: int) -> torch.Tensor:
        table = self._table(int(self._shard_id[i]))
        row = int(self._row[i])
        return torch.stack(
            [_decode(table.column(cam)[row]["bytes"].as_py(), self.image_size) for cam in self.cameras]
        )

    def __getitem__(self, i: int) -> dict[str, Any]:
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        end = self._ep_end[i]
        steps = i + np.arange(self.action_horizon)
        is_pad = steps >= end
        actions = self._actions[np.minimum(steps, end - 1)]
        state = self._state[i]
        if self.normalizer is not None:
            actions = self.normalizer.normalize(actions, ACTION_KEY)
            state = self.normalizer.normalize(state, STATE_KEY)
        task_index = int(self._task_index[i])
        return {
            "images": self.load_images(i),
            "state": torch.from_numpy(np.ascontiguousarray(state, dtype=np.float32)),
            "actions": torch.from_numpy(np.ascontiguousarray(actions, dtype=np.float32)),
            "action_is_pad": torch.from_numpy(is_pad),
            "task": self.meta.tasks[task_index],
            "task_index": torch.tensor(task_index),
            "episode_index": torch.tensor(int(self._episode_index[i])),
            "frame_index": torch.tensor(int(self._frame_index[i])),
            "index": torch.tensor(int(self._global_index[i])),
        }

    def __repr__(self) -> str:
        return (
            f"LiberoDataset(root={str(self.root)!r}, episodes={self.num_episodes}, frames={len(self)}, "
            f"shards={len(self._shard_files)}, cameras={self.cameras}, H={self.action_horizon}, "
            f"normalize={self.normalizer.mode if self.normalizer else None})"
        )
