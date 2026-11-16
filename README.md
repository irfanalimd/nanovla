# nanovla

A small vision-language-action (VLA) model trained on [LIBERO](https://libero-project.github.io/), using the LeRobot-format dataset [`HuggingFaceVLA/libero`](https://huggingface.co/datasets/HuggingFaceVLA/libero).

```
nanovla/
  data/
    libero.py            LiberoDataset: Hub download -> Arrow cache -> (images, state, action chunk, task)
    normalize.py         invertible quantile / mean-std normalization from meta/stats.json
    assets/libero_shard_index.json   which episodes live in which parquet shard (pinned revision)
  models/                VLA model (todo)
scripts/
  smoke_test_data.py     CPU smoke test on real data (downloads ~100 MB)
  build_shard_index.py   regenerate the shard index when bumping the dataset revision
notebooks/colab.ipynb    clone + install + test + GPU loader benchmark on Colab
tests/                   offline tests on a synthetic dataset, plus opt-in real-data test (--hub)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # CPU-only torch: add --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q                       # offline, ~5 s
python -m pytest -q --hub                 # + downloads the first real shard
python scripts/smoke_test_data.py         # real-data CPU smoke test
```

On Colab, open `notebooks/colab.ipynb`, add a `GITHUB_TOKEN` Colab secret (the repo is private), and run all cells.

## Using the dataset

```python
from torch.utils.data import DataLoader
from nanovla.data import LiberoDataset, train_val_episodes

ds = LiberoDataset("data/libero", episodes=range(100), action_horizon=16)
sample = ds[0]
# images uint8 [2, 3, 256, 256] (agentview, wrist) | state [8] | actions [16, 7]
# action_is_pad [16] | task "put the white mug on ..." | task_index, episode_index, ...

train_eps, val_eps = train_val_episodes(ds.meta, val_fraction=0.05)  # stratified by task
dl = DataLoader(LiberoDataset("data/libero", episodes=train_eps), batch_size=64, shuffle=True, num_workers=4)

# At eval time, map predicted actions back to robot units:
ds.normalizer.unnormalize(pred_actions, "action")
ds.normalizer.state_dict()   # save this with checkpoints
```

Options: `task_indices=[...]` filters by task, `cameras=[...]` picks a subset of cameras, `image_size=` resizes images, `normalize="quantile" | "mean_std" | None`, `keep_raw=False` deletes parquet after conversion, and `download=False` runs fully offline.

## Data notes

- **About 35 GB, 377 shards, 273k frames, 1693 episodes, 40 tasks, 10 fps.** Images are PNG bytes stored inline in parquet; there are no videos.
- **`meta/episodes` is stale.** Its `data/file_index` points at 69 files, but the Hub repo has 377 shards (the data was re-sharded). The loader ignores that field and uses a shard index built from parquet footer statistics. A precomputed index ships in `nanovla/data/assets/` for the pinned `REVISION`.
- **Only the shards you need are downloaded.** On first use each one is converted to an uncompressed Arrow IPC file under `<root>/arrow/` and memory-mapped, which gives O(1) random access. Conversions are atomic, so an interrupted run never leaves a corrupt cache.
- **The loader validates on load:** episode lengths against metadata, `frame_index` contiguity, and frame-level vs episode-level task labels.
- **Normalization:** quantile mode maps the dataset-wide q01..q99 to [-1, 1] without clipping. The gripper's q99 is 0.92, so +1 normalizes to about 1.08.
- **Throughput:** PNG decoding costs about 2.5 ms per image, which is about 170 samples/s per worker with both cameras. If the loader becomes the bottleneck on Colab's 2 vCPUs, the next step is a pre-decoded, lower-resolution uint8 cache.
