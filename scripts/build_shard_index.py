"""Regenerate nanovla/data/assets/libero_shard_index.json from the Hub (footer reads only, ~15 s).

Run this when bumping ``REVISION`` in nanovla/data/libero.py:
    python scripts/build_shard_index.py --revision <sha>
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovla.data.libero import _SHIPPED_INDEX, REPO_ID, REVISION, build_shard_index_remote  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--out", type=Path, default=_SHIPPED_INDEX)
    args = ap.parse_args()

    idx = build_shard_index_remote(args.repo_id, args.revision)
    shards = idx["shards"]
    # Sanity: global index ranges must tile [0, total) with no gaps.
    for a, b in zip(shards, shards[1:]):
        assert a["index"][1] + 1 == b["index"][0], (a, b)
    total = sum(s["num_rows"] for s in shards)
    assert total == shards[-1]["index"][1] + 1, total
    split = sum(a["episode_index"][1] == b["episode_index"][0] for a, b in zip(shards, shards[1:]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(idx, indent=1) + "\n")
    print(f"{len(shards)} shards, {total} frames, {split} episodes split across shards -> {args.out}")


if __name__ == "__main__":
    main()
