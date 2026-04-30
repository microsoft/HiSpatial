from pathlib import Path
from typing import List


def split_shards_by_node(
    shards: List[str],
    num_nodes: int = 1,
    node_rank: int = 0,
) -> List[Path]:
    """Split sorted shard list evenly across nodes, return the slice for *node_rank*."""
    if num_nodes <= 0:
        raise ValueError("num_nodes must be > 0")
    if not (0 <= node_rank < num_nodes):
        raise ValueError(f"node_rank must be in 0 .. {num_nodes - 1}")

    sorted_paths = sorted(map(Path, shards))

    n = len(sorted_paths)
    base, extra = divmod(n, num_nodes)

    start = node_rank * base + min(node_rank, extra)
    end = start + base + (1 if node_rank < extra else 0)

    return sorted_paths[start:end]
