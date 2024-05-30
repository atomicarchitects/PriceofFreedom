from typing import Dict, Iterator
from absl import logging
import torch
import torch_geometric as pyG
import jraph
import numpy as np


def add_edges_transform(cutoff: float, add_self_edges: bool):
    """Returns a PyG transform that adds edges to the graph."""

    def add_edges(data):
        data.edge_index = pyG.nn.radius_graph(
            data.pos, r=cutoff, loop=add_self_edges
        ).numpy()
        return data
    
    return add_edges


def split_dataset(dataset: torch.utils.data.Dataset, splits: Dict[str, int], seed: int) -> Dict[str, torch.utils.data.Dataset]:
    if splits.get("test") is None:
        splits["test"] = len(dataset) - splits["train"] - splits["val"]
    if splits["test"] < 0:
        raise ValueError(f"Number of test graphs ({splits['test']}) cannot be negative.")
    datasets = torch.utils.data.random_split(
        dataset,
        [splits["train"], splits["val"], splits["test"]],
        generator=torch.Generator().manual_seed(seed),
    )
    return {
        "train": datasets[0],
        "val": datasets[1],
        "test": datasets[2],
    }


def _nearest_multiple_of_8(x: int) -> int:
    return int(np.ceil(x / 8) * 8)


def estimate_padding(graph: pyG.data.Data, cutoff: float, add_self_edges: bool, batch_size: int) -> Dict[str, int]:
    """Estimates the padding needed to batch the graphs."""
    n_node = int(graph.pos.shape[0])
    n_edge = pyG.nn.radius_graph(graph.pos, r=cutoff, loop=add_self_edges).shape[1]
    return dict(
        n_node=_nearest_multiple_of_8(n_node * batch_size),
        n_edge=_nearest_multiple_of_8(n_edge * batch_size),
        n_graph=batch_size,
    )