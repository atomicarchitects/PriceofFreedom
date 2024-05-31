"""Exposes the MD17 datasets in a convenient format."""

from typing import Dict, Generator

from absl import logging
import torch_geometric as pyG
import jraph
import tqdm
import numpy as np
import torch

from src import datatypes


def pyG_dataset_to_graphs_iterator(
    pyG_dataset: pyG.data.Dataset, cutoff: float, add_self_edges: bool
) -> Generator[datatypes.Graphs, None, None]:
    """Converts a pyG dataset to a dataset of graphs."""

    def to_graph(data):
        edge_index = pyG.nn.radius_graph(
            data.pos, r=cutoff, loop=add_self_edges
        ).numpy()
        data = data.to_dict()
        data = {key: val.numpy() for key, val in data.items()}
        data = datatypes.Graphs(
            nodes=datatypes.NodesInfo(
                positions=data["pos"], numbers=data["z"], forces=data["force"]
            ),
            edges=np.ones_like(edge_index[0], dtype=np.float32),
            globals=datatypes.GlobalsInfo(energies=data["energy"]),
            n_node=np.asarray([data["pos"].shape[0]]),
            n_edge=np.asarray([edge_index.shape[1]]),
            senders=edge_index[0, :],
            receivers=edge_index[1, :],
        )
        return data

    for data in tqdm.tqdm(pyG_dataset, desc="Converting to jraph.GraphsTuples..."):
        yield to_graph(data)


def get_md17_datasets(
    molecule: str,
    batch_size: int,
    cutoff: float,
    add_self_edges: bool,
    splits: Dict[str, int],
    datadir: str,
    seed: int,
) -> Dict[str, tf.data.Dataset]:
    """Returns a TensorFlow dataset of graphs from the MD17 dataset."""

    def nearest_multiple_of_8(x: int) -> int:
        return int(np.ceil(x / 8) * 8)

    def estimate_padding(graph: pyG.data.Data) -> Dict[str, int]:
        """Estimates the padding needed to batch the graphs."""
        n_node = int(graph.pos.shape[0])
        n_edge = pyG.nn.radius_graph(graph.pos, r=cutoff, loop=add_self_edges).shape[1]
        return dict(
            n_node=nearest_multiple_of_8(n_node * batch_size),
            n_edge=nearest_multiple_of_8(n_edge * batch_size),
            n_graph=batch_size,
        )

    # Load the dataset.
    pyG_dataset = pyG.datasets.MD17(root=datadir, name=molecule)

    if splits["test"] is None:
        splits["test"] = len(pyG_dataset) - splits["train"] - splits["val"]

    if splits["test"] < 0:
        raise ValueError(
            f"Number of test graphs ({splits['test']}) cannot be negative."
        )

    # Split the dataset.
    pyG_seed = torch.Generator().manual_seed(seed)
    pyG_datasets = torch.utils.data.random_split(
        pyG_dataset,
        [splits["train"], splits["val"], splits["test"]],
        generator=pyG_seed,
    )

    # Compute padding, using the first graph in the dataset.
    first_graph = pyG_dataset[0]
    padding = estimate_padding(first_graph)

    print("Padding", padding)
    datasets = {}
    for split, pyG_dataset in zip(["train", "val", "test"], pyG_datasets):
        logging.info(f"Creating {split} dataset...")

        # Convert pyG dataset to a generator of graphs.
        graphs_iterator = pyG_dataset_to_graphs_iterator(
            pyG_dataset, cutoff=cutoff, add_self_edges=add_self_edges
        )

        # Dynamically batch the graphs.
        datasets[split] = jraph.dynamically_batch(graphs_iterator, **padding)

    return datasets
