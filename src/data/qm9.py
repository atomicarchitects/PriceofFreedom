"""Exposes the QM9 datasets in a convenient format."""

from typing import Dict, Iterator

from sklearn.preprocessing import StandardScaler
from absl import logging
import torch_geometric as pyG
import jraph
import numpy as np
import jax
import jax.numpy as jnp
import torch


from src.data import utils


def to_graph(data: pyG.data.Data, target_property_index: int) -> jraph.GraphsTuple:
    """Converts a datum from a PyG dataset to a jraph.GraphsTuple."""
    data = data.to_dict()
    data = {
        key: val.numpy() if isinstance(val, torch.Tensor) else val
        for key, val in data.items()
    }

    # Compute relative positions.
    pos = data["pos"]
    senders = data["edge_index"][0, :]
    receivers = data["edge_index"][1, :]
    relative_vectors = pos[receivers] - pos[senders]

    data = jraph.GraphsTuple(
        nodes=dict(
            positions=data["pos"],
            numbers=data["z"],
        ),
        edges=dict(
            relative_vectors=relative_vectors,
        ),
        globals=np.asarray([data["y"][:, target_property_index]]),
        n_node=np.asarray([pos.shape[0]]),
        n_edge=np.asarray([relative_vectors.shape[0]]),
        senders=senders,
        receivers=receivers,
    )
    return data


class QM9Dataset(torch.utils.data.Dataset):
    """Exposes the QM9 dataset in a convenient format."""

    def __init__(
        self,
        root: str,
        target_property: str,
        radial_cutoff: float,
        add_self_edges: bool,
        splits: Dict[str, int],
        seed: int,
    ):
        self.radial_cutoff = radial_cutoff
        self.add_self_edges = add_self_edges

        # Define the properties.
        properties = {
            0: {"name": "mu", "description": "Dipole moment", "unit": "D"},
            1: {
                "name": "alpha",
                "description": "Isotropic polarizability",
                "unit": "a_0^3",
            },
            2: {
                "name": "epsilon_HOMO",
                "description": "Highest occupied molecular orbital energy",
                "unit": "eV",
            },
            3: {
                "name": "epsilon_LUMO",
                "description": "Lowest unoccupied molecular orbital energy",
                "unit": "eV",
            },
            4: {
                "name": "Delta epsilon",
                "description": "Gap between epsilon_HOMO and epsilon_LUMO",
                "unit": "eV",
            },
            5: {
                "name": "R^2",
                "description": "Electronic spatial extent",
                "unit": "a_0^2",
            },
            6: {
                "name": "ZPVE",
                "description": "Zero point vibrational energy",
                "unit": "eV",
            },
            7: {"name": "U_0", "description": "Internal energy at 0K", "unit": "eV"},
            8: {"name": "U", "description": "Internal energy at 298.15K", "unit": "eV"},
            9: {"name": "H", "description": "Enthalpy at 298.15K", "unit": "eV"},
            10: {"name": "G", "description": "Free energy at 298.15K", "unit": "eV"},
            11: {
                "name": "c_v",
                "description": "Heat capavity at 298.15K",
                "unit": "cal/(mol K)",
            },
            12: {
                "name": "U_0_ATOM",
                "description": "Atomization energy at 0K",
                "unit": "eV",
            },
            13: {
                "name": "U_ATOM",
                "description": "Atomization energy at 298.15K",
                "unit": "eV",
            },
            14: {
                "name": "H_ATOM",
                "description": "Atomization enthalpy at 298.15K",
                "unit": "eV",
            },
            15: {
                "name": "G_ATOM",
                "description": "Atomization free energy at 298.15K",
                "unit": "eV",
            },
            16: {"name": "A", "description": "Rotational constant", "unit": "GHz"},
            17: {"name": "B", "description": "Rotational constant", "unit": "GHz"},
            18: {"name": "C", "description": "Rotational constant", "unit": "GHz"},
        }
        # Search for the target property.
        target_property_index = None
        for key, value in properties.items():
            if value["name"] == target_property:
                target_property_index = key
                logging.info(
                    f"Target property {target_property}: {value['description']} ({value['unit']})"
                )
                break
        if target_property_index is None:
            raise ValueError(
                f"Unknown target property {target_property}. Available properties are: {', '.join([value['name'] for value in properties.values()])}"
            )
        self.target_property_index = target_property_index

        # Split the dataset.
        dataset = pyG.datasets.QM9(
            root=root,
            transform=None,
            pre_transform=utils.add_edges_transform(
                radial_cutoff=radial_cutoff, add_self_edges=add_self_edges
            ),
            force_reload=False,
        )
        self.datasets = utils.split_dataset(dataset, splits, seed)

    def to_jraph_graphs(
        self, split: str, batch_size: int
    ) -> Iterator[jraph.GraphsTuple]:
        """Returns batched and padded graphs."""
        logging.info(f"Creating {split} dataset.")

        batch = []
        padding = None

        while True:
            for data in self.datasets[split]:
                data_as_jraph = to_graph(
                    data, target_property_index=self.target_property_index
                )
                batch.append(data_as_jraph)

                if len(batch) == batch_size:
                    batch_jraph = jraph.batch(batch)

                    # Compute padding if not already computed.
                    if padding is None:
                        padding = dict(
                            n_node=int(
                                2.0 * np.ceil(batch_jraph.n_node.sum() / 64) * 64
                            ),
                            n_edge=int(
                                2.0 * np.ceil(batch_jraph.n_edge.sum() / 64) * 64
                            ),
                            n_graph=batch_size + 1,
                        )
                        logging.info(f"Split {split}: Padding computed as {padding}")

                    batch_jraph = jraph.pad_with_graphs(batch_jraph, **padding)
                    yield batch_jraph

                    batch = []

    def get_datasets(self, batch_size: int) -> Dict[str, Iterator[jraph.GraphsTuple]]:
        """Returns the splits of the dataset."""
        return {
            split: self.to_jraph_graphs(split, batch_size)
            for split in self.datasets.keys()
        }
