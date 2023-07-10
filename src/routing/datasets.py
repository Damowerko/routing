from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from random import choice

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from routing.utils import auto_args

rng = np.random.default_rng()


@auto_args
class DijkstraDataset(InMemoryDataset):
    def __init__(self, root, n_samples=10_000, n_nodes=100, edge_factor=1.5, **kwargs):
        self.n_samples = n_samples
        self.n_nodes = n_nodes
        self.edge_factor = edge_factor
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return "dijkstra.pt"

    def process(self):
        with ProcessPoolExecutor() as e:
            data_list = list(
                tqdm(
                    e.map(
                        self.generate_sample,
                        repeat(self.n_nodes, self.n_samples),
                        repeat(self.edge_factor, self.n_samples),
                    ),
                    total=self.n_samples,
                    desc="Generating samples",
                )
            )

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def generate_sample(n_nodes: int, edge_factor: float = 1.5):
        """
        Generate a random geometric graph and find the distance matrix using Dijkstra.
        Args:
            n_nodes: number of nodes in the graph
            threshold: maximum distance between nodes for them to be connected
        Returns:
            g: networkx graph
            D: distance matrix
        """

        # start with a random tree
        G = nx.random_tree(n_nodes)

        # add weights to tree
        for u, v in G.edges:
            G[u][v]["distance"] = rng.uniform(0.0, 1.0)

        distances = dict(nx.all_pairs_dijkstra_path_length(G), weight="distance")

        # add random edges
        n_edges = int(edge_factor * n_nodes)
        while G.number_of_edges() < n_edges:
            u = choice(G.nodes)
            v = choice(G.nodes)
            if u == v:
                continue
            if G.has_edge(u, v):
                continue
            # the edge has larger distance than d(u,v) so it doesn't affect the shortest path
            min_dist: float = distances[u][v]  # type: ignore
            G.add_edge(u, v, distance=rng.uniform(min_dist, min_dist + 1.0))

        _eye = torch.eye(n_nodes)
        for source, distances in nx.all_pairs_dijkstra_path_length(
            G, weight="distance"
        ):
            G.nodes[source]["y"] = [distances[dest] for dest in G.nodes()]
            G.nodes[source]["x"] = _eye[source].tolist()
        return from_networkx(G, group_edge_attrs=["distance"])
