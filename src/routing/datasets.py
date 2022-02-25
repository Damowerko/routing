from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
from routing.utils import auto_args
import networkx as nx
import numpy as np
import torch


@auto_args
class DijkstraDataset(InMemoryDataset):
    def __init__(self, root, **kwargs):
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return "dijkstra.pt"

    def process(self):
        n_samples = 10000
        n_nodes = 10
        threshold = 0.4

        data_list = [self.generate_sample(n_nodes, threshold) for _ in range(n_samples)]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def generate_sample(n_nodes=10, threshold=0.4):
        """
        Generate a random geometric graph and find the distance matrix using Dijkstra.
        Args:
            n_nodes: number of nodes in the graph
            threshold: maximum distance between nodes for them to be connected
        Returns:
            g: networkx graph
            D: distance matrix
        """
        G = None
        while G is None or not nx.is_connected(G):
            G = nx.random_geometric_graph(n_nodes, threshold)
        for e in G.edges():
            p1 = np.asarray(G.nodes[e[0]]["pos"])
            p2 = np.asarray(G.nodes[e[1]]["pos"])
            d = np.linalg.norm(p1 - p2)
            G[e[0]][e[1]]["weight"] = d

        for i, node in enumerate(G.nodes()):
            distance = nx.single_source_dijkstra_path_length(G, node)
            G.nodes[node]["y"] = [distance[node] for node in G.nodes()]
            G.nodes[node]["x"] = np.eye(n_nodes)[i].tolist()
        return from_networkx(G, group_edge_attrs=["weight"])
