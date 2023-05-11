

'''
def generate_graphs(num_graphs=100, num_nodes=50):
    graphs = []
    labels = []
    for _ in range(num_graphs):
        p = np.random.uniform(0.01, 0.1)
        G = nx.gnp_random_graph(num_nodes, p, directed=True)
        graphs.append(dgl.from_networkx(G))
        labels.append(0)  # random graph label
        m = np.random.randint(1, 5)
        G = nx.barabasi_albert_graph(num_nodes, m)
        G = nx.DiGraph(G)  # make the graph directed
        graphs.append(dgl.from_networkx(G))
        labels.append(1)  # scale-free graph label
        # Add the directed synthetic rich-club network class
        G = generate_rich_club_network(num_nodes)
        graphs.append(dgl.from_networkx(G))
        labels.append(2)  # directed synthetic rich-club network label
    return graphs, torch.tensor(labels)

'''

import numpy as np
import networkx as nx
import dgl
import torch
import random


# Not enough edges
def generate_rich_club_network(n, phi=0.7):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # Connect nodes with higher probability for higher degree nodes
    for i in range(n):
        for j in range(i + 1, n):
            p = phi * (G.in_degree(i) + G.out_degree(i) + 1) * (G.in_degree(j) + G.out_degree(j) + 1) / (n * n)
            if np.random.uniform(0, 1) < p:
                G.add_edge(i, j)
            if np.random.uniform(0, 1) < p:
                G.add_edge(j, i)

    return G



# NOTE phi controls the "strength" of the rich-club phenomenon, i.e. # of connections among hubs

def generate_rich_club_network2(n, m, phi=0.7):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # Start with a random directed graph
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.uniform(0, 1) < phi:
                G.add_edge(i, j)
            if np.random.uniform(0, 1) < phi:
                G.add_edge(j, i)

    # Add m additional edges to create the rich-club phenomenon
    sorted_nodes = sorted(G.nodes, key=lambda x: G.degree(x), reverse=True)
    for _ in range(m):
        i, j = random.sample(sorted_nodes[:n//10], 2)
        if not G.has_edge(i, j):
            G.add_edge(i, j)
        if not G.has_edge(j, i):
            G.add_edge(j, i)

    return G




def generate_graphs(num_graphs=100, num_nodes=50):
    graphs = []
    labels = []

    for _ in range(num_graphs):
        p = np.random.uniform(0.01, 0.1)
        G = nx.gnp_random_graph(num_nodes, p, directed=True)
        graphs.append(dgl.from_networkx(G))
        labels.append(0)  # random graph label

        m = np.random.randint(1, 5)
        G = nx.barabasi_albert_graph(num_nodes, m)
        G = nx.DiGraph(G)  # make the graph directed
        graphs.append(dgl.from_networkx(G))
        labels.append(1)  # scale-free graph label

        # Add the directed synthetic rich-club network class
        #G = generate_rich_club_network(num_nodes, 0.9)
        G = generate_rich_club_network2(num_nodes, int(0.1*np.power(num_nodes,2)), 0.05)
        #G = generate_rich_club_network2(num_nodes, 0, 0.05)
        graphs.append(dgl.from_networkx(G))
        labels.append(2)  # directed synthetic rich-club network label
        print(G)
    return graphs, torch.tensor(labels)


graphs, labels = generate_graphs(num_graphs=1024)
print(graphs)
print(labels)

