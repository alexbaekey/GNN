import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

def plot_graphs(graphs, graphs_dir, actors, edge_type, te_val, plot_type=None):
    for edge_type, graph in graphs.items():
        nx.relabel_nodes(graph,actors,copy=False)
        if(plot_type):
            #pos=nx.spring_layout(graph)
            #nx.draw(graph,pos)
            nx.draw_circular(graph,with_labels=True)
        else:
            nx.draw_networkx(graph)
        plt.savefig(f"{graphs_dir}{edge_type}-te-{te_val}-graph.jpg")
        plt.clf()

def te_rollout(in_roots, in_edges_df, max_visits=6): 
    lengths = {}
    te_levels_df = {}
    all_root_dfs = {}
    for in_root in in_roots:
        visited = {}
        root_df = pd.DataFrame()
        for node in range(10000):
            visited.update({node:0})
        this_level_nodes = in_root
        te_values = []
        this_level = 0
        while True:
            if(this_level==0):
                this_level_nodes = [this_level_nodes]
            last_visited = visited.copy()
            for node in this_level_nodes:
                visited[node] += 1
            if(last_visited == visited):
                lengths.update({in_root:this_level})
                break
            this_level += 1
            edges_from_this_level = in_edges_df[in_edges_df['Source'].isin(this_level_nodes)]

            visited_cap = set([k for k, v in visited.items() if v > max_visits])
            e = edges_from_this_level[~edges_from_this_level['Target'].isin(visited_cap)]
            #updated to work with pandas >=2.0
            #root_df = root_df.append(e, ignore_index=True)
            root_df = pd.concat([root_df, e], ignore_index=True)
            this_level_nodes = set(edges_from_this_level['Target'].to_list()).difference(visited_cap)

        all_root_dfs.update({in_root:root_df})
    
    return lengths, all_root_dfs


