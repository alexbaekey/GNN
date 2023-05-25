import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from util import plot_graphs
import csv
import pickle

# Directories
subgraphs_dir = 'results/subgraphs/'

# Thresholds for overview analytics (degree/centrality)
te_thresh = 0.05 # used for influence across classifications, ex// TM_TM, UM_TM
te_total_thresh = 0.1

# Limits
vis_lim = 5
dep_lim = 10
max_nodes = 101. # To disregard communities (nodes 101 and up)

# Dataframe of TE network (v2/v4/dynamic)
graph_df = pd.read_csv('data/actor_te_edges_2018_03_01_2018_05_01.csv')
graph_df = graph_df.loc[(graph_df['Target'] > 1.) & (graph_df['Source']<max_nodes) & \
(graph_df['Target']<max_nodes)]

# Dict of actor names (v2/v4/dynamic)
actor_df = pd.read_csv('data/actors.csv')
actors = dict(zip(actor_df.actor_id, actor_df.actor_label))
actors_orig = actors
orig_nodes = list(actors_orig.values())

# Capture all edge types
from_edges = ['UF', 'UM', 'TF', 'TM']
to_edges   = ['UF', 'UM', 'TF', 'TM']

edge_types = ['UF_TM','UF_TM']
for i in from_edges:
    for j in to_edges:
        #edge_types.append(str(i + '_' + j))
        edge_types.append(f"{i}_{j}")

# initialize places to store results
subgraphs = {}

# Main
if __name__ == "__main__":
    for edge_type in edge_types:
        threshes = np.arange(0,0.51,0.01)
        threshes = list(np.round(threshes, 2))
        for te_thresh in threshes:
            subgraph_df = graph_df.loc[(graph_df[edge_type] > te_thresh) & \
            (graph_df['Target'] > 1.) & (graph_df['Source']<101.) & (graph_df['Target']<101.)] 
            
            g = nx.from_pandas_edgelist(subgraph_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
            subgraphs.update({f'{edge_type}-{te_thresh}':g})
            

    print(subgraphs) 
    plot_graphs(subgraphs, subgraphs_dir, actors = actors_orig)
    
    with open('subgraphs.pickle', 'wb') as handle:
        pickle.dump(subgraphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
