import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from util import plot_graphs, plot_betweenness_centrality, plot_degree, te_rollout, te_rollout_addnodes, plot_path_lengths, plot_htrees
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
cascade_df = graph_df.loc[(graph_df['Target'] > 1.) & (graph_df['Source']<max_nodes) & (graph_df['Target']<max_nodes)]

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
graphs = {}

# Main
if __name__ == "__main__":
    for edge_type in edge_types:
        # Filter for TE edges above threshold value
        graph_df1 = cascade_df.loc[(graph_df[edge_type] > te_thresh)]
        g = nx.from_pandas_edgelist(graph_df1, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())

        # Collect generated graphs labeled by edge types (UFTM classifications)
        graphs.update({edge_type:g})
        
        # Differentiate between total TE threshold and individual TE thresholds
        if(edge_type == 'total_te'):
            thresh = te_total_thresh
        elif(edge_type != 'total_te'):
            thresh = te_thresh

    
    ##### Pathways analysis #####
    subgraphs = {}
    for edge_type in edge_types:
        #for te_thresh in [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
        threshes = np.arange(0,0.5,0.1)
        threshes = list(np.round(threshes, 1))
        for te_thresh in threshes:
            cascade_df = graph_df.loc[(graph_df[edge_type] > te_thresh) & \
                                        (graph_df['Target'] > 1.) & (graph_df['Source']<101.) & (graph_df['Target']<101.)]
            
            # root nodes are those identified previously as most influential.
            # In the dynamic v4, these nodes are: 12=Ian56789, 23=NAJ562, 100=peter_pobjecky, 32=georgegalloway
            #root_nodes=[12, 32, 100, 23]
            root_nodes=[12]
            root_graphs = {}
            lengths, all_root_dfs = te_rollout(in_roots = root_nodes, in_edges_df = cascade_df, max_visits=vis_lim)
            root_graphs = {}
            for roots, root_df in all_root_dfs.items():
                g = nx.from_pandas_edgelist(root_df, 'Source', 'Target', [edge_type], create_using=nx.DiGraph())
                root_graphs.update({roots:g})
                subgraphs.update({f'{edge_type}-{te_thresh}':g})
            plot_graphs(root_graphs, subgraphs_dir, actors = actors_orig, edge_type=edge_type, te_val = te_thresh)

    print(subgraphs) 
    with open('subgraphs.pickle', 'wb') as handle:
        pickle.dump(subgraphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
