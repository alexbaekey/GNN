import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

def plot_graphs(graphs, graphs_dir, actors, plot_type=None):
    for name, graph in graphs.items():
        nx.relabel_nodes(graph,actors,copy=False)
        if(plot_type):
            #pos=nx.spring_layout(graph)
            #nx.draw(graph,pos)
            nx.draw_circular(graph,with_labels=True)
        else:
            nx.draw_networkx(graph)
        plt.savefig(f"{graphs_dir}{name}-graph.jpg")
        plt.clf()

