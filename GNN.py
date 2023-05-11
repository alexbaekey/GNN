import os
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim as optim
from torch.utils.data import DataLoader
from generate_rich_clubs import generate_graphs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# generate random and scale-free graphs
#def generate_graphs(num_graphs=100, num_nodes=50):
#    graphs = []
#    labels = []
#    for _ in range(num_graphs):
#        p = np.random.uniform(0.01, 0.1)
#        G = nx.gnp_random_graph(num_nodes, p)
#        graphs.append(dgl.from_networkx(G))
#        labels.append(0)  # random graph label
#        m = np.random.randint(1, 5)
#        G = nx.barabasi_albert_graph(num_nodes, m)
#        G = dgl.from_networkx(G)
#        graphs.append(G)
#        labels.append(1)  # scale-free graph label
#    return graphs, torch.tensor(labels)

graphs, labels = generate_graphs(num_graphs=1024)



class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCNClassifier, self).__init__()
        self.gcn1 = dgl.nn.GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True)
        self.gcn2 = dgl.nn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.classify = nn.Linear(hidden_dim, n_classes)
    def forward(self, g):
        h = g.in_degrees().view(-1, 1).float()
        h = F.relu(self.gcn1(g, h))
        h = F.relu(self.gcn2(g, h))
        g.ndata['h'] = h
        # average over node attribute 'h'
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)


def evaluate(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    
    for data in dataloader:
        g, labels = data
        g = g.to(device)
        labels = labels.to(device)
        #data = data.to(device)
        with torch.no_grad():
            output = model(g)
            pred = output.argmax(dim=1)
            y_pred.extend(pred.cpu().numpy())
            #y_true.extend(data.y.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    #precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    #recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    #f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1



# TODO add option to save/load model
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def load_model(model, save_path):
    if os.path.isfile(save_path):
        model.load_state_dict(torch.load(save_path))
        print("model loaded from", save_path)
        return
    else:
        print("no saved model found", save_path)
        return


# train test split, dataloaders
train_graphs, test_graphs, train_labels, test_labels = train_test_split(graphs, labels, test_size=0.2, random_state=42)
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)
train_data = list(zip(train_graphs, train_labels))
test_data = list(zip(test_graphs, test_labels))

#train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate)
#test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=collate)

train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, collate_fn=collate)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False, collate_fn=collate)



# initializing model
model = GCNClassifier(in_dim=1, hidden_dim=64, n_classes=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()
n_epochs = 1000


# training 
#TODO accuracy training curve
epoch_t = []
loss_t  = []
for epoch in range(n_epochs):
    epoch_loss = 0
    for batched_graph, batch_labels in train_dataloader:
        batched_graph = batched_graph.to(device)
        batch_labels = batch_labels.to(device)
        # forward propagation
        logits = model(batched_graph)
        # calculate loss
        loss = loss_func(logits, batch_labels)
        epoch_loss += loss.item()
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
    epoch_t.append(epoch)
    loss_t.append(epoch_loss / len(train_dataloader))
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss / len(train_dataloader)}")
    if (loss < 0.00001):
        print("model converged")
        pass



acc, prec, rec, f1 = evaluate(model, test_dataloader, device)
print(f'acc: {acc}')
print(f'prec: {prec}')
print(f'rec: {rec}')
print(f'f1: {f1}')

import matplotlib.pyplot as plt

plt.plot(epoch_t, loss_t)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('GCN classifier')
plt.savefig('lossplot.png')

#plt.clf()
#plt.plot(acc_t
