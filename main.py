import numpy as np
import random
import os
import copy 
import torch_geometric as pyg
import torch
import pandas as pd
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils.convert import from_networkx
dataset = EllipticBitcoinDataset(root='data/whole_graph')
data = dataset[0]




# -------------------------- GCN ------------------------------ # 
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x, data.edge_index)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc


for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')



# -------------------------- GAT ------------------------------ # 
# from torch_geometric.nn import GATConv


# class GAT(torch.nn.Module):
#     def __init__(self, hidden_channels, heads):
#         super().__init__()
#         torch.manual_seed(1234567)
#         self.conv1 = GATConv(dataset.num_features, hidden_channels, heads)
#         self.conv2 = GATConv(hidden_channels*heads, dataset.num_classes, heads)


#     def forward(self, x, edge_index):
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv1(x, edge_index)
#         x = F.elu(x)
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x

# model = GAT(hidden_channels=8, heads=8)
# print(model)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
# criterion = torch.nn.CrossEntropyLoss()

# def train():
#       model.train()
#       optimizer.zero_grad()  # Clear gradients.
#       out = model(data.x, data.edge_index)  # Perform a single forward pass.
#       loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
#       loss.backward()  # Derive gradients.
#       optimizer.step()  # Update parameters based on gradients.
#       return loss

# def test(mask):
#       model.eval()
#       out = model(data.x, data.edge_index)
#       pred = out.argmax(dim=1)  # Use the class with highest probability.
#       correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
#       acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
#       return acc


# for epoch in range(1, 201):
#     loss = train()
#     val_acc = test(data.val_mask)
#     test_acc = test(data.test_mask)
#     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

# # Testing
# model.eval()
# with torch.no_grad():
#     logits = model(data.x, data.edge_index)
#     pred = logits.argmax(dim=1)
#     test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
#     print(f"Test Accuracy: {test_acc:.4f}")


# -------------------------- GraphSAGE ------------------------------ # 

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.nn import SAGEConv


# Define the GraphSAGE model
class GraphSAGENet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Instantiate the model
model = GraphSAGENet(in_channels=dataset.num_features, hidden_channels=64, out_channels=dataset.num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Split the dataset into a training and testing set
data = dataset[0]

# Training
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')


# Testing
model.eval()
with torch.no_grad():
    logits = model(data)
    pred = logits.argmax(dim=1)
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    print(f"Test Accuracy: {test_acc:.4f}")