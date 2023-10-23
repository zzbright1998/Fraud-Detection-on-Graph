import torch
from torch_geometric.datasets import EllipticBitcoinDataset
import torch.nn as nn
import torch.nn.functional as F
# from ogb.nodeproppred import Evaluator
from sklearn import metrics as metrics




# licit -> label 0
# unknown -> label 2
# illicit -> label 1
if torch.cuda.is_available():
    device = torch.device("cuda:1")  # This selects the first GPU. For second GPU use "cuda:1" and so on.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

dataset = EllipticBitcoinDataset(root='data/whole_graph')
data = dataset[0]

# -------------------------- GCN ------------------------------ # 
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout):
        super().__init__()
        torch.manual_seed(777)

        # Convolution layers
        if num_layers > 1:
            self.convs = nn.ModuleList([GCNConv(dataset.num_features, hidden_channels)])
            self.convs.extend([GCNConv(hidden_channels, hidden_channels) for i in range(num_layers - 2)])
            self.convs.append(GCNConv(hidden_channels, dataset.num_classes))
        
            # Batch normilization 
            self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels) 
                                     for i in range(num_layers - 1)])
        else:
            self.convs = nn.ModuleList([GCNConv(dataset.num_features, dataset.num_classes)])
            self.bns = nn.ModuleList([])

        # Softmax layer
        self.softmax = nn.LogSoftmax(1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    # initialize parameters
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for gcn, bn in zip(self.convs, self.bns):
            x = self.dropout(torch.relu(bn(gcn(x, edge_index))))
        x = self.convs[-1](x, edge_index)
        
        return self.softmax(x)


# -------------------------- GAT ------------------------------ # 
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads, num_layers, dropout):
        super().__init__()
        torch.manual_seed(777)
        self.num_layers = num_layers
        
        if num_layers > 1:
            # GAT layers
            self.convs = nn.ModuleList([GATConv(dataset.num_features, hidden_channels, heads)])
            self.convs.extend([GATConv(heads*hidden_channels, hidden_channels, heads) for i in range(num_layers - 2)])
            self.convs.append(GATConv(heads*hidden_channels, dataset.num_classes))

            # Batch Normilization
            self.bns = nn.ModuleList([nn.BatchNorm1d(heads*hidden_channels) 
                                     for i in range(num_layers - 1)])
        else:
            self.convs = nn.ModuleList([GATConv(dataset.num_features, dataset.num_classes)])
            self.bns = nn.ModuleList([])

         # Softmax layer
        self.softmax = nn.LogSoftmax(1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    # initialize parameters
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for gat, bn in zip(self.convs, self.bns):
            x = self.dropout(torch.relu(bn(gat(x, edge_index))))
        x = self.convs[-1](x, edge_index)
        
        return self.softmax(x)



# -------------------------- GraphSAGE ------------------------------ # 
from torch_geometric.nn import SAGEConv

class GraphSAGENet(nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout):
        super().__init__()
        torch.manual_seed(777)

        if num_layers > 1:
            # Convolution layers
            self.convs = nn.ModuleList([SAGEConv(dataset.num_features, hidden_channels)])
            self.convs.extend([SAGEConv(hidden_channels, hidden_channels) for i in range(num_layers - 2)])
            self.convs.append(SAGEConv(hidden_channels, dataset.num_classes))
        
             # Batch normilization 
            self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels) 
                                     for i in range(num_layers - 1)])

        else:
            self.convs = nn.ModuleList([SAGEConv(dataset.num_features, dataset.num_classes)])
            self.bns = nn.ModuleList([])

        # Softmax layer
        self.softmax = nn.LogSoftmax(1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    # initialize parameters
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for gcn, bn in zip(self.convs, self.bns):
            x = self.dropout(torch.relu(bn(gcn(x, edge_index))))
        x = self.convs[-1](x, edge_index)
        
        return self.softmax(x)
    


#----------------------------Train---------------------------------------#
def train(model, data, optimizer, loss_fn):
    model.train()

    # Clear gradients.
    optimizer.zero_grad()

    # feed datas into the model
    output = model(data.x, data.edge_index)
    
    # Get the model's predictions and labels
    pred, label = output[data.train_mask], data.y[data.train_mask].view(-1)

    loss = loss_fn(pred, label)

    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss.item()

#------------------------------Test-------------------------------------#
def test(model, data):
    model.eval()

    output = model(data.x, data.edge_index)
    y_pred = output.argmax(dim=1)

    # train_acc = metrics.accuracy_score(data.y[data.train_mask], y_pred[data.train_mask])
    # test_acc = metrics.accuracy_score(data.y[data.test_mask], y_pred[data.test_mask])
    # test_pre = metrics.precision_score(data.y[data.test_mask], y_pred[data.test_mask])
    # test_recall = metrics.recall_score(data.y[data.test_mask], y_pred[data.test_mask])
    # test_f1 = metrics.f1_score(data.y[data.test_mask], y_pred[data.test_mask])
    train_acc = metrics.accuracy_score(data.y[data.train_mask].cpu(), y_pred[data.train_mask].cpu())
    test_acc = metrics.accuracy_score(data.y[data.test_mask].cpu(), y_pred[data.test_mask].cpu())
    test_pre = metrics.precision_score(data.y[data.test_mask].cpu(), y_pred[data.test_mask].cpu())
    test_recall = metrics.recall_score(data.y[data.test_mask].cpu(), y_pred[data.test_mask].cpu())
    test_f1 = metrics.f1_score(data.y[data.test_mask].cpu(), y_pred[data.test_mask].cpu())

    return train_acc, test_acc, test_pre, test_recall, test_f1

#--------------------------------Run Model-------------------------------#
def runModel(model, data, optimizer, loss_fn):
    model.reset_parameters()
    data = data.to(device)
    for epoch in range(1, 501):
        loss = train(model, data, optimizer, loss_fn)
        result = test(model, data)
        train_acc, _, _, _, _ = result
    
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100*train_acc:.2f}%')

    result = test(model, data)
    _, test_acc, test_pre, test_recall, test_f1 = result
    print(f'Test Accuracy: {100*test_acc:.2f}%  '
          f'Test Precision: {100*test_pre:.2f}%  '
          f'Test Recall: {100*test_recall:.2f}%  '
          f'Test F1: {100*test_f1:.2f}%  ')


weight = torch.tensor([11, 1.1]).to(device)
model_GAT = GAT(hidden_channels=64, heads=8, num_layers=2, dropout=0.3).to(device)
model_SAGE = GraphSAGENet(hidden_channels=64, num_layers=2, dropout=0.3).to(device)
model_GCN = GCN(hidden_channels=128, num_layers=2, dropout=0.3).to(device)
model = model_GAT
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
loss_fn = torch.nn.CrossEntropyLoss(weight = weight)

runModel(model, data, optimizer, loss_fn)
