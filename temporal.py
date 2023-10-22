import numpy as np
import random
import os
import copy 
import torch_geometric as pyg
import torch
import pandas as pd
from torch_geometric.datasets import EllipticBitcoinTemporalDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils.convert import from_networkx

dataset = []
for i in range(1, 50):
    print('processing {}/49 graph'.format(i))
    dataset.append(EllipticBitcoinTemporalDataset(root='data/temporal', t=i))