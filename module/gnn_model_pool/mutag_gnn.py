import time
import random
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential as Seq, ReLU, Tanh, Linear as Lin, Softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric
from torch_geometric.utils import degree
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINEConv, BatchNorm, global_mean_pool
from module.utils import set_seed, Gtrain, Gtest

# print(os.getcwd())
# print(__file__)
# print(torch.__version__)
# print(torch.__file__)
# print(np.__version__)
# print(np.__file__)
# print(torch_geometric.__version__)
# print(torch_geometric.__file__)
# print(torch.cuda.is_available())
# print(torch.backends.mps.is_available())
