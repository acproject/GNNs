import torch
import torch.nn as nn
import torch as th
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from modules.layers import *
from modules.functions import *
from modules.embedding import *
from modules.viz import att_animation, get_attention_map
from optims import NoamOpt
from loss import LabelSmoothing, SimpleLossCompute
from dataset import get_dataset, GraphPool

import dgl.function as fn
import torch.nn.init as INIT
from torch.nn import LayerNorm


class MultiHeadAttention(nn.Module):
    def __init__(self, h, dim_model):
        '''

        :param h: number of heads
        :param dim_model: hidden dimension
        '''
        super(MultiHeadAttention, self).__init__()
        self.d_k = dim_model // h
        self.h = h
        # W_q, W_k, W_v, W_o
        self.linears = clones(nn.Linear(dim_model, dim_model), 4)
