import os
import numpy as np
import torch as th
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from networkx.algorithms import bipartite

def get_attention_map(g, src_nodes, dst_nodes, h):
    '''
    To visualize the attention score between two set of nodes
    :param g:
    :param src_nodes:
    :param dst_nodes:
    :param h:
    :return:
    '''
    n, m = len(src_nodes), len(dst_nodes)
    weight = th.zeros(n. m, h).fill_(-1e8)
    for i, src in enumerate(src_nodes.tolist()):
        for j, dst in enumerate(dst_nodes.tolist()):
            if not g.has_edge_between(src, dst):
                continue
            eid = g.edge_id(src, dst)
            weight[i][j] = g.edata['score'][eid].squeeze(-1).cpu().detach()

    weight = weight.transpose(0, 2)
    att = th.softmax(weight, -2)
    return att.numpy()
