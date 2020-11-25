import torch
import torch.nn as nn



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
