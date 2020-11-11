import torch
from torch import Tensor
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.datasets import KarateClub

dataset: torch.utils.data.Dataset = KarateClub()
print(f'Dataset: {dataset}：')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

# Gather some statistics about the graph.

data = dataset[0]
# 由于现在的KarateClub没有train_mask, 所以自己加上
import numpy as np
# randint的区间[0, 2)
np_array= np.random.randint(0, 2 ,34)

mask = np_array >=1
# print(mask)
train_mask = torch.from_numpy(mask)
# 加入自动随机生成的mask
data['train_mask'] = train_mask

print(data)
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

edge_index: torch.Tensor = data.edge_index
print(edge_index.t())


def visualize(h: torch.Tensor, color: any, epoch=None, loss=None) -> None:
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")



from torch_geometric.utils import to_networkx

G: nx.Graph = to_networkx(data, to_undirected=True)
visualize(G, color=data.y)

import torch
from torch.nn import Linear, Module
from torch_geometric.nn import GCNConv

class GCN(Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1: Module= GCNConv(dataset.num_features, 4)
        self.conv2: Module = GCNConv(4, 4)
        self.conv3: Module = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x: int, edge_index: int):
        h: Tensor = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh() # 最终的GNN内嵌空间

        out = self.classifier(h)
        return out, h

model = GCN()
print(model)


model = GCN()
_, h = model(data.x, data.edge_index)
print(f'Embedding shape: {list(h.shape)}')

visualize(h, color=data.y)

import time
from torch.optim import Optimizer

model = GCN()

criterion: Module = torch.nn.CrossEntropyLoss() # 定义损失函数
optimizer: Optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # 定义优化器

def train(data: torch.utils.data.Dataset) -> (Module, Tensor):
    optimizer.zero_grad()  # 在训练过程中清空导数（梯度）
    out, h = model(data.x, data.edge_index)
    loss: Module = criterion(out[data.train_mask], data.y[data.train_mask])  # 计算损失
    loss.backward()  # 方向传播，进行梯度计算
    optimizer.step() # 基于梯度计算后的结果进行更新优化参数
    return loss, h

for epoch in range(401):
    loss, h = train(data)
    if epoch % 10 == 0:
        visualize(h ,color=data.y, epoch=epoch, loss=loss)
        time.sleep(0.3)