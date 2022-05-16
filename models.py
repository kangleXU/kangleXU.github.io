import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from dgl.nn.pytorch import GATConv
from han_gat import GAT, SpGAT
import math

CUDA = torch.cuda.is_available()

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)

class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, num_meta_paths, in_ent_size, hid_size, out_ent_size,  num_heads, dropout, alpha):
        super(HANLayer, self).__init__()
        # num_meta_paths: 10
        # in_ent_size : 50
        # hid_size : 8
        # out_ent_size : 200
        # num_heads: 8
        # dropout : 0.3
        # alpha : 0.2
        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):       # num_heads 表示多头注意力的头数
            self.gat_layers.append(GAT(in_ent_size, hid_size, num_heads, out_ent_size, dropout, alpha))
        self.semantic_attention = SemanticAttention(in_size=hid_size * num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []
        # 每个mate-path下对应到的节点级别的attention
        for i, g in enumerate(gs):  # 每个mate-path的图信息，求节点的attention.
            g = g.cuda()
            g= Variable(g)
            semantic_embeddings.append(self.gat_layers[i](h, g).flatten(1))   # 计算语义级别att，这里还需要修改一下。
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)   #每个节点对应到 mate-path下的每个节点embedding的值                # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

class HAN(nn.Module):
    def __init__(self, num_meta_paths, embedd_dim, hidden_size, entity_out_dim, num_heads, dropout, alpha ):
        super(HAN, self).__init__()
        #num_meta_paths: 10
        #embedd_dim : 50
        # hidden_size : 8
        # entity_out_dim : 200
        #num_heads: [8]
        #dropout : 0.3
        # alpha : 0.2

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths,embedd_dim, hidden_size, entity_out_dim[0], num_heads[0], dropout, alpha))
        # self.layers.append(HANLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)): # 这里表示做多层HANlayer
            self.layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[l-1],hidden_size, entity_out_dim , num_heads[l], dropout,alpha))
        self.W1 = nn.Parameter(torch.empty(size=(embedd_dim, entity_out_dim[0])))  # 50  * 200
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
    def forward(self, g, h, relation_embeddings):
        out_relation_1 = relation_embeddings.mm(self.W1)
        for gnn in self.layers:
            h = gnn(g, h)  # g是多个mate-path下的邻接矩阵， h是节点特征



        return  h , out_relation_1



