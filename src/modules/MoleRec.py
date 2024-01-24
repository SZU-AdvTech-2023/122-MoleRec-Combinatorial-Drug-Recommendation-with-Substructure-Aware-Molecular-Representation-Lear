from .SetTransformer import SAB
from .gnn import GNNGraph

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch
import math
import os

from .layers import GraphConvolution

os.environ['CUDA_VISIBEL_DEVICES'] = '5'


class GCN(nn.Module):
    def __init__(self, voc_size, emb_dim, ehr_adj, ddi_adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device
        ddi_adj = ddi_adj.cpu()
        ehr_adj = self.normalize(ehr_adj + np.eye(ehr_adj.shape[0]))
        ddi_adj = self.normalize(ddi_adj + np.eye(ddi_adj.shape[0]))

        self.ehr_adj = torch.FloatTensor(ehr_adj).to(device)
        self.ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)
        self.gcn3 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        ehr_node_embedding = self.gcn1(self.x, self.ehr_adj)
        ddi_node_embedding = self.gcn1(self.x, self.ddi_adj)

        ehr_node_embedding = F.relu(ehr_node_embedding)
        ddi_node_embedding = F.relu(ddi_node_embedding)
        ehr_node_embedding = self.dropout(ehr_node_embedding)
        ddi_node_embedding = self.dropout(ddi_node_embedding)

        ehr_node_embedding = self.gcn2(ehr_node_embedding, self.ehr_adj)
        ddi_node_embedding = self.gcn3(ddi_node_embedding, self.ddi_adj)
        return ehr_node_embedding, ddi_node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class AdjAttenAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, mid_dim, *args, **kwargs):
        super(AdjAttenAgger, self).__init__(*args, **kwargs)
        self.model_dim = mid_dim
        self.Qdense = torch.nn.Linear(Qdim, mid_dim)
        self.Kdense = torch.nn.Linear(Kdim, mid_dim)
        # self.use_ln = use_ln

    def forward(self, main_feat, other_feat, fix_feat, mask=None):
        # main_feat:global_embeddings  other_feat:substruct_embeddings  fix_feat: substruct_weight
        # 对应论文的Attention Block模块
        Q = self.Qdense(main_feat)
        K = self.Kdense(other_feat)
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)

        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))

        Attn = torch.softmax(Attn, dim=-1)
        # print(Attn[0])
        # print(mask[0])
        fix_feat = torch.diag(fix_feat)
        # 对应论文scale
        other_feat = torch.matmul(fix_feat, other_feat)
        # 对应论文Attention coef a 的 matmul
        O = torch.matmul(Attn, other_feat)

        return O


class MoleRecModel(torch.nn.Module):
    def __init__(
        self, global_para, substruct_para, emb_dim, voc_size,
        substruct_num, global_dim, substruct_dim, ehr_adj, ddi_adj, use_embedding=False,
        device=torch.device('cpu'), dropout=0.5, *args, **kwargs
    ):
        super(MoleRecModel, self).__init__(*args, **kwargs)
        self.device = device
        self.use_embedding = use_embedding

        if self.use_embedding:
            self.substruct_emb = torch.nn.Parameter(
                torch.zeros(substruct_num, emb_dim)
            )
        else:
            self.substruct_encoder = GNNGraph(**substruct_para)

        self.global_encoder = GNNGraph(**global_para)

        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),
            torch.nn.Embedding(voc_size[1], emb_dim)
        ])
        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)
        ])
        if dropout > 0 and dropout < 1:
            self.rnn_dropout = torch.nn.Dropout(p=dropout)
        else:
            self.rnn_dropout = torch.nn.Sequential()
        self.sab = SAB(substruct_dim, substruct_dim, 2, use_ln=True)
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 4, emb_dim)
        )
        self.substruct_rela = torch.nn.Linear(emb_dim, substruct_num)
        self.aggregator = AdjAttenAgger(
            global_dim, substruct_dim, max(global_dim, substruct_dim)
        )
        score_extractor = [
            torch.nn.Linear(substruct_dim, substruct_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(substruct_dim // 2, 1)
        ]
        self.score_extractor = torch.nn.Sequential(*score_extractor)
        self.init_weights()

        self.gcn = GCN(voc_size=voc_size[2], emb_dim=emb_dim, ehr_adj=ehr_adj, ddi_adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))
        self.inter2 = nn.Parameter(torch.FloatTensor(1))

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
        if self.use_embedding:
            torch.nn.init.xavier_uniform_(self.substruct_emb)

    def forward(
        self, substruct_data, mol_data, patient_data,
        ddi_mask_H, tensor_ddi_adj, average_projection
    ):
        seq1, seq2 = [], []
        # todo:消融实验，验证对纵向信息的依赖性，去除RNN
        # adm = patient_data[-1]
        # Idx1 = torch.LongTensor([adm[0]]).to(self.device)  # 诊断编码ID
        # Idx2 = torch.LongTensor([adm[1]]).to(self.device)  # 操作编码ID
        # repr1 = self.rnn_dropout(self.embeddings[0](Idx1))  # 对诊断编码ID进行embedding
        # repr2 = self.rnn_dropout(self.embeddings[1](Idx2))  # 对操作编码ID进行embedding
        # seq1.append(torch.sum(repr1, keepdim=True, dim=1))
        # seq2.append(torch.sum(repr2, keepdim=True, dim=1))
        for adm in patient_data:
            Idx1 = torch.LongTensor([adm[0]]).to(self.device)  # 诊断编码ID
            Idx2 = torch.LongTensor([adm[1]]).to(self.device)  # 操作编码ID
            repr1 = self.rnn_dropout(self.embeddings[0](Idx1))  # 对诊断编码ID进行embedding
            repr2 = self.rnn_dropout(self.embeddings[1](Idx2))  # 对操作编码ID进行embedding
            seq1.append(torch.sum(repr1, keepdim=True, dim=1))
            seq2.append(torch.sum(repr2, keepdim=True, dim=1))
        seq1 = torch.cat(seq1, dim=1)  # 把患者多次就医的诊断编码ID的embedding拼接成一个向量
        seq2 = torch.cat(seq2, dim=1)  # 把患者多次就医的操作编码ID的embedding拼接成一个向量
        output1, hidden1 = self.seq_encoders[0](seq1)
        output2, hidden2 = self.seq_encoders[1](seq2)
        seq_repr = torch.cat([hidden1, hidden2], dim=-1)
        last_repr = torch.cat([output1[:, -1],  output2[:, -1]], dim=-1)
        # 把所有隐藏状态e(i)和最后的结果e(t)拼接，（最后的结果就是最后的隐藏状态），得到E
        patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])

        # 对应论文Substructure Relevancy Module，通过上面的E得到c(i)
        # todo：消融实验把query（）（即SRM模块）删除
        query = self.query(patient_repr)
        # 对应select 得到Cs(i)
        substruct_weight = torch.sigmoid(self.substruct_rela(query))

        global_embeddings = self.global_encoder(**mol_data)  # 对应论文Graph Neural Network
        # todo：global_embeddings 为什么要跟平均投影相乘，投影到另一个空间？是否是对应论文的READOUT函数，
        # 进行平均投影，把属于同一个ACT-4编码的药物合并成一个向量，所以global_embeddings从283x64维变为131x64维
        # 比如ACT-4编码N02B下有两个药物
        # 1、'COC1=C2O[C@H]3C(=O)CC[C@@]4(O)[C@H]5CC(C=C1)=C2[C@@]34CCN5C’
        # 2、'CC(=O)NC1=CC=C(O)C=C1'
        # 对应着两个1x64的特征向量，然后
        ehr_embedding, ddi_embedding = self.gcn()
        drug_memory = ehr_embedding - ddi_embedding * self.inter
        global_embeddings = torch.mm(average_projection, global_embeddings)
        global_embeddings = (1 - self.inter2) * global_embeddings + self.inter2 * drug_memory
        # 对应论文Es -》 E*s，SAB自注意力模块，对应论文Substructure Interaction Module
        # 用于建模子结构之间的相互作用
        # todo：消融实验，把子结构相互作用模块删除
        # if self.use_embedding :
        #     substruct_embeddings = self.substruct_emb.unsqueeze(0)
        # else:
        #     substruct_embeddings = self.substruct_encoder(**substruct_data).unsqueeze(0)
        # substruct_embeddings = substruct_embeddings.squeeze(0)
        substruct_embeddings = self.sab(
            self.substruct_emb.unsqueeze(0) if self.use_embedding else
            # todo:消融试验，把下面的embedding改为GNN
            self.substruct_encoder(**substruct_data).unsqueeze(0)
        ).squeeze(0)
        torch.save(substruct_embeddings,'substruct_embeddings.pt')
        # 对应论文
        # 1、Attention Block -> Attention coef a
        # 2、E*s 乘 Cs
        # 3、a 乘 rs
        molecule_embeddings = self.aggregator(
            global_embeddings, substruct_embeddings,
            substruct_weight, mask=torch.logical_not(ddi_mask_H > 0)
        )
        # molecule_embeddings对应论文中的Rm(i)
        score = self.score_extractor(molecule_embeddings).t()  # 对应论文Prediction Module中的前馈神经网络

        neg_pred_prob = torch.sigmoid(score)  # 论文中的前馈神经网络后的sigmod激活函数
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(tensor_ddi_adj).sum()
        return score, batch_neg
