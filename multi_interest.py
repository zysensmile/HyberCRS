import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import pickle
import gzip
import numpy as np
import time

from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv
import dgl.nn as dglnn
import dgl.function as fn
import dgl
from utils import *


def G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence sparse matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    DV = torch.sum(H, dim=1, keepdim=True) + 1e-5
    DE = torch.sum(H, dim=0, keepdim=True) + 1e-5

    invDE = torch.diag(DE.pow(-1).reshape(-1))
    DV2 = torch.diag(DV.pow(-1).reshape(-1))
    HT = H.transpose(0, 1)

    G = DV2[:1, :].matmul(H).matmul(invDE).matmul(HT).matmul(DV2)

    return G


class SelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0., anchor=False):
        super().__init__()
        self.anchor = anchor
        if anchor:
            self.scorer = nn.Linear(d_hid * 2, 1)
        else:
            self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters(d_hid)

    def reset_parameters(self, d_hid):
        if self.anchor:
            stdv = 1. / math.sqrt(d_hid * 2)
        else:
            stdv = 1. / math.sqrt(d_hid)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input_seq, anchor=None, s=None):
        batch_size, seq_len, feature_dim = input_seq.size()
        input_seq = self.dropout(input_seq)

        if anchor != None:
            size = input_seq.shape[1]
            anchor = anchor.repeat(1, size, 1)
            seq = torch.cat((input_seq, anchor), 2)
            # enablePrint()
            # ipdb.set_trace()
            scores = self.scorer(seq.contiguous().view(-1, feature_dim * 2)).view(batch_size, seq_len) + s
        else:
            scores = self.scorer(input_seq.contiguous().view(-1, feature_dim)).view(batch_size, seq_len)
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(input_seq).mul(input_seq).sum(1)
        return context, scores  # 既然命名为context就应该是整句的表示


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class HGNN_conv(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(HGNN_conv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # hyperG=G_from_H(adj)
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def outprop(self, H_input, adj):
        # hyperG=G_from_H(adj)
        output = torch.sparse.mm(adj, H_input)
        return output


class GraphEncoder(Module):
    def __init__(self, graph, device, entity, emb_size, kg, embeddings=None, fix_emb=True, seq='rnn', gcn=True,
                 hidden_size=100, layers=1, rnn_layer=1, u=None, v=None, f=None):
        super(GraphEncoder, self).__init__()

        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        # self.eps = 0.0
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        rel_names = ['interact', 'friends', 'like', 'belong_to']
        self.G = graph.to(device)
        self.conv1 = dglnn.HeteroGraphConv({rel: dglnn.GraphConv(emb_size, hidden_size) for rel in rel_names},
                                           aggregate='mean')

        self.embedding = nn.Embedding(entity, emb_size, padding_idx=entity - 1)
        if embeddings is not None:
            print("pre-trained embeddings")
            self.embedding.from_pretrained(embeddings, freeze=fix_emb)
        self.layers = layers
        self.user_num = u
        self.item_num = v
        self.PADDING_ID = entity - 1
        self.device = device
        self.seq = seq
        self.gcn = gcn
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        # self.fc_frd = nn.Linear(hidden_size, hidden_size)
        if self.seq == 'rnn':
            self.rnn = nn.GRU(hidden_size, hidden_size, rnn_layer, batch_first=True)
        elif self.seq == 'transformer':
            self.transformer1 = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=400),
                num_layers=rnn_layer)
            self.transformer2 = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=400),
                num_layers=rnn_layer)
            self.transformer3 = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=400),
                num_layers=rnn_layer)
            self.transformer4 = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=400),
                num_layers=rnn_layer)

        if self.gcn:
            indim, outdim = emb_size, hidden_size
            self.gnns = nn.ModuleList()
            self.hypergnns = nn.ModuleList()
            for l in range(layers):
                self.hypergnns.append(HGNN_conv(indim, outdim))
                indim = outdim
        else:
            self.fc2 = nn.Linear(emb_size, hidden_size)

        self.position_embedding = nn.Embedding(9, hidden_size)
        # self.num_pers=4
        # self.multi_head_self_attention_init = nn.ModuleList([SelfAttention(self.hidden_size, 0.3) for _ in range(self.num_pers)])
        # self.multi_head_self_attention = nn.ModuleList([SelfAttention(self.hidden_size, 0.3,anchor=True) for _ in range(self.num_pers)])

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def ssl(self, b_state):
        tau = 0.6  # default = 0.8
        f = lambda x: torch.exp(x / tau)
        hyper_batch_output = []
        for s in b_state:
            # neighbors, adj = self.get_state_graph(s)
            hyperneigh, HT = s['hyperneigh'].to(self.device), s['hyperHT'].to(self.device)
            hyper_input_state = self.embedding(hyperneigh)
            if self.gcn:
                for hypergnn in self.hypergnns:
                    hyper_output_state = hypergnn(hyper_input_state, HT)
                    # cand_act_embedding=hypergnn.outprop(hyper_output_state, G)
                    hyper_input_state = hyper_output_state
                hyper_batch_output.append(hyper_output_state)

        ssl_loss_set = []
        for s, o in zip(b_state, hyper_batch_output):
            # seq_embeddings.append(o[:len(s['cur_node']),:][None,:])
            ssl_loss = torch.tensor(0.0).cuda()
            # cnt = 1e-10
            cnt = 10
            if s['lens'][0] > 0 and s['lens'][1] > 0:
                feature_pos_emb = o[:sum(s['lens'][:1])]
                feature_posctr_emb = feature_pos_emb.mean(axis=0, keepdim=True)
                feature_neg_emb = o[sum(s['lens'][:1]):sum(s['lens'][:2])]
                feature_negctr_emb = feature_neg_emb.mean(axis=0, keepdim=True)
                pos_pos_sim = f(self.sim(feature_posctr_emb, feature_pos_emb))
                pos_neg_sim = f(self.sim(feature_posctr_emb, feature_neg_emb))
                neg_pos_sim = f(self.sim(feature_negctr_emb, feature_pos_emb))
                neg_neg_sim = f(self.sim(feature_negctr_emb, feature_neg_emb))
                ssl_loss += -torch.log(pos_pos_sim.sum() / pos_neg_sim.sum())
                ssl_loss += -torch.log(neg_neg_sim.sum() / neg_pos_sim.sum())
                # if len(s['friend'])>0:
                #     feature_frd_emb=o[len(s['acc_feature'])+len(s['rej_feature']):len(s['acc_feature'])+len(s['rej_feature'])+len(s['friend'])]
                #     feature_frdctr_emb=feature_frd_emb.mean(axis=0, keepdim=True)
                #     frd_pos_sim=f(self.sim(feature_posctr_emb, feature_frd_emb))
                #     frd_neg_sim=f(self.sim(feature_negctr_emb, feature_frd_emb))
                #     ssl_loss += -torch.log(frd_pos_sim.sum() / frd_neg_sim.sum())
            ssl_loss_set.append(ssl_loss / cnt)
        ssl_loss_mean = torch.mean(torch.stack(ssl_loss_set))
        return ssl_loss_mean

    def forward(self, b_state, b_act=None):
        hyper_batch_output = []
        # hyper_cand_emb=[]
        for s in b_state:
            hyperneigh, HT = s['hyperneigh'].to(self.device), s['hyperHT'].to(self.device)
            hyper_input_state = self.embedding(hyperneigh)
            if self.gcn:
                for hypergnn in self.hypergnns:
                    hyper_output_state = hypergnn(hyper_input_state, HT)
                    # cand_act_embedding=hypergnn.outprop(hyper_output_state, G)
                    hyper_input_state = hyper_output_state
                hyper_batch_output.append(hyper_output_state)

        seq_embeddings_feature = []
        seq_embeddings_item = []
        seq_embeddings_user = []
        flag_item = []
        for s, o in zip(b_state, hyper_batch_output):
            tmp_seq_embeddings_feature = []
            tmp_seq_embeddings_feature.append(o[:sum(s['lens'][:1]), :][None, :] + self.position_embedding(torch.tensor(0).to(self.device)))
            if s['lens'][1] > 0:
                tmp_seq_embeddings_feature.append(o[sum(s['lens'][:1]):sum(s['lens'][:2]), :][None, :] \
                    + self.position_embedding(torch.tensor(1).to(self.device)))
            if s['lens'][2] > 0:
                tmp_seq_embeddings_feature.append(o[sum(s['lens'][:2]):sum(s['lens'][:3]), :][None, :] \
                    + self.position_embedding(torch.tensor(2).to(self.device)))

            tmp_seq_embeddings_item = []
            if sum(s['lens'][3:6]) > 0:
                if s['lens'][3] > 0:
                    tmp_seq_embeddings_item.append(o[sum(s['lens'][:3]):sum(s['lens'][:4]), :][None, :] \
                        + self.position_embedding(torch.tensor(3).to(self.device)))
                if s['lens'][4] > 0:
                    tmp_seq_embeddings_item.append(o[sum(s['lens'][:4]):sum(s['lens'][:5]), :][None, :] \
                        + self.position_embedding(torch.tensor(4).to(self.device)))
                if s['lens'][5] > 0:
                    tmp_seq_embeddings_item.append(o[sum(s['lens'][:5]):sum(s['lens'][:6]), :][None, :] \
                        + self.position_embedding(torch.tensor(5).to(self.device)))
            else:
                tmp_seq_embeddings_item.append(torch.zeros(1,1,100).to(self.device))

            tmp_seq_embeddings_user = []
            tmp_seq_embeddings_user.append(o[sum(s['lens'][:6]):sum(s['lens'][:7]), :][None, :] + self.position_embedding(torch.tensor(6).to(self.device)))
            if s['lens'][7] > 0:
                tmp_seq_embeddings_user.append(o[sum(s['lens'][:7]):sum(s['lens'][:8]), :][None, :] + self.position_embedding(torch.tensor(7).to(self.device)))
            if s['lens'][8] > 0:
                tmp_seq_embeddings_user.append(o[sum(s['lens'][:8]):sum(s['lens'][:9]), :][None, :] + self.position_embedding(torch.tensor(8).to(self.device)))

            seq_embeddings_feature.append(torch.cat(tmp_seq_embeddings_feature, dim=1))
            seq_embeddings_item.append(torch.cat(tmp_seq_embeddings_item, dim=1))
            seq_embeddings_user.append(torch.cat(tmp_seq_embeddings_user, dim=1))
        if len(b_state) > 1:
            seq_embeddings_feature, src_key_padding_mask_feature = self.padding_seq(seq_embeddings_feature)
            seq_embeddings_item, src_key_padding_mask_item = self.padding_seq(seq_embeddings_item)
            seq_embeddings_user, src_key_padding_mask_user = self.padding_seq(seq_embeddings_user)
        else:
            src_key_padding_mask_feature = torch.tensor([[False]*len(seq_embeddings_feature[0][0])]).to(self.device)
            src_key_padding_mask_item = torch.tensor([[False]*len(seq_embeddings_item[0][0])]).to(self.device)
            src_key_padding_mask_user = torch.tensor([[False]*len(seq_embeddings_user[0][0])]).to(self.device)

        seq_embeddings_feature = torch.cat(seq_embeddings_feature, dim=0)
        seq_embeddings_feature = self.transformer1(seq_embeddings_feature.permute(1, 0, 2), src_key_padding_mask=src_key_padding_mask_feature)
        seq_embeddings_feature = seq_embeddings_feature.permute(1, 0, 2).masked_fill(src_key_padding_mask_feature.unsqueeze(-1), 0)
        non_padding_count = src_key_padding_mask_feature.ne(1).sum(dim=1, keepdim=True).float()
        seq_embeddings_feature = seq_embeddings_feature.sum(dim=1, keepdim=True)/ non_padding_count.unsqueeze(-1)


        seq_embeddings_item = torch.cat(seq_embeddings_item, dim=0)
        seq_embeddings_item = self.transformer2(seq_embeddings_item.permute(1, 0, 2))
        seq_embeddings_item = seq_embeddings_item.permute(1, 0, 2).masked_fill(src_key_padding_mask_item.unsqueeze(-1), 0)
        non_padding_count = src_key_padding_mask_item.ne(1).sum(dim=1, keepdim=True).float()
        seq_embeddings_item = seq_embeddings_item.sum(dim=1, keepdim=True)/ non_padding_count.unsqueeze(-1)

        seq_embeddings_user = torch.cat(seq_embeddings_user, dim=0)
        seq_embeddings_user = self.transformer3(seq_embeddings_user.permute(1, 0, 2))
        seq_embeddings_user = seq_embeddings_user.permute(1, 0, 2).masked_fill(src_key_padding_mask_user.unsqueeze(-1),0)
        non_padding_count = src_key_padding_mask_user.ne(1).sum(dim=1, keepdim=True).float()
        seq_embeddings_user = seq_embeddings_user.sum(dim=1, keepdim=True) / non_padding_count.unsqueeze(-1)

        seq_embeddings = torch.cat([seq_embeddings_feature,
                                    seq_embeddings_item,
                                    seq_embeddings_user], dim=1)
        seq_embeddings = self.transformer4(seq_embeddings.permute(1, 0, 2))
        seq_embeddings = seq_embeddings.permute(1, 0, 2)
        seq_embeddings = seq_embeddings.mean(dim=1, keepdim=True)



        # seq_embeddings_feature = F.relu(self.fc1(seq_embeddings_feature))
        seq_embeddings = F.relu(self.fc1(seq_embeddings))
        return seq_embeddings

    def padding_seq(self, seq):
        src_key_padding_mask = []
        padding_size = max([len(x[0]) for x in seq])
        padded_seq = []
        for s in seq:
            cur_size = len(s[0])
            emb_size = len(s[0][0])
            new_s = torch.zeros((padding_size, emb_size)).to(self.device)
            new_s[:cur_size, :] = s[0]
            padded_seq.append(new_s[None, :])
            src_key_padding_mask.append([False]*cur_size + [True]*(padding_size-cur_size))
        return padded_seq, torch.tensor(src_key_padding_mask).to(self.device)

    def padding(self, cand_embs):
        pad_size = max([len(c) for c in cand_embs])
        padded_cand = []
        for c in cand_embs:
            cur_size = len(c)
            new_c = torch.zeros((pad_size - cur_size, c.size(1))).to(self.device)
            padded_cand.append(torch.cat((c, new_c), dim=0))
        return padded_cand
