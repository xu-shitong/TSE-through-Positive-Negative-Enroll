'''
Code for the Encoder Fusion Module
Adopted from the TFGridnet code provided in ESPnet: end-to-end speech processing toolkit and LookOnceToHear
- ESPnet: https://github.com/espnet/espnet
- LookOnceToHear: https://github.com/vb000/lookoncetohear
The modification includes the concatenation of the input two embedding sequences, and the addition of Segmentation Embeddings
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from espnet2.torch_utils.get_layer_from_string import get_layer
from espnet2.enh.separator.tfgridnet_separator import GridNetBlock

class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat

class GridNetBlock_attnhead(nn.Module):
    def __init__(self, layer_num, pooling_size, stride, return_clean_dvec=False, out_dim=0, refine_layer_num=0, 
                 fusion_shortcut=[0], cut_pos=False):
        super().__init__()

        self.pooling_size = pooling_size
        self.stride = stride

        self.segment_embedding = nn.Embedding(2, 64 * 65)

        self.model = nn.ModuleList([])
        for _ in range(layer_num):
            self.model.append(
                GridNetBlock_attn(
                    emb_dim=64,
                    emb_ks=1,
                    emb_hs=1,
                    n_freqs=65,
                    n_head=4,
                    eps=1.0e-5,
                ))
        if return_clean_dvec:
            self.embed_proj = nn.Sequential(
                nn.Linear(65 * 64, 256),
                nn.LayerNorm(256),
            )

        self.refine_layer_num = refine_layer_num
        if refine_layer_num > 0:
            self.pending_module = nn.ModuleList([])
            for _ in range(refine_layer_num):
                self.pending_module.append(GridNetBlock(
                    emb_dim=64,
                    emb_ks=1,
                    emb_hs=1,
                    n_freqs=65,
                    hidden_channels=64,
                    n_head=4,
                    approx_qk_dim=512,
                    activation="prelu",
                    eps=1.0e-5,
                ))

        self.fusion_shortcut = fusion_shortcut
        self.cut_pos = cut_pos

        if out_dim != 0:
            assert not return_clean_dvec, "hotfix for now: linear project for stylespeech is different from dvec output"
            self.embed_proj = nn.Sequential(
                nn.Linear(65 * 64, out_dim),
            )
            
    def forward(self, pos_cond, neg_cond):
        B, C, T_pos, F = pos_cond.shape
        B, C, T_neg, F = neg_cond.shape

        x = torch.concat([pos_cond, neg_cond], dim=2) # [B, C, 2T', F]

        seg_idx = torch.concat([torch.zeros((B, T_pos), device=pos_cond.device), torch.ones((B, T_neg), device=pos_cond.device)], dim=1)
        seg_emb = self.segment_embedding(seg_idx.to(torch.int32)) # [B, 2T', C * F]
        seg_emb = seg_emb.unflatten(dim=2, sizes=(C, F)).permute((0, 2, 1, 3)) # [B, C, 2T', F]
        
        x = x + seg_emb

        for ii, layer in enumerate(self.model):
            if ii in self.fusion_shortcut:
                x = x + layer(x)
            else:
                x = layer(x)

        if self.cut_pos:
            x = x[:, :, :T_pos]

        if self.refine_layer_num > 0:
            for ii in range(self.refine_layer_num):
                x = self.pending_module[ii](x)  # [B, -1, T, F]
        return x

class GridNetBlock_attn(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        n_head=4,
        approx_qk_dim=512,
        activation="prelu",
        eps=1e-5,
    ):
        super().__init__()

        E = math.ceil(
            approx_qk_dim * 1.0 / n_freqs
        )  # approx_qk_dim is only approximate
        assert emb_dim % n_head == 0
        for ii in range(n_head):
            self.add_module(
                "attn_conv_Q_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_K_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                "attn_conv_V_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim // n_head, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps),
                ),
            )
        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                get_layer(activation)(),
                LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
            ),
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        """GridNetBlock Forward.

        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """
        B, C, old_T, old_Q = x.shape
        batch = x

        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(self["attn_conv_Q_%d" % ii](batch))  # [B, C, T, Q]
            all_K.append(self["attn_conv_K_%d" % ii](batch))  # [B, C, T, Q]
            all_V.append(self["attn_conv_V_%d" % ii](batch))  # [B, C, T, Q]

        Q = torch.cat(all_Q, dim=0)  # [B', C, T, Q]
        K = torch.cat(all_K, dim=0)  # [B', C, T, Q]
        V = torch.cat(all_V, dim=0)  # [B', C, T, Q]

        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)  # [B', T, C*Q]
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)  # [B', T, C*Q]
        V = V.transpose(1, 2)  # [B', T, C, Q]
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*Q]

        V = V.reshape(old_shape)  # [B', T, C, Q]
        V = V.transpose(1, 2)  # [B', C, T, Q]
        emb_dim = V.shape[1]

        batch = V.view([self.n_head, B, emb_dim, old_T, -1])  # [n_head, B, C, T, Q])
        batch = batch.transpose(0, 1)  # [B, n_head, C, T, Q])
        batch = batch.contiguous().view(
            [B, self.n_head * emb_dim, old_T, -1]
        )  # [B, C, T, Q])
        batch = self["attn_concat_proj"](batch)  # [B, C, T, Q])

        return batch
