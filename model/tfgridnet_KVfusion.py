'''
Code for the cross attention based Extraction Fusion Module
Adopted from the TFGridnet code provided in ESPnet: end-to-end speech processing toolkit and LookOnceToHear
- ESPnet: https://github.com/espnet/espnet
- LookOnceToHear: https://github.com/vb000/lookoncetohear
The module is based on the Full-band Self-attention Module in the TFGridnet blocks
'''
import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.nn import init
from espnet2.torch_utils.get_layer_from_string import get_layer
import torch.nn.functional as F


class TFGridNet_KVfusion(nn.Module):
    def __init__(self,
        emb_dim,
        n_freqs,
        n_head=4,
        approx_qk_dim=512,
        activation="prelu",
        eps=1e-5,
        pooling_size=10,
        stride=5):
        super().__init__()

        self.model = nn.ModuleDict([])

        E = math.ceil(
            approx_qk_dim * 1.0 / n_freqs
        )  # approx_qk_dim is only approximate
        assert emb_dim % n_head == 0
        for ii in range(n_head):
            self.model.add_module(
                "attn_conv_Q_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.model.add_module(
                "attn_conv_K_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.model.add_module(
                "attn_conv_V_%d" % ii,
                nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim // n_head, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps),
                ),
            )
        self.model.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                get_layer(activation)(),
                LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
            ),
        )
        self.pooling_size = pooling_size
        self.stride = stride
        self.n_head = n_head

    def forward(self, batch, pos_cond, neg_cond):
        '''
        input shape: [B, C, T, F]
        pos_cond shape: [B, C, T, F]
        neg_cond shape: [B, C, T, F]
        '''
        B, C, T, _ = batch.shape

        # 1. change pos neg cond shape to [B, C, T // n, F]
        ## change cond to [B, C, F, T], then [B, C*F, T] shape
        pos_cond = pos_cond.transpose(2, 3) # [B, C, F, T]
        pos_cond = pos_cond.flatten(0, 1) # [B*C, F, T]
        pos_cond = nn.functional.avg_pool1d(input=pos_cond, kernel_size=self.pooling_size, stride=self.stride)
        pos_cond = pos_cond.transpose(1, 2) # [B*C, T', F]
        pos_cond = pos_cond.unflatten(dim=0, sizes=(B, C))

        if neg_cond is not None:
            neg_cond = neg_cond.transpose(2, 3) # [B, C, F, T]
            neg_cond = neg_cond.flatten(0, 1) # [B*C, F, T]
            neg_cond = nn.functional.avg_pool1d(input=neg_cond, kernel_size=self.pooling_size, stride=self.stride)
            neg_cond = neg_cond.transpose(1, 2) # [B*C, T', F]
            neg_cond = neg_cond.unflatten(dim=0, sizes=(B, C)) # [B, C, T', F]

            cond = torch.concat([pos_cond, -neg_cond], dim=2) # [B, C, 2T', F]
        else:
            cond = pos_cond

        # 2. cross attn
        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(self.model["attn_conv_Q_%d" % ii](batch))  # [B, C, T, Q]
            all_K.append(self.model["attn_conv_K_%d" % ii](cond))  # [B, C, 2T', Q]
            all_V.append(self.model["attn_conv_V_%d" % ii](cond))  # [B, C', 2T', Q]

        Q = torch.cat(all_Q, dim=0)  # [B', C, T, Q]
        K = torch.cat(all_K, dim=0)  # [B', C, 2T', Q]
        V = torch.cat(all_V, dim=0)  # [B', C', 2T', Q]

        Q = Q.transpose(1, 2)
        old_shape = Q.shape
        Q = Q.flatten(start_dim=2)  # [B', T, C*Q]
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)  # [B', 2T', C*Q]
        V = V.transpose(1, 2)  # [B', T, C, Q]
        V = V.flatten(start_dim=2)  # [B', 2T', C'*Q]
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, 2T']
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, 2T']
        V = torch.matmul(attn_mat, V)  # [B', T, C'*Q]

        V = V.reshape((old_shape[0], old_shape[1], -1, old_shape[3]))  # [B', T, C', Q]
        V = V.transpose(1, 2)  # [B', C', T, Q]
        emb_dim = V.shape[1]

        batch = V.view([self.n_head, B, emb_dim, T, -1])  # [n_head, B, C', T, Q])
        batch = batch.transpose(0, 1)  # [B, n_head, C', T, Q])
        batch = batch.contiguous().view(
            [B, self.n_head * emb_dim, T, -1]
        )  # [B, C', T, Q])
        batch = self.model["attn_concat_proj"](batch)  # [B, C'', T, Q])

        return batch

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
