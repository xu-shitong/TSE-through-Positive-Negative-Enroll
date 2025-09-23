'''
Code for the Encoder Fusion Module
Adopted from the TFGridnet code provided in USEF-TSE: Universal Speaker Embedding Free Target Speaker Extraction
- https://github.com/ZBang/USEF-TSE
  - Original code licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
The modification includes adding the encoding branch, and split trainable parameters between encoding branch and extraction branch
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from improved_model.TFgridnet import GridNetV2Block, TF_gridnet_attentionblock

EPS = 1e-8


class STFT(nn.Module):
    def __init__(self, n_fft=256, hop_length=128, win_length=256):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
    
    def forward(self, y):
        num_dims = y.dim()
        assert num_dims == 2 or num_dims == 3, f"Only support 2D or 3D Input: {num_dims}"

        batch_size = y.shape[0]
        num_samples = y.shape[-1]

        if num_dims == 3:
            y = y.reshape(-1, num_samples)  # [B * C ,T]

        complex_stft = torch.stft(
            y,
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=torch.hann_window(self.win_length, device=y.device),
            return_complex=True,
        )
        _, num_freqs, num_frames = complex_stft.shape

        if num_dims == 3:
            complex_stft = complex_stft.reshape(batch_size, -1, num_freqs, num_frames)
        
        # print(complex_stft)

        mag = torch.abs(complex_stft)
        phase = torch.angle(complex_stft)
        real = complex_stft.real
        imag = complex_stft.imag
        return mag, phase, real, imag, complex_stft


class iSTFT(nn.Module):
    def __init__(self, n_fft=256, hop_length=128, win_length=256, length=None):
        super(iSTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.length = length
    
    def forward(self, features, input_type):
        if input_type == "real_imag":
        # the feature is (real, imag) or [real, imag]
            assert isinstance(features, tuple) or isinstance(features, list)
            real, imag = features
            features = torch.complex(real, imag)
        elif input_type == "complex":
            assert torch.is_complex(features), "The input feature is not complex."
        elif input_type == "mag_phase":
            # the feature is (mag, phase) or [mag, phase]
            assert isinstance(features, tuple) or isinstance(features, list)
            mag, phase = features
            features = torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
        else:
            raise NotImplementedError(
                "Only 'real_imag', 'complex', and 'mag_phase' are supported."
            )

        return torch.istft(
            features,
            self.n_fft,
            self.hop_length,
            self.win_length,
            window=torch.hann_window(self.win_length, device=features.device),
            length=self.length,
        )

class Tar_Model(nn.Module):

    def __init__(
        self,
        n_freqs,
        hidden_channels,
        n_head,
        emb_dim,
        emb_ks,
        emb_hs,
        num_layers=6,
        eps = 1e-5,
        encoder=None, encoder_head=None, train_encoder=False, train_encoder_head=False,
        binaural=False,
    ):
        super(Tar_Model, self).__init__()
        self.num_layers = num_layers
        self.binaural = binaural

        self.stft = STFT(n_fft=128,hop_length=64,win_length=128,)
        self.istft = iSTFT(n_fft=128,hop_length=64,win_length=128,)

        self.att = TF_gridnet_attentionblock(emb_dim=emb_dim,n_freqs=n_freqs ,n_head=4,approx_qk_dim=512,)

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)

        self.conv = nn.Sequential(
            nn.Conv2d(4 if binaural else 2, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        main_emb_dim = 2 * emb_dim

        self.deconv = nn.ConvTranspose2d(main_emb_dim, 4 if binaural else 2, ks, padding=padding)
        
        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    GridNetV2Block(
                        main_emb_dim,
                        emb_ks,
                        emb_hs,
                        n_freqs,
                        hidden_channels,
                        n_head,
                        approx_qk_dim=512,
                        activation="prelu",
                    )
                )
            )

        self.siamese = encoder
        self.encoder_head = encoder_head
        self.train_encoder = train_encoder
        self.train_encoder_head = train_encoder_head
        
    def to_train(self):
        self.train()

    def encoder_state_dict(self):
        return {"siamese": self.siamese.state_dict(), "encoder_head": self.encoder_head.state_dict()}
    
    def encoder_params(self):
        modules_ = [self.siamese, self.encoder_head]
        return sum([list(m.parameters()) for m in modules_], [])

    def main_params(self):
        modules_ = [self.conv, self.att, self.dual_mdl, self.deconv]
        return sum([list(m.parameters()) for m in modules_], [])

    def encoder(self, aux):
        # [B, N, L]
        std = aux.std(dim=(1, 2), keepdim=True)
        aux_c = self.stft(aux / std)[-1]

        aux_ri = torch.cat([aux_c.real, aux_c.imag],dim = 1)
        aux_ri = aux_ri.permute(0,1,3,2).contiguous()

        aux_ri = self.conv(aux_ri)

        return aux_ri, std
    
    def encoder_pos_neg(self, pos, neg, recons=False):
        pos = pos.transpose(1, 2)
        neg = neg.transpose(1, 2)
        if not self.train_encoder:
            with torch.no_grad():
                pos_emb = self.siamese(pos).detach()
                neg_emb = self.siamese(neg).detach()
        else:
            pos_emb = self.siamese(pos)
            neg_emb = self.siamese(neg)

        if not self.train_encoder_head:
            with torch.no_grad():
                cond_emb = self.encoder_head(pos_emb, neg_emb).detach()
        else:
            cond_emb = self.encoder_head(pos_emb, neg_emb)

        cond_emb = cond_emb[:, :, :pos_emb.shape[2]]
        return cond_emb, None, None


    def decoder(self, x, std):
        x = self.deconv(x)

        out_r = x[:,0,:,:].permute(0,2,1).contiguous()
        out_i = x[:,1,:,:].permute(0,2,1).contiguous()

        est_source = self.istft((out_r, out_i), input_type="real_imag").unsqueeze(1)

        est_source = est_source * std
        if self.binaural:
            out2_r = x[:,2,:,:].permute(0,2,1).contiguous()
            out2_i = x[:,3,:,:].permute(0,2,1).contiguous()
            est_source2 = self.istft((out2_r, out2_i), input_type="real_imag").unsqueeze(1)

            est_source2 = est_source2 * std

            est_source = torch.stack([est_source, est_source2], dim=1)
            return est_source
        else:
            return est_source.squeeze(1)

    def forward(self, input, emb):
        mix_ri, std = self.encoder(input)

        aux_ri = self.att(mix_ri, emb)

        x = torch.cat([mix_ri,aux_ri], dim=1)


        for i in range(self.num_layers):

            x = self.dual_mdl[i](x)
        
        return self.decoder(x, std)
