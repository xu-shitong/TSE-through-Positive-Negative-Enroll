'''
Code for one branch of the Siamese Encoder
Adopted from the TFGridnet code provided in ESPnet: end-to-end speech processing toolkit and LookOnceToHear
- ESPnet: https://github.com/espnet/espnet
- LookOnceToHear: https://github.com/vb000/lookoncetohear
The modification includes removal of the final pooling layers and layers that decode embedding back to waveforms
'''

import torch
from espnet2.enh.separator.tfgridnet_separator import TFGridNet as TFGridNet_original

class TFGridNet_encoder(TFGridNet_original):
    def __init__(self, num_ch, n_fft, stride, num_blocks, binaural):
        super().__init__(
            input_dim=None, n_fft=n_fft, stride=stride, n_imics=num_ch, n_srcs=1,
            lstm_hidden_units=64, n_layers=num_blocks, emb_dim=64)
        self.binaural = binaural

    def forward(
        self,
        input: torch.Tensor,
        ilens=None,
    ) -> torch.Tensor:
        '''
        input shape: [B, audio length N, audio channel M]
        ilens shape: [B]
        '''
        
        if ilens is None:
            ilens = torch.tensor([_.shape[0] for _ in input])

        if not self.binaural:
            input = torch.concat([input, input], dim=2) # duplicate mono audio to create binaural audio 

        mix_std_ = torch.std(input, dim=(1, 2), keepdim=True)  # [B, 1, 1]
        input = input / mix_std_  # RMS normalization

        batch = self.enc(input, ilens)[0]  # [B, T, M, F]
        batch0 = batch.transpose(1, 2)  # [B, M, T, F]
        batch = torch.cat((batch0.real, batch0.imag), dim=1)  # [B, 2*M, T, F]

        batch = self.conv(batch)  # [B, -1, T, F]

        for ii in range(self.n_layers):
            batch = self.blocks[ii](batch)  # [B, -1, T, F]

        return batch
