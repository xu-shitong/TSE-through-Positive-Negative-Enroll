'''
Code for the monaural models' evaluation
Reference:
- TCE repository: https://github.com/chentuochao/Target-Conversation-Extraction/tree/main
- SpeakerBeam repository: https://github.com/BUTSpeechFIT/speakerbeam/tree/main
'''
from utils import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from tqdm import tqdm
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr_loss,
    signal_noise_ratio as snr_loss,
    signal_distortion_ratio as si_sdr_loss)
import numpy as np
from torch import nn
from dataset.LibriSpeech_single_emb import LibriDataset_single_emb
import random

# Import packages
import sys,humanize,psutil,GPUtil

# Define function
def mem_report():
    print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))

    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
print("using device: ", dev)

# ========= hyperparams ===========

test_data_dir = "data/LibriSpeech/LibriSpeech/_test_data/"

source_num = 3 # number of speakers in the Audio Mixture (including the target speaker)
enroll_num = 3 # number of speakers in the Enrollments (including the target speaker)
pos_example_length = 48000 # Positive Enrollment length, in waveform length (pos_example_length / 16000 is the length in second)
neg_example_length = 48000 # Negative Enrollment length, in waveform length (neg_example_length / 16000 is the length in second)
noise_dir = "data/wham_noise/"

pos_num = [0, 2] # range of number of positive interferer
hybrid_num = 0 # number of Hybrid Interferers
neglect_require_num = 0 # number of Neglect-Required Interferers


# ======== model =========
args = dict(
    model_name = "tfgridnet_causal",

    model_path = "output/proposed-monaural.pt",

    fusion_name = "tfgridnet_kv",
    model_normalize = True,
    pooling_size=40,
    fusion_stride=40,
    head_layer_num=2,
    fusion_layer=[0,1],
    return_dvec = False,
)

# args = dict(
#     model_name = "usef-tf",
#     load_model="output/improved-monaural.pt",

#     hidden_channels=64,
#     emb_dim=64,
#     enc_num_block=1,

#     refine_layer_num=2,
#     fusion_shortcut=[0,1],
#     cut_pos=True,

#     return_dvec = False,

# )


# ======== default hyperparameters =========

repeat_num = 100
set_size = 50
active_num = [-1, 1, -1]
wave_length = 6 # 1 fixed to 1 for now, decidde later how much audio waveform should be left after fftconvolve
filling_pattern = "repeat"
dvec_rate = 50
reverb_cond = False
snr_db_range = [-2.5, 2.5]
partial_range = [0.33, 0.66]
neg_partial_range = [0.33, 1.0] # disturbing speaker are either partial or full disturb

# ======== model ========
if args["model_name"] == "tfgridnet_causal":
    from model.tfgridnet_encoder import TFGridNet_encoder
    encoder = TFGridNet_encoder(
            num_ch=2,
            n_fft=128,
            stride=64,
            num_blocks=3,
            binaural=False,
        )

    from model.GridnetAttnHead import GridNetBlock_attnhead
    encoder_head = GridNetBlock_attnhead(
        layer_num=args["head_layer_num"],
        pooling_size=1,
        stride=1,
    )

    from model.tfgridnet_KVfusion import TFGridNet_KVfusion
    from model.tfgridnet_crossattn_causal_single_emb import TFGridNet_origcrossattn_causal_single_emb
    model = TFGridNet_origcrossattn_causal_single_emb(
        n_fft=128,
        stride=64,
        n_layers=3,
        lstm_hidden_units=64,
        emb_dim=64,
        emb_ks=1,
        model_normalize = args["model_normalize"],
        Fusion_class=TFGridNet_KVfusion,
        pooling_size=args["pooling_size"],
        fusion_stride=args["fusion_stride"],
        encoder=encoder,
        encoder_head=encoder_head,
        train_encoder=False,
        train_encoder_head=False,
        fusion_layer=args["fusion_layer"],
        binaural=False,
    )
    model.to(device)
    model.load_state_dict(torch.load(
        args["model_path"],
        map_location=torch.device('cpu'))["state_dict"],
        strict=True)
    sample_rate = 16000

    same_disturb = False

    normalize = False

if args["model_name"] == "usef-tf":
    from model.tfgridnet_encoder import TFGridNet_encoder
    encoder = TFGridNet_encoder(
            num_ch=2,
            n_fft=128,
            stride=64,
            num_blocks=args["enc_num_block"],
            binaural=False,
        )        

    from improved_model.GridnetAttnHead import GridNetBlock_attnhead
    encoder_head = GridNetBlock_attnhead(
        layer_num=2,
        pooling_size=1,
        stride=1,
        return_clean_dvec=False,
        out_dim=0,
        refine_layer_num=args["refine_layer_num"],
        fusion_shortcut=args["fusion_shortcut"],
        cut_pos=args["cut_pos"],
    )

    from improved_model.USEF_TFGridnet import Tar_Model
    model = Tar_Model(
        n_freqs=65,
        hidden_channels=64,
        n_head=4,
        emb_dim=64,
        emb_ks=1,
        emb_hs=1,
        num_layers=3,
        encoder=encoder,
        encoder_head=encoder_head,
        train_encoder=False,
        train_encoder_head=False,            
    ) 
    model.to(device)

    if args["load_model"] != "":
        model.load_state_dict(torch.load(args["load_model"])["state_dict"], strict=True)

    sample_rate = 16000

    same_disturb = False

    normalize = False


# ======= metric evaluators =========
from pypesq import pesq
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from speechmos import dnsmos

def compute_pesq(ref_audio, deg_audio, sample_rate=16000):
    return pesq(ref_audio, deg_audio, sample_rate)

stoi_model = ShortTimeObjectiveIntelligibility(fs=sample_rate, extended=False)
def compute_stoi(ref_audio, deg_audio):
    return stoi_model(ref_audio, deg_audio).item()


# ===========================
model.eval()

test_dataloader = LibriDataset_single_emb(
    test_data_dir, sample_rate=sample_rate, wave_length=wave_length*sample_rate, pos_example_length=pos_example_length, neg_example_length=neg_example_length,
    snr_db_range=snr_db_range, source_num=source_num, min_source_num=source_num, enroll_num=enroll_num, min_enroll_num=enroll_num, active_num=active_num, 
    reproducable=True, normalize=False, filling_pattern=filling_pattern,
    return_dvec=args["return_dvec"], dvec_rate=dvec_rate, include_silent=False, special_spk=[],
    reverb='none', binaural=False, reverb_cond=reverb_cond, zero_in_tgt=False, noise_dir=noise_dir + "tt/", 
    same_disturb=same_disturb)

def eval(model, dataloader, epoch):
    acc_mse = []
    acc_snr = []
    acc_si_snr = []
    acc_imp_snr = []
    acc_imp_si_snr = []
    acc_si_sdr = []
    acc_imp_si_sdr = []

    acc_pesq = []
    acc_stoi = []
    acc_dnsmos = []    

    with torch.no_grad():
        for i in tqdm(range(set_size)):
            audio, pos, neg = dataloader[i + epoch * set_size]
            audio = audio.to(device)[None]
            pos = pos.to(device)[None]
            neg = neg.to(device)[None]

            gt_resampled = audio[:, :active_num[1]].sum(dim=1)

            # special speakers are not simulated by dataset, but added using the following code
            # partial interferer
            partial_num = random.randint(pos_num[0], pos_num[1])
            for i in range(partial_num):
                active_len = int(pos_example_length * random.uniform(partial_range[0], partial_range[1]))
                start = random.randint(0, pos_example_length - active_len)
                end = start + active_len
            
                pos[:, active_num[1] + i, :, :start] = 0 
                pos[:, active_num[1] + i, :, end:] = 0 
                neg[:, i] = 0

            # negtive interferer
            for i in range(enroll_num-1):
                active_len = int(neg_example_length * random.uniform(neg_partial_range[0], neg_partial_range[1]))
                start = random.randint(0, neg_example_length - active_len)
                end = int(start + active_len)
                neg[:, i, :, :start] = 0
                neg[:, i, :, end:] = 0
            
            # hybrid
            for i in range(hybrid_num):
                active_len = int(pos_example_length * random.uniform(partial_range[0], partial_range[1]))
                start = random.randint(0, pos_example_length - active_len)
                end = int(start + active_len)
                pos[:, -2 - i, :, :start] = 0
                pos[:, -2 - i, :, end:] = 0

            # neglect-required
            for i in range(neglect_require_num):
                pos[:, -2 - i] = 0
            
            if args["model_name"] == "tfgridnet_causal":
                cond_emb = model.encode(pos.sum(dim=1), neg.sum(dim=1))
                init_state = model.init_buffers(audio.shape[0], device)
                out, init_state = model(audio.sum(dim=1), cond_emb, init_state)

            elif args["model_name"] == "usef-tf":
                cond_emb, pos_std, dec_audio = model.encoder_pos_neg(pos.sum(dim=1), neg.sum(dim=1), recons=False)
                out = model(audio.sum(dim=1), cond_emb)
                out = out.unsqueeze(1)

            if normalize:
                out = (
                    out / out.abs().max(dim=-1, keepdim=True)[0] * gt_resampled.abs().max(dim=-1, keepdim=True)[0]
                )
                
            out = torch.concat([out, -out], dim=1)
            idxs = torch.min((out - gt_resampled).pow(2).sum(dim=-1), dim=1)[1]
            idxs = idxs[:, None, None].repeat((1, 1, out.shape[-1]))

            out = torch.gather(out, 1, idxs)                
            mse_l = torch.nn.functional.mse_loss(out, gt_resampled, reduction="none").mean(dim=-1) # weighted loss for each loss term
            snr_l = snr_loss(out, gt_resampled)
            imp_snr_l = snr_l - snr_loss(audio.sum(dim=1), gt_resampled)
            si_snr_l = si_snr_loss(out, gt_resampled)
            imp_si_snr_l = si_snr_l - si_snr_loss(audio.sum(dim=1), gt_resampled)
            si_sdr_l = si_sdr_loss(out, gt_resampled, zero_mean=True, load_diag=False)
            imp_si_sdr_l = si_sdr_l - si_sdr_loss(audio.sum(dim=1), gt_resampled, zero_mean=True, load_diag=False)

            gt_numpy = gt_resampled.cpu().squeeze().numpy()
            out_numpy = out.cpu().squeeze().numpy()
            out_numpy_scaled = out_numpy
            if out.abs().max() > 1:
                out_numpy_scaled = out / out.abs().max()
                out_numpy_scaled = out_numpy_scaled.cpu().squeeze().numpy()

            # pesq
            acc_pesq.append(compute_pesq(gt_numpy, out_numpy))

            # stoi
            acc_stoi.append(compute_stoi(gt_resampled, out))

            # dnsmos
            acc_dnsmos.append(dnsmos.run(out_numpy_scaled, sr=16000)['ovrl_mos'])

            acc_mse.extend([_.mean().item() for _ in mse_l])
            acc_snr.extend([_.mean().item() for _ in snr_l])
            acc_si_snr.extend([_.mean().item() for _ in si_snr_l])
            acc_imp_snr.extend([_.mean().item() for _ in imp_snr_l])
            acc_imp_si_snr.extend([_.mean().item() for _ in imp_si_snr_l])
            acc_si_sdr.extend([_.mean().item() for _ in si_sdr_l])
            acc_imp_si_sdr.extend([_.mean().item() for _ in imp_si_sdr_l])

    print("average mse: ", np.array(acc_mse).mean(), np.array(acc_mse).std())
    print("average snr: ", np.array(acc_snr).mean(), np.array(acc_snr).std())
    print("average imp snr: ", np.array(acc_imp_snr).mean(), np.array(acc_imp_snr).std())
    print("average si snr: ", np.array(acc_si_snr).mean(), np.array(acc_si_snr).std())
    print("average imp si snr: ", np.array(acc_imp_si_snr).mean(), np.array(acc_imp_si_snr).std())
    print("average si sdr: ", np.array(acc_si_sdr).mean(), np.array(acc_si_sdr).std())
    print("average imp si sdr: ", np.array(acc_imp_si_sdr).mean(), np.array(acc_imp_si_sdr).std())

    print("avg pesq", np.array(acc_pesq).mean(), np.array(acc_pesq).std())
    print("avg stoi", np.array(acc_stoi).mean(), np.array(acc_stoi).std())
    print("avg dnsmos", np.array(acc_dnsmos).mean(), np.array(acc_dnsmos).std())
    return acc_mse, acc_snr, acc_si_snr, acc_imp_snr, acc_imp_si_snr, acc_si_sdr, acc_imp_si_sdr, acc_pesq, acc_stoi, acc_dnsmos

print(f"evaling {' '.join([k + ':' + str(args[k]) for k in args])}, pos_num: {pos_num}, pos range: {partial_range}, neg range: {neg_partial_range}, hybrid: {hybrid_num}, neglect: {neglect_require_num}")
print(f"{test_dataloader.__class__.__name__} {source_num}C{''.join([str(n) for n in active_num])}, enroll {enroll_num}, wave length {wave_length}, pos {pos_example_length}, neg {neg_example_length}")
mses, snrs, si_snrs, imp_snrs, imp_si_snrs, si_sdrs, imp_si_sdrs = [], [], [], [], [], [], []
pesqs, stois, dnsmoses, wers = [], [], [], []
for epoch in range(repeat_num):
    mse_i, snr_i, si_snr_i, imp_snr_i, imp_si_snr_i, si_sdr, imp_si_sdr, pesq_i, stoi_i, dnsmos_i = eval(model=model, dataloader=test_dataloader, epoch=epoch)
    mses.append(mse_i)
    snrs.append(snr_i)
    si_snrs.append(si_snr_i)
    imp_snrs.append(imp_snr_i)
    imp_si_snrs.append(imp_si_snr_i)
    si_sdrs.append(si_sdr)
    imp_si_sdrs.append(imp_si_sdr)
    pesqs.append(pesq_i)
    stois.append(stoi_i)
    dnsmoses.append(dnsmos_i)
    print(f"finished epoch {epoch}")
mses = np.array(mses)
snrs = np.array(snrs)
si_snrs = np.array(si_snrs)
imp_snrs = np.array(imp_snrs)
imp_si_snrs = np.array(imp_si_snrs)
si_sdrs = np.array(si_sdrs)
imp_si_sdrs = np.array(imp_si_sdrs)
pesqs = np.array(pesqs)
stois = np.array(stois)
dnsmoses = np.array(dnsmoses)
print(f"evaling {' '.join([k + ':' + str(args[k]) for k in args])}, pos_num: {pos_num}, pos range: {partial_range}, neg range: {neg_partial_range}, hybrid: {hybrid_num}, neglect: {neglect_require_num}")
print(f"{test_dataloader.__class__.__name__} {source_num}C{''.join([str(n) for n in active_num])}, enroll {enroll_num}, wave length {wave_length}, pos {pos_example_length}, neg {neg_example_length}")
print(f"mse: {mses.mean()} \t {mses.std()}")
print(f"{snrs.mean():.5f} \t {snrs.std():.5f}")
print(f"{imp_snrs.mean():.5f} \t {imp_snrs.std():.5f}")
print(f"{si_snrs.mean():.5f} \t {si_snrs.std():.5f}")
print(f"{imp_si_snrs.mean():.5f} \t {imp_si_snrs.std():.5f}")
print(f"{si_sdrs.mean():.5f} \t {si_sdrs.std():.5f}")
print(f"{imp_si_sdrs.mean():.5f} \t {imp_si_sdrs.std():.5f}")
print(f"{pesqs.mean():.5f} \t {pesqs.std():.5f}")
print(f"{stois.mean():.5f} \t {stois.std():.5f}")
print(f"{dnsmoses.mean():.5f} \t {dnsmoses.std():.5f}")
