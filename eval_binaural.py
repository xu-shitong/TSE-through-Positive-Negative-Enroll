'''
Code for the binaural models' evaluation
The inference code for the LookOnceToHear model include code from the LookOnceToHear repository: https://github.com/vb000/LookOnceToHear/tree/main
'''
from utils import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import traceback
from tqdm import tqdm
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr_loss,
    signal_noise_ratio as snr_loss)
import numpy as np
from torch import nn
from dataset.LibriSpeech_single_emb import LibriDataset_single_emb

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

test_data_dir = "../data/LibriSpeech/LibriSpeech/_test_data/"

source_num = 3 # number of speakers in the Audio Mixture (including the target speaker)
enroll_num = 3 # number of speakers in the Enrollments (including the target speaker)
pos_example_length = 48000 # Positive Enrollment length, in waveform length (pos_example_length / 16000 is the length in second)
neg_example_length = 48000 # Negative Enrollment length, in waveform length (neg_example_length / 16000 is the length in second)
brir_dir = [
    "../data/MixLibriSpeech/CIPIC", 
    "../data/RRBRIR", 
    "../data/ASH-Listening-Set-8.0/BRIRs", 
    "../data/CATT_RIRs/Binaural/16k"]
noise_dir = "../data/wham_noise/"

# ======== model =========
args = dict(
    model_name = "tfgridnet_causal",
    model_path = "output/proposed-binaural.pt",
    fusion_name = "tfgridnet_kv",
    model_normalize = True,
    pooling_size=20,
    fusion_stride=20,
    head_layer_num=2,
    fusion_layer=[0,1],
)

# ========= default hyperparameters ==============

repeat_num = 100
set_size = 50
active_num = [-1, 1, -1] # have to be this, otherwise rir convolve is wrong
wave_length = 6 # 1 fixed to 1 for now, decide later how much audio waveform should be left after fftconvolve
filling_pattern = "repeat"
dvec_rate = 50
reverb_cond = True
snr_db_range = [-2.5, 2.5]
zero_degree_pos = True
special_spk = ["Partial_Pos", "Partial_Neg"]
PI_range = [0.33, 0.66]
NI_range = [0.33, 1.0]

# ======== model ========
if args["model_name"] == "tfgridnet_causal":
    from model.tfgridnet_encoder import TFGridNet_encoder
    encoder = TFGridNet_encoder(
            num_ch=2,
            n_fft=128,
            stride=64,
            num_blocks=3,
            binaural=True,
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
        binaural=True,
    )
    model.to(device)
    model.load_state_dict(torch.load(
        args["model_path"],
        map_location=torch.device('cpu'))["state_dict"],
        strict=True)
    sample_rate = 16000

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

test_dataset = LibriDataset_single_emb(
    f"data/LibriSpeech/LibriSpeech/_test_data", sample_rate=sample_rate, wave_length=wave_length*sample_rate, pos_example_length=pos_example_length, neg_example_length=neg_example_length,
    snr_db_range=snr_db_range, source_num=source_num, min_source_num=source_num, enroll_num=enroll_num, min_enroll_num=enroll_num, active_num=active_num, 
    reproducable=True, normalize=False, filling_pattern=filling_pattern,
    return_dvec=("tfgridnet" in args["model_name"] and "dvec" in args["fusion_name"]), dvec_rate=dvec_rate, include_silent=False, special_spk=special_spk, partial_range=PI_range, neg_partial_range=NI_range,
    reverb="all", brir_dir=brir_dir, binaural=True, reverb_cond=reverb_cond, noise_dir=noise_dir + "tt/", zero_degree_pos=zero_degree_pos,
    same_disturb=False)

test_dataloader = test_dataset

def eval(model, dataloader, epoch):

    acc_mse = []
    acc_snr = []
    acc_si_snr = []
    acc_imp_snr = []
    acc_imp_si_snr = []

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

            cond_emb = model.encode(pos.sum(dim=1), neg.sum(dim=1))
            init_state = model.init_buffers(audio.shape[0], device)

            # inference
            out, init_state = model(audio.sum(dim=1), cond_emb, init_state)

            out = torch.concat([out, -out], dim=0)
            idxs = torch.min((out - gt_resampled).pow(2).sum(dim=-1), dim=0)[1]
            idxs = idxs[None, :, None].repeat((1, 1, out.shape[-1]))

            out = torch.gather(out, 0, idxs) 
            mse_l = torch.nn.functional.mse_loss(out, gt_resampled, reduction="none").mean(dim=-1)
            snr_l = snr_loss(out, gt_resampled)
            imp_snr_l = snr_l - snr_loss(audio.sum(dim=1), gt_resampled)
            si_snr_l = si_snr_loss(out, gt_resampled)
            imp_si_snr_l = si_snr_l - si_snr_loss(audio.sum(dim=1), gt_resampled)

            gt_numpy = gt_resampled.cpu().squeeze().numpy()
            out_numpy = out.cpu().squeeze().numpy()
            out_numpy_scaled = out_numpy
            if out.abs().max() > 1:
                out_numpy_scaled = out / out.abs().max()
                out_numpy_scaled = out_numpy_scaled.cpu().squeeze().numpy()

            # pesq
            acc_pesq.append(compute_pesq(gt_numpy[0], out_numpy[0], sample_rate=16000))
            acc_pesq.append(compute_pesq(gt_numpy[1], out_numpy[1], sample_rate=16000))

            # stoi
            acc_stoi.append(compute_stoi(gt_resampled[:, 0], out[:, 0]))
            acc_stoi.append(compute_stoi(gt_resampled[:, 1], out[:, 1]))

            # dnsmos
            acc_dnsmos.append(dnsmos.run(out_numpy_scaled[0], sr=16000)['ovrl_mos'])
            acc_dnsmos.append(dnsmos.run(out_numpy_scaled[1], sr=16000)['ovrl_mos'])

            acc_mse.extend([_.mean().item() for _ in mse_l])
            acc_snr.extend([_.mean().item() for _ in snr_l])
            acc_si_snr.extend([_.mean().item() for _ in si_snr_l])
            acc_imp_snr.extend([_.mean().item() for _ in imp_snr_l])
            acc_imp_si_snr.extend([_.mean().item() for _ in imp_si_snr_l])
    print("average mse: ", np.array(acc_mse).mean(), np.array(acc_mse).std())
    print("average snr: ", np.array(acc_snr).mean(), np.array(acc_snr).std())
    print("average imp snr: ", np.array(acc_imp_snr).mean(), np.array(acc_imp_snr).std())
    print("average si snr: ", np.array(acc_si_snr).mean(), np.array(acc_si_snr).std())
    print("average imp si snr: ", np.array(acc_imp_si_snr).mean(), np.array(acc_imp_si_snr).std())

    print("avg pesq", np.array(acc_pesq).mean(), np.array(acc_pesq).std())
    print("avg stoi", np.array(acc_stoi).mean(), np.array(acc_stoi).std())
    print("avg dnsmos", np.array(acc_dnsmos).mean(), np.array(acc_dnsmos).std())
    return acc_mse, acc_snr, acc_si_snr, acc_imp_snr, acc_imp_si_snr, acc_pesq, acc_stoi, acc_dnsmos

print(f"evaling {' '.join([k + ':' + str(args[k]) for k in args])}, zero degree pos: {zero_degree_pos}, special spk: {special_spk}, PI range: {PI_range}, NI range: {NI_range}")
print(f"{test_dataloader.__class__.__name__} {source_num}C{''.join([str(n) for n in active_num])}, enroll num: {enroll_num}, wave length {wave_length}, pos {pos_example_length}, neg {neg_example_length}")
mses, snrs, si_snrs, imp_snrs, imp_si_snrs = [], [], [], [], []
pesqs, stois, dnsmoses, wers = [], [], [], []
for epoch in range(repeat_num):
    mse_i, snr_i, si_snr_i, imp_snr_i, imp_si_snr_i, pesq_i, stoi_i, dnsmos_i = eval(model=model, dataloader=test_dataloader, epoch=epoch)
    mses.append(mse_i)
    snrs.append(snr_i)
    si_snrs.append(si_snr_i)
    imp_snrs.append(imp_snr_i)
    imp_si_snrs.append(imp_si_snr_i)
    pesqs.append(pesq_i)
    stois.append(stoi_i)
    dnsmoses.append(dnsmos_i)
    print(f"finished epoch {epoch}")
mses = np.array(mses)
snrs = np.array(snrs)
si_snrs = np.array(si_snrs)
imp_snrs = np.array(imp_snrs)
imp_si_snrs = np.array(imp_si_snrs)
pesqs = np.array(pesqs)
stois = np.array(stois)
dnsmoses = np.array(dnsmoses)
print(f"evaling {' '.join([k + ':' + str(args[k]) for k in args])}, zero degree pos: {zero_degree_pos}, special spk: {special_spk}, PI range: {PI_range}, NI range: {NI_range}")
print(f"{test_dataloader.__class__.__name__} {source_num}C{''.join([str(n) for n in active_num])}, enroll num: {enroll_num}, wave length {wave_length}, pos {pos_example_length}, neg {neg_example_length}")
print(f"mse: {mses.mean()} \t {mses.std()}")
print(f"{snrs.mean():.5f} \t {snrs.std():.5f}")
print(f"{imp_snrs.mean():.5f} \t {imp_snrs.std():.5f}")
print(f"{si_snrs.mean():.5f} \t {si_snrs.std():.5f}")
print(f"{imp_si_snrs.mean():.5f} \t {imp_si_snrs.std():.5f}")
print(f"{pesqs.mean():.5f} \t {pesqs.std():.5f}")
print(f"{stois.mean():.5f} \t {stois.std():.5f}")
print(f"{dnsmoses.mean():.5f} \t {dnsmoses.std():.5f}")
