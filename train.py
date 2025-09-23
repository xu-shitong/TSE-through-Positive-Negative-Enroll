import sys
from attrdict import AttrDict
from utils import *
import os
args_dict = get_config(sys.argv[1])
args = AttrDict(args_dict)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import torch
from torch.utils.data import DataLoader
import traceback
from tqdm import tqdm
from dataset.LibriSpeech_single_emb import LibriDataset_single_emb
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr_loss,
    signal_noise_ratio as snr_loss)
from torch import nn

# Import packages
import sys,humanize,psutil,GPUtil

# Define function
def mem_report():
    print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))

    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))

def encode(frozen_encoder, s):
    with torch.no_grad():
        s = s.transpose(1, 2)
        embed_pos_a = frozen_encoder(s, None).detach()
    return embed_pos_a

def normalize_samples(audio):
    '''shape: [B, audio_num, 2, audio_length]'''
    norm_factor = torch.abs(audio.sum(dim=1)).max()
    if norm_factor > 1.0:
        audio = audio / norm_factor
    return audio 

def train_func(args, epoch, model, frozen_encoder, dataloader, optimizer, device, train):

    if train:
        model.to_train()
    else:
        model.eval()

    acc_losses = [0] * 4
    if args.loss_type == "snr":
        loss_func = snr_loss
    elif args.loss_type == "sisnr":
        loss_func = si_snr_loss

    titer = tqdm(dataloader, unit="iter")
    for i, (audio, pos_separated, neg_separated) in enumerate(titer):
        audio = audio.to(device)
        pos_separated = pos_separated.to(device)
        neg_separated = neg_separated.to(device)

        gt = audio[:, 0]

        pos_clean = pos_separated[:, :args.active_num[1]].sum(dim=1)

        clean_emb = None
        if args.Embedding_Weight != 0 and not args.return_clean_dvec:
            clean_emb = encode(frozen_encoder, pos_clean)


        pos = pos_separated.sum(dim=1)
        neg = neg_separated.sum(dim=1)

        audio = audio.sum(dim=1)

        # 1. split audio into segments
        audio_segs = torch.split(audio, args.sample_rate, dim=-1)

        # # 2. extract pos and neg enroll
        cond_emb = model.encode(pos, neg)
        
        embed_l = torch.zeros((1,), device=device)
        if args.Embedding_Weight != 0:
            embed_l = nn.functional.mse_loss(cond_emb, clean_emb)
        
        tmp1 = torch.zeros((1,), device=device)
        snr_l = torch.zeros((1,), device=device)
        if args.SNR_Weight != 0:
            init_state = model.init_buffers(audio.shape[0], device)

            # 4. inference
            out = []
            tmp1 = torch.zeros((1,), device=device)
            for audio_seg in audio_segs:
                out_seg, init_state = model(audio_seg, cond_emb, init_state)
                out.append(out_seg)

            out = torch.concat(out, dim=-1)

            snr_l = -loss_func(out, gt).mean()
        
        l = args.SNR_Weight * snr_l + args.Embedding_Weight * embed_l

        losses = [snr_l, embed_l, tmp1]

        if train:
            titer.set_description(f"train iter {i}")
            titer.set_postfix(snr=snr_l.item(),
                              embed_l=embed_l.item(),
                              )
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        else:
            titer.set_description(f"val iter {i}")
            titer.set_postfix(snr=snr_l.item(),
                              embed_l=embed_l.item(),
                              )

        acc_losses[1:] = [acc_l + new_l.item() for acc_l, new_l in zip(acc_losses[1:], losses)]
        acc_losses[0] += l.item()
        
    return [l / len(dataloader) for l in acc_losses]

def main_func(log, args):

    mem_report()

    torch.cuda.manual_seed(42)
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print("using device: ", dev)

    # ======== dataset ========

    if "LibriSpeech" in args.train_dataset_dir:
        train_dataset = LibriDataset_single_emb(
            args.train_dataset_dir, sample_rate=args.sample_rate, wave_length=args.wave_length, pos_example_length=args.pos_example_length, neg_example_length=args.neg_example_length,
            snr_db_range=args.snr_db_range, source_num=args.source_num, min_source_num=args.min_source_num, active_num=args.active_num, normalize=args.normalize, reproducable=args.reproducable,
            return_dvec=False, perturb_speeds=args.perturb_speeds, filling_pattern=args.filling_pattern, dvec_rate=args.dvec_rate, tgt_intensity=args.tgt_snr,
            reverb=args.reverb, binaural=args.binaural, reverb_cond=args.reverb_cond, zero_in_tgt=args.zero_in_tgt, noise_dir=args.noise_dir + "tr/", special_spk=args.special_spk, partial_range=args.PI_range, neg_partial_range=args.NI_range,
            same_disturb=args.same_disturb, zero_degree_pos=args.zero_degree_pos, return_clean_dvec=args.return_clean_dvec, brir_dir=args.brir_dir)
        val_dataset = LibriDataset_single_emb(
            args.val_dataset_dir, sample_rate=args.sample_rate, wave_length=args.wave_length, pos_example_length=args.pos_example_length, neg_example_length=args.neg_example_length,
            snr_db_range=args.snr_db_range, source_num=args.source_num, min_source_num=args.min_source_num, active_num=args.active_num, normalize=args.normalize, filling_pattern=args.filling_pattern, tgt_intensity=args.tgt_snr,
            return_dvec=False, dvec_rate=args.dvec_rate, reverb=args.reverb, binaural=args.binaural, reverb_cond=args.reverb_cond, partial_range=args.PI_range, neg_partial_range=args.NI_range,
            zero_in_tgt=args.zero_in_tgt, noise_dir=args.noise_dir + "cv/", special_spk=args.special_spk,
            same_disturb=args.same_disturb, zero_degree_pos=args.zero_degree_pos, return_clean_dvec=args.return_clean_dvec, brir_dir=args.brir_dir)
    else:
        raise NotImplementedError(args.train_dataset_dir)
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=10)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=10)

    # ======== encoder ========
    if "tfgridnet" in args.fusion_name:
        from model.tfgridnet_encoder import TFGridNet_encoder
        encoder = TFGridNet_encoder(
                num_ch=2,
                n_fft=128,
                stride=64,
                num_blocks=3,
                binaural=args.binaural,
            )
        frozen_encoder_param = os.path.join("model", 'best.ckpt')
        frozen_encoder_param = torch.load(frozen_encoder_param, map_location='cpu')
        state_dict = dict([(k[6:], frozen_encoder_param['state_dict'][k]) for k in frozen_encoder_param['state_dict']])
        
        if args.Embedding_Weight != 0:
            frozen_encoder = TFGridNet_encoder(
                num_ch=2,
                n_fft=128,
                stride=64,
                num_blocks=3,
                binaural=args.binaural,
            )
            frozen_encoder.load_state_dict(state_dict, strict=False)
            frozen_encoder.to(device)
            frozen_encoder.eval()
        else:
            frozen_encoder = None

        if args.load_encoder != "":
            encoder.load_state_dict(torch.load(args.load_encoder)["state_dict"], strict=True)
    else:
        raise NotImplementedError(args.fusion_name)
    
    # ======== encoder head ==========

    if "tfgridnet" in args.fusion_name:
        from model.GridnetAttnHead import GridNetBlock_attnhead
        encoder_head = GridNetBlock_attnhead(
            layer_num=args.layer_num,
            pooling_size=1,
            stride=1,
            return_clean_dvec=args.return_clean_dvec,
            out_dim=args.out_dim,
        )
    else:
        raise NotImplementedError("encoder_head")
    
    if args.load_encoder_head != "":
        encoder_head.load_state_dict(torch.load(args.load_encoder_head)["state_dict"], strict=True)
    
    # ======== model ========
    if args.model_name == "tfgridnet_causal":
        from model.tfgridnet_KVfusion import TFGridNet_KVfusion
        from model.tfgridnet_crossattn_causal_single_emb import TFGridNet_origcrossattn_causal_single_emb
        model = TFGridNet_origcrossattn_causal_single_emb(
            n_fft=args.n_fft,
            stride=args.stride,
            n_layers=args.n_layers,
            lstm_hidden_units=args.lstm_hidden_units,
            emb_dim=args.emb_dim,
            emb_ks=args.emb_ks,
            model_normalize = args.model_normalize,
            Fusion_class=TFGridNet_KVfusion,
            pooling_size=args.pooling_size,
            fusion_stride=args.fusion_stride,
            encoder=encoder,
            encoder_head=encoder_head,
            train_encoder=(args.encoder_lr != 0),
            train_encoder_head=(args.head_lr != 0),
            fusion_layer=args.fusion_layer,
            binaural=args.binaural,
        )
        if args.n_layers == 0 and args.main_lr != 0:
            raise ValueError("no extraction branch, and not in stage 1 training")
        model.to(device)
    else:
        raise NotImplementedError(args.model_name)
    
    if args.load_model != "":
        model.load_state_dict(torch.load(args.load_model)["state_dict"], strict=True)

    # ======== optizer, scheduler =========
    param_list = []
    if args.encoder_lr != 0:
        param_list.append(dict(name='encoder', params=model.get_encoder_params(), lr=args.encoder_lr))
    if args.head_lr != 0:
        param_list.append(dict(name='head', params=model.get_encoder_head_params(), lr=args.head_lr))
    if args.main_lr != 0:
        param_list.append(dict(name='main', params=model.get_main_params(), lr=args.main_lr))
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            param_list,
            lr=args.main_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0001,
            amsgrad=True)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            param_list,
            lr=args.main_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0001,
            amsgrad=True)
    else:
        raise NotImplementedError(args.optimizer)
    
    if args.lr_schedule == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=args.mode,
            patience=args.patience,
            factor=args.factor,
            min_lr=args.min_lr
        )
    elif args.lr_schedule == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.lr_decay_epoch,
                                                    gamma=args.lr_decay_gamma)
    else:
        raise NotImplementedError(args.lr_schedule)

    # ======== train =========
    print("param size:", sum(p.numel() for p in model.parameters()))
    for epoch in range(args.epoch_num):
        train_losses = train_func(args, epoch, model, frozen_encoder, train_dataloader, optimizer, device=device, train=True)
        train_losses = [str(num) for num in train_losses]

        with torch.no_grad():
            val_losses = train_func(args, epoch, model, frozen_encoder, val_dataloader, optimizer=None, device=device, train=False)
            val_losses = [str(num) for num in val_losses]

        if args.lr_schedule == 'plateau':
            lr_scheduler.step(float(val_losses[0]))
        else:
            lr_scheduler.step()

        log.write(f"Epoch: {epoch}, train_losses: {', '.join(train_losses)}, " + 
                  f"val_losses: {', '.join(val_losses)}, lr:{lr_scheduler.get_last_lr()[0] if args.lr_schedule != 'plateau' else lr_scheduler._last_lr[0]}\n")
        log.flush()
        print(f"Finish {epoch} / {args.epoch_num}, id {process_id}")

        if epoch % args.save_epoch == 0:
            if args.main_lr == 0: # pretraining
                torch.save({"state_dict": model.encoder.state_dict()}, f"output/encoder_{process_id}_{epoch}.pt")
                torch.save({"state_dict": model.encoder_head.state_dict()}, f"output/encoder_head_{process_id}_{epoch}.pt")
            else:
                torch.save({"state_dict": model.state_dict()}, f"output/main_branch_{process_id}_{epoch}.pt")
    if args.main_lr == 0: # pretraining
        torch.save({"state_dict": model.encoder.state_dict()}, f"output/encoder_{process_id}.pt")
        torch.save({"state_dict": model.encoder_head.state_dict()}, f"output/encoder_head_{process_id}.pt")
    else:
        torch.save({"state_dict": model.state_dict()}, f"output/main_branch_{process_id}.pt")
    log.close()

    mem_report()
    print("Finish experiment", process_id)


if __name__ == "__main__":
    process_id = os.getpid()
    try:
        print("Start experiment", process_id)
        log = open(f"output/{process_id}.txt", "a")
        log.write("\n".join([str(key) + " " + str(args.get(key)) for key in args.keys()]) + "\n") # write hyperparameter in log file
        log.flush()

        args_dict = get_config(sys.argv[1])
        args = AttrDict(args_dict)

        main_func(log, args)
    except KeyboardInterrupt:
        log.close()
    except Exception:
        print("training failed", traceback.format_exc())
