import sys
from attrdict import AttrDict
from utils import *
import os
args_dict = get_config(sys.argv[1])
args = AttrDict(args_dict)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import torch
from torch.utils.data import DataLoader
from dataset.LibriSpeech_single_emb import LibriDataset_single_emb
import traceback
from tqdm import tqdm
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr_loss,
    signal_noise_ratio as snr_loss)
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torch import nn
import torchaudio

def encode(frozen_encoder, s):
    with torch.no_grad():
        s = s.transpose(1, 2)
        embed_pos_a = frozen_encoder(s, None).detach()
    return embed_pos_a

def clip_gradients(model, clip):
    norms = []
    for _, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms

# Import packages
import sys,humanize,psutil,GPUtil

# Define function
def mem_report():
    print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))

    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))

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
    elif args.loss_type == "sisdr":
        loss_func = ScaleInvariantSignalDistortionRatio().to(device)

    titer = tqdm(dataloader, unit="iter")
    for i, (audio, pos_separated, neg_separated) in enumerate(titer):
        audio = audio.to(device)
        pos_separated = pos_separated.to(device)
        neg_separated = neg_separated.to(device)
        
        gt = audio[:, 0]
        pos_clean = pos_separated[:, :args.active_num[1]].sum(dim=1)

        audio = audio.sum(dim=1)
        pos = pos_separated.sum(dim=1)
        neg = neg_separated.sum(dim=1)

        # 1. extract clean and noisy enrollment embedding
        clean_emb = None
        with torch.no_grad():
            clean_emb = encode(frozen_encoder, pos_clean)

        cond_emb, pos_std, dec_audio = model.encoder_pos_neg(pos, neg, recons=False)

        # 2. extract clean enrollment embedding
        embed_l = torch.zeros((1,), device=device)
        if args.Embedding_Weight != 0:
            if args.model_name == "USEF-TFGridnet":
                if args.emb_loss_type == "mse":
                    embed_l = nn.functional.mse_loss(cond_emb, clean_emb)
                elif args.emb_loss_type == "cos":
                    embed_l = 1 - nn.functional.cosine_similarity(cond_emb, clean_emb).mean()
                else:
                    raise NotImplementedError(args.emb_loss_type)
        
        # 3. extract audio
        tmp1 = torch.zeros((1,), device=device)
        snr_l = torch.zeros((1,), device=device)
        if args.SNR_Weight != 0:
            if args.model_name == "USEF-TFGridnet":
                out = model(audio, cond_emb)

            if args.normalize:
                out = out / out.abs().max(dim=-1, keepdim=True)[0] * gt.abs().max(dim=-1, keepdim=True)[0]

            snr_l = -loss_func(out.squeeze(), gt.squeeze()).mean()
        
        l = args.SNR_Weight * snr_l + args.DEC_Weight * tmp1 + args.Embedding_Weight * embed_l

        if l.isnan():
            breakpoint()

        losses = [snr_l, embed_l, tmp1]

        if train:
            titer.set_description(f"train iter {i}")
            titer.set_postfix(snr=snr_l.item(),
                              embed_l=embed_l.item(),
                              )
            optimizer.zero_grad()
            l.backward()
            if args.gradient_clip_val > 0:
                clip_gradients(model, args.gradient_clip_val)
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

    # ======== lookonce encoder ==========
    encoder = None
    encoder_head = None    
    from model.tfgridnet_encoder import TFGridNet_encoder
    encoder = TFGridNet_encoder(
            num_ch=2,
            n_fft=128,
            stride=64,
            num_blocks=args.enc_num_block,
            binaural=args.binaural,
        )        

    from improved_model.GridnetAttnHead import GridNetBlock_attnhead
    encoder_head = GridNetBlock_attnhead(
        layer_num=args.layer_num,
        pooling_size=1,
        stride=1,
        return_clean_dvec=args.return_clean_dvec,
        out_dim=0,
        refine_layer_num=args.refine_layer_num,
        fusion_shortcut=args.fusion_shortcut,
        cut_pos=args.cut_pos,
    )
    
    # ======== model ========
    if args.model_name == "USEF-TFGridnet":
        from improved_model.USEF_TFGridnet import Tar_Model
        model = Tar_Model(
            n_freqs=65,
            hidden_channels=args.hidden_channels,
            n_head=4,
            emb_dim=args.emb_dim,
            emb_ks=1,
            emb_hs=1,
            num_layers=args.sep_layer_num,
            encoder=encoder,
            encoder_head=encoder_head,
            train_encoder=(args.encoder_lr != 0),
            train_encoder_head=(args.encoder_lr != 0),            
            binaural=args.binaural,
        ) 
        model.to(device)

        # load trained encoder
        if args.load_encoder != "":
            state_dict = torch.load(args.load_encoder, weights_only=False)
            model.siamese.load_state_dict(state_dict["siamese"], strict=False)
            model.encoder_head.load_state_dict(state_dict["encoder_head"], strict=True)

    if args.load_model != "":
        model.load_state_dict(torch.load(args.load_model, weights_only=False)["state_dict"], strict=True)

    # ======== frozen lookoncetohear encoder to distill againest ==========
    from model.tfgridnet_encoder import TFGridNet_encoder
    frozen_encoder_param = os.path.join("model", 'best.ckpt')
    frozen_encoder_param = torch.load(frozen_encoder_param, map_location='cpu', weights_only=False)
    state_dict = dict([(k[6:], frozen_encoder_param['state_dict'][k]) for k in frozen_encoder_param['state_dict']])
    
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

    # ======== optizer, scheduler =========
    param_list = []
    if args.model_name == "USEF-TFGridnet":
        param_list.append(dict(name='encoder', params=model.encoder_params(), lr=args.encoder_lr))
        param_list.append(dict(name='main', params=model.main_params(), lr=args.main_lr))

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            param_list,
            lr=args.main_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay,
            amsgrad=True)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            param_list,
            lr=args.main_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay,
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
    elif args.lr_schedule == "exp":
        assert args.main_lr != 0, "assume the main lr is the start lr"
        gamma = (args.min_lr / args.main_lr) ** (1 / args.epoch_num)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        raise NotImplementedError(args.lr_schedule)

    # ======== train =========
    # send_email(f"Started experiment {process_id}", "experiment args " + str(args))

    # train
    print("param size:", sum(p.numel() for p in model.parameters() if p.requires_grad))
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

        max_lr = max(pg['lr'] for pg in optimizer.param_groups)

        log.write(f"Epoch: {epoch}, train_losses: {', '.join(train_losses)}, " + 
                  f"val_losses: {', '.join(val_losses)}, lr:{max_lr}\n")
        log.flush()
        print(f"Finish {epoch} / {args.epoch_num}, id {process_id}")

        if epoch % args.save_epoch == 0:
            if args.SNR_Weight == 0: # stage 1
                torch.save(model.encoder_state_dict(), f"output/encoder_{process_id}_{epoch}.pt")
            else:
                torch.save({"state_dict": model.state_dict()}, f"output/main_branch_{process_id}_{epoch}.pt")
    if args.SNR_Weight == 0: # stage 1
        torch.save(model.encoder_state_dict(), f"output/encoder_{process_id}.pt")
    else:
        torch.save({"state_dict": model.state_dict()}, f"output/main_branch_{process_id}.pt")
    log.close()

    # send_email(f"Finished experiment {process_id}", "experiment args " + str(args))

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
        # send_email(f"Experiment {process_id} failed", traceback.format_exc())
