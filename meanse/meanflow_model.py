import lightning as L
from torch.optim.optimizer import Optimizer
import torch
import warnings
from meanse.ncsnpp import NCSNpp
from meanse.config import Config 
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from torch_ema import ExponentialMovingAverage
from meanse.odes import FLOWMATCHING
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder
from functools import partial
import numpy as np
from meanutils.myutils import read_yaml
from einops import rearrange
from meanutils.deal_with_checkpoints import load_state_dict_from_ckpt, compare_state_dicts
import os

def adaptive_l2_loss(error, gamma=0, c=1e-3):
    error = torch.abs(error)
    delta_sq = torch.mean(error ** 2, dim=tuple(range(1, error.ndim)), keepdim=False) # dim=(1, 2)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return w.detach() * loss

def sample_bool(flow_ratio):
    return np.random.rand() < flow_ratio

class MeanFlowSEModel(L.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()

        print("### MeanFlowSEModel init ###")
        print("init_from:", cfg.init_from, flush=True)
        print("loss_type:", cfg.loss_type, flush=True)
        print("flow_ratio:", cfg.flow_ratio, flush=True)
        print("max_interval:", getattr(cfg, "max_interval", "1.0"), flush=True)

        model_cfg = getattr(cfg, "model_config", {})
        if isinstance(model_cfg, str):
            model_cfg = read_yaml(model_cfg)
        print("model_config:", model_cfg, flush=True)
        model_name = model_cfg.get("model_name", "ncsnpp")

        self.sisnr_loss = SISNRLoss()

        self.save_hyperparameters()
        self.cfg = cfg
        self.ode = FLOWMATCHING(**model_cfg.get("ode_config", {}))
        self.encoder = STFTEncoder(**model_cfg.get("encoder_config", {}))
        self.decoder = STFTDecoder(**model_cfg.get("decoder_config", {}))
        
        if model_name == "ncsnpp":
            self.dnn = NCSNpp(**model_cfg.get("model_config", {}))
        else:
            raise NotImplementedError(f"unknown model name: {model_name}")
        
        if cfg.jvp_api == "funtorch":
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif cfg.jvp_api == "autograd":
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

        self.lr = cfg.learning_rate
        self.ema_decay = cfg.ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = cfg.t_eps
        self.T_rev = cfg.T_rev
        self.ode.T_rev = cfg.T_rev
        self.loss_type = cfg.loss_type
        self.num_eval_files = 3
        self.loss_abs_exponent = cfg.loss_abs_exponent
        self.flow_ratio = cfg.flow_ratio
        self.max_interval = getattr(cfg, "max_interval", 1.0)

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        return

    def on_after_backward(self):
        return
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):

        # check if has grad of NaN
        grad_has_nan = any(
            torch.isnan(p.grad).any() 
            for p in self.parameters() 
            if p.grad is not None
        )
        if grad_has_nan:
            rank = torch.distributed.get_rank()
            print(f'RANK {rank}: NaN in grad has been decected, reset grad to zero')
            optimizer.zero_grad()
            
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res
    
    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    def _mse_loss(self, x, x_hat):    
        err = x - x_hat
        losses = torch.square(err.abs())
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss
    
    def _loss(self, error):    
        if self.loss_type == 'mse':
            losses = torch.square(error.abs())
        elif self.loss_type == 'mae':
            losses = error.abs()
        elif self.loss_type == 'adaptive_l2':
            losses = adaptive_l2_loss(error)
        else:
            raise ModuleNotFoundError("unknown loss type!")
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss
    
    def speech_to_feature(self, speech, fs, speech_length):

        feature, f_lens = self.encoder(speech, speech_length, fs=fs) # B, T, F
        feature = feature.permute(0, 2, 1).unsqueeze(1)
        return feature
    
    def feature_to_speech(self, feature, fs, speech_length):
        
        feature = feature.squeeze(1).permute(0, 2, 1)
        speech, _=  self.decoder(feature, speech_length, fs)
        return speech
    
    def on_train_start(self):
        if self.cfg.init_from is not None and os.path.isfile(self.cfg.init_from):
            print(f"Checking the loaded state dict from {self.cfg.init_from}", flush=True)
            state_dict = load_state_dict_from_ckpt(self.cfg.init_from, self.device)
            compare_state_dicts(self.state_dict(), state_dict, visible=False)
        return

    def forward_step(self, batch):

        clean_speech, noisy_speech, fs, speech_length = batch
        B, C, T = clean_speech.shape
        assert C == 1
        
        clean_speech = clean_speech.view(B, T).float()
        noisy_speech = noisy_speech.view(B, T).float()
        clean_speech = torch.nan_to_num(clean_speech, nan=0)
        noisy_speech = torch.nan_to_num(noisy_speech, nan=0)
        
        x0 = self.speech_to_feature(clean_speech, fs, speech_length) # B, 1, F, T
        y = self.speech_to_feature(noisy_speech, fs, speech_length) # B, 1, F, T

        # sample t and r (r <= t)
        r, t, train_flow = self.sample_r_t(x0.shape[0], self.T_rev, self.t_eps, self.max_interval, x0.device)
        assert (r <= t).all()
        t_ = rearrange(t, "b -> b 1 1 1") # t_ = t.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        r_ = rearrange(r, "b -> b 1 1 1") # r_ = r.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        e, _ = self.ode.prior_sampling(x0.shape, y)
        z = (1 - t_) * x0 + t_ * e # B, 1, F, T
        v = e - x0
        v_hat = v

        model_partial = partial(self.forward, y=y) # self.dnn
        
        jvp_args = (
            lambda xt_f, r_f, t_f: model_partial(x=xt_f, r=r_f, t=t_f), # xt=xt_f
            (z, r, t),
            (v_hat, torch.zeros_like(r), torch.ones_like(t)),
        )
        
        # with torch.backends.cudnn.flags(enabled=False):
        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)
            
        u_tgt = v_hat - (t_ - r_) * dudt
        err = u - u_tgt.detach() # B, 1, F, T, complex
        loss = self._loss(err)

        return loss, train_flow

    def sample_r_t(self, batch_size, T_rev, t_eps, max_interval, device):

        assert max_interval <= 1.0, "max_interval should be <= 1.0"
        samples = np.random.rand(batch_size, 2).astype(np.float32)

        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        train_flow = sample_bool(self.flow_ratio)
        if train_flow:
            r_np = t_np
        else:
            interval = t_np - r_np
            t_np = r_np + interval * max_interval

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return r, t, train_flow

    def enhance(self, y, fs, speech_length, N=15):
        # y: B, T
        Y = self.speech_to_feature(y, fs, speech_length)
        x0, _ = self.ode.prior_sampling(Y.shape, Y)
        fn = partial(self.dnn, y=Y)
        steps = torch.linspace(self.T_rev, self.t_eps, N + 1).to(device=Y.device)

        for i, t in enumerate(steps[:-1]):
            stepsize = t - steps[i + 1]
            t = torch.ones(Y.shape[0], device=Y.device) * t

            next_t = torch.ones(Y.shape[0], device=Y.device) * steps[i + 1]

            u_flow = fn(r=next_t, t=t, xt=x0)
            dt = stepsize
            x0 = x0 + dt * u_flow

        enhanced = self.feature_to_speech(x0, fs, speech_length)
        return enhanced

    def forward(self, x, r, t, y):
        score = -self.dnn(xt=x, y=y, r=r, t=t)
        return score

    def training_step(self, batch, batch_idx):

        loss, train_flow = self.forward_step(batch)
        self.log('train_loss', loss, on_step=True, prog_bar=True, on_epoch=True, sync_dist=True)
        if train_flow:
            self.log('train_flow_loss', loss, on_step=True, prog_bar=True, on_epoch=True, sync_dist=True)
        else:
            self.log('train_meanflow_loss', loss, on_step=True, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):

        loss, val_flow = self.forward_step(batch)
        self.log('val_loss', loss, on_step=False, prog_bar=True, on_epoch=True, sync_dist=True)
        if val_flow:
            self.log('val_flow_loss', loss, on_step=False, prog_bar=True, on_epoch=True, sync_dist=True)
        else:
            self.log('val_meanflow_loss', loss, on_step=False, prog_bar=True, on_epoch=True, sync_dist=True)

        if batch_idx % 100 == 0:
            clean_speech, noisy_speech, fs, speech_length = batch
            B, C, T = clean_speech.shape
            assert C == 1
            clean_speech = clean_speech.view(B, T).float()
            noisy_speech = noisy_speech.view(B, T).float()

            predicted = self.enhance(noisy_speech, fs, speech_length, N=10)
            sisnr_loss = self.sisnr_loss(clean_speech, predicted).mean()
            self.log(f'sisnr', - sisnr_loss.detach().item(), on_step=True, prog_bar=True, batch_size=B, sync_dist=True)

        return {'val_loss': loss.detach()}
    
    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            eps=self.cfg.adam_epsilon,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.cfg.lr_step_size, gamma=self.cfg.lr_gamma)

        return [optimizer], [scheduler]

if __name__ == "__main__":
    pass