import os
import lightning as L
import torch
import argparse
from argparse import ArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import  ModelCheckpoint,LearningRateMonitor
import torch.multiprocessing as mp
from meanse.meanflow_model import MeanFlowSEModel
from meanse.config import Config
from meanse.dataset import AudioDataModule
from meanutils.myutils import read_yaml

def config_parser():
    cfg = Config(
        model_config="none",
        learning_rate=1e-4,
        batch_size=2,
        weight_decay=1e-6,
        adam_epsilon=1e-8,
        warmup_steps=2,
        num_worker=4,
        num_train_epochs=150,
        gradient_accumulation_steps=1,
        device="cuda",
        num_gpu=1,
        train_version=0,
        train_tag="meanflow",
        train_name='ncsnpp',
        val_check_interval=50000,
        save_top_k=3,
        resume=True,
        seed=1996,
        gradient_clip=0.5,
        lr_step_size=1,
        lr_gamma=0.85,
        train_set_path='none',
        train_set_dynamic_mixing=False,
        valid_set_path='none',
        max_duration=192000,
        init_from='none',
        use_high_pass=True,
        jvp_api='funtorch',  # 'autograd' or 'funtorch'
        flow_ratio=0.5,
        max_interval=1.0,
        freeze_t_r_fuse=False,
        theta=1.5,
        sigma_max=0.5,
        sigma_min=0.05,
        ema_decay=0.999,
        t_eps=0.0,
        T_rev=1.0,
        loss_type='adaptive_l2',
        loss_abs_exponent=0.5,
        fix_data_fs=0,
    )

    parameters = vars(cfg)
    parser = ArgumentParser()

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    for par in parameters:
        default = parameters[par]
        parser.add_argument(f"--{par}", type=str2bool if isinstance(default, bool) else type(default), default=default)
    args = parser.parse_args()
    return args

def prepare_call_backs(cfg):

    best_metrics = [
        ("val_loss", "min"),
    ]
    call_backs = [LearningRateMonitor(logging_interval='epoch')]
    for i, (metric, min_or_max) in enumerate(best_metrics):
        call_back = ModelCheckpoint(
            filename="best_{epoch:02d}-{step:06d}-{"+ metric + ":.3f}",
            save_top_k=cfg.save_top_k,
            monitor=metric,
            every_n_train_steps=cfg.val_check_interval,
            mode=min_or_max,
            save_weights_only=(metric != "val_loss"),
            save_last=(metric == "val_loss"),
        )
        call_backs.append(call_back)


    return call_backs

if __name__ == "__main__":
    mp.set_start_method('spawn')
    torch.set_float32_matmul_precision('medium')
    args = config_parser()
    cfg = Config(**vars(args))
    print(cfg)
    L.seed_everything(seed=cfg.seed)

    try:
        cfg.model_config = read_yaml(cfg.model_config)
    except:
        cfg.model_config = {}
    
    model = MeanFlowSEModel(cfg=cfg)

    if cfg.init_from != 'none':
        
        state_dict = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        if (not any('t_r_fuse.linear' in key for key in state_dict)) and any('t_r_fuse.linear' in key for key in model.state_dict()):
            ndim = model.state_dict()["dnn.t_r_fuse.linear.bias"].shape[0]
            assert model.state_dict()["dnn.t_r_fuse.linear.weight"].shape == (ndim, 2 * ndim)
            print(f"init linear t_r_fuse with diag and zeros! ndim={ndim}", flush=True)
            
            diag_matrix = torch.diag(torch.ones(ndim))
            zero_matrix = torch.zeros(ndim, ndim)
            
            state_dict["dnn.t_r_fuse.linear.weight"] = torch.cat([diag_matrix, zero_matrix], dim=1)
            state_dict["dnn.t_r_fuse.linear.bias"] = torch.zeros(ndim)

        model.load_state_dict(state_dict, strict=True)

        if cfg.freeze_t_r_fuse:
            for param in model.dnn.t_r_fuse.parameters():
                param.requires_grad = False
            print("t_r_fuse is freezed!", flush=True)

        del state_dict
        print(f"Init param loaded from {cfg.init_from}")

    logger = TensorBoardLogger(save_dir=f"./exp/{cfg.train_tag}", version=cfg.train_version, name=cfg.train_name)
    call_backs = prepare_call_backs(cfg=cfg)

    last_ckpt = f"./exp/{cfg.train_tag}/{cfg.train_name}/version_{cfg.train_version}/checkpoints/last.ckpt"
    last_ckpt = last_ckpt if cfg.resume and os.path.exists(last_ckpt) else None

    trainer = L.Trainer(
        max_epochs=cfg.num_train_epochs,
        num_sanity_val_steps=0,
        accelerator=cfg.device,
        devices=cfg.num_gpu,
        gradient_clip_val=cfg.gradient_clip,
        logger=logger,
        val_check_interval=cfg.val_check_interval,
        callbacks=call_backs,
        strategy='ddp_find_unused_parameters_true',
    )

    trainer.fit(model=model, datamodule=AudioDataModule(config=cfg), ckpt_path=last_ckpt)
