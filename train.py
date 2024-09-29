# wrapper for RVT > train.py
from pathlib import Path 
import os, sys
sys.path.append(str(Path.cwd().parent)) # for pythonpath 
sys.path.append(str(Path.cwd()))
 # for pythonpath


from mxlstm.train import main as mxlstm_train 
# from RVTClass import train as rvt_train 

# RVT train.py and https://pytorch.org/docs/master/multiprocessing.html?highlight=sharing%20strategy#sharing-strategies
# and start of https://pytorch.org/docs/stable/notes/cuda.html
# see also https://github.com/pytorch/pytorch/issues/11201
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.backends import cuda, cudnn
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
from pytorch_lightning.strategies import DDPStrategy

from configs.envar import COMET_API_KEY


cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True

# import argparse
import configargparse as argparse

from omegaconf import DictConfig, OmegaConf

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument("--walk", type=str, default='random')
parser.add_argument("--preprocess", type = bool, default=True)

parser.add_argument("--model", type=str, default='mxlstmvit', choices=['mxlstmvit','rvt']) # arg not found 
parser.add_argument("--use_saved_data", type=bool, default = True) # mod for testing use saved
parser.add_argument('--config_dir', type=str, default='configs/')
parser.add_argument('--config_relpath', type=str, default='configs/mxlstm_cfg.yaml')

parser.add_argument("--trial_run", type=bool, default=True) # for code testing 

parser.add_argument("--logger", type=str, default='wandb', choices=['wandb','comet',''])
parser.add_argument("--gpu", type=int, default=0)

# RANDOM
parser.add_argument("--reload_path", type=str, default='') # if not reload empty string
parser.add_argument("--reload_hparams", type=str, default='/home/renaj/Drift/lightning_logs/version_141/hparams.yaml')
parser.add_argument("--kth_reload", type=int, default=1)

args = parser.parse_args()

# if args.model == 'mxlstmvit': # TODO abstract to train 

def main():
    if args.model == "mxlstmvit":
        config_relpath = os.path.join(args.config_dir,'mxlstm_cfg.yaml')

    # elif args.model == 'rvt':
    else:
        raise NotImplementedError
        # config_relpath = os.path.join(args.config_dir,'rvt_cfg.yaml')
        
    config = OmegaConf.load(os.path.join(os.getcwd(),config_relpath))
    # ---------------------
    # DDP
    # ---------------------
    # gpus = config.hardware.gpus
    gpus = args.gpu
    gpus = gpus if isinstance(gpus, list) else [gpus]
    distributed_backend = config.hardware.dist_backend
    assert distributed_backend in ('nccl', 'gloo'), f'{distributed_backend=}'
    strategy = DDPStrategy(process_group_backend=distributed_backend,
                           find_unused_parameters=False,
                           gradient_as_bucket_view=True) if len(gpus) > 1 else None
    
    # ---------------------
    # Logging and Checkpoints
    # ---------------------

    run_name = f"train_{args.model}_{args.walk}walk_reload{args.kth_reload}"
    if not args.trial_run:
        if args.logger == 'wandb': 
            logger = WandbLogger(project='diq',name= run_name,job_type='train', log_model='all')
            
        elif args.logger == 'comet':
            logger = CometLogger(api_key=COMET_API_KEY,project_name='driftiq',experiment_name=run_name)
         # if not testing code
        else:
            logger = None
    else:
        logger = None


    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = list()
    if config.training.lr_scheduler.use:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    callbacks.append(ModelSummary(max_depth=2))
    # see RVT > callbacks > custom > get_ckpt_callback 
    ckpt_filename = f"{run_name}_"+"epoch={epoch:03d}-step={step}"

    checkpoint_callback = ModelCheckpoint(monitor='val_accuracy',mode='max',
                                          every_n_epochs=config.logging.ckpt_every_n_epochs,
                                          filename=ckpt_filename, dirpath=f"./driftiq/tsmu{config.walk.ts.mu_init}w{config.walk.ts.sigma_init}/",
                                          save_top_k=3,save_last=True,verbose=True) # 2 if restarting with arbtrary accuracy 
    checkpoint_callback.CHECKPOINT_NAME_LAST = 'last_epoch={epoch:03d}-step={step}'
    callbacks.append(checkpoint_callback)
    early_stop_callback = EarlyStopping(monitor="val_loss")
    callbacks.append(early_stop_callback)


    if args.model == "mxlstmvit":
        mxlstm_train(config, logger, strategy, callbacks, args)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()