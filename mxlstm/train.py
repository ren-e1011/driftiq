# wrapper for RVT > train.py
from pathlib import Path 
import os, sys
sys.path.append(str(Path.cwd().parent)) # for pythonpath 
sys.path.append(str(Path.cwd()))
 # for pythonpath
import yaml
import pickle

import numpy as np

# RVT train.py and https://pytorch.org/docs/master/multiprocessing.html?highlight=sharing%20strategy#sharing-strategies
# and start of https://pytorch.org/docs/stable/notes/cuda.html
# see also https://github.com/pytorch/pytorch/issues/11201
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True

# import argparse
import configargparse as argparse

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from pytorch_lightning import Trainer


from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


from Data.dataset import DataSet, Collator


from mxlstm.litmod import MxLSTMClassifier


from omegaconf import DictConfig, OmegaConf

from pytorch_lightning.loggers import WandbLogger


def main(config:DictConfig, logger, strategy, callbacks, args):

    # ---------------------
    # Model
    # ---------------------
    module = MxLSTMClassifier(config) 

    if not args.trial_run and args.logger == 'wandb':
        logger.watch(model=module, log='all', log_freq=config.logging.train.log_model_every_n_steps, log_graph=True)
    
    # ---------------------
    # Dataset 
    # ---------------------
    dataset = DataSet(walk = args.walk, 
                      architecture= config.model.name,
                      steps = config.time.steps, bins=config.time.bins, 
                      refrac_pd=config.emulator.refrac_pd, threshold= config.emulator.threshold,
                      use_saved_data= args.use_saved_data,
                      frame_hw = (config.input.height,config.input.width), fps=config.time.fps, preproc_data=args.preprocess, 
                      ts_mu = config.walk.ts.mu_init, ts_s = config.walk.ts.sigma_init,
                      test=False)
    # snippet from https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    
    try:
        with open('./Data/train_indices.pkl','rb') as fp:
            train_indices = pickle.load(fp)

        with open('./Data/eval_indices.pkl','rb') as fp:
            val_indices = pickle.load(fp)

    except OSError:
        if os.path.isfile('./Data/train_indices.pkl'):
            os.remove('./Data/train_indices.pkl')

        if os.path.isfile('./Data/eval_indices.pkl'):
            os.remove('./Data/eval_indices.pkl')
    
        shuffle_dataset = True
        random_seed = 42
        # 50k len(CIFAR)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))

        split = int(np.floor(config.data.train_eval_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[:split], indices[split:]
        

        with open('./Data/train_indices.pkl','wb') as outf:
            pickle.dump(train_indices, outf)

        with open('./Data/eval_indices.pkl','wb') as outf:
            pickle.dump(val_indices, outf)

    # recreate DataLoader with a new sampler in each epoch
    # https://discuss.pytorch.org/t/new-subset-every-epoch/85018
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)    
    
  
    # num_workers = 24 rm for debugging - RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
    collate_func = Collator()
    train_loader = DataLoader(dataset, batch_size=config.batch_size.train,
                                                    sampler=train_sampler,num_workers=config.hardware.num_workers.train, collate_fn=collate_func)
    
    validation_loader = DataLoader(dataset, batch_size=config.batch_size.eval,
                                                    sampler=val_sampler,num_workers=config.hardware.num_workers.eval, collate_fn=collate_func)

    # ---------------------
    # Params
    # ---------------------
    gpus = args.gpu
    gpus = gpus if isinstance(gpus, list) else [gpus]


    # ---------------------
    # Training
    # ---------------------

    val_check_interval = config.validation.val_check_interval
    check_val_every_n_epoch = config.validation.check_val_every_n_epoch
    assert val_check_interval is None or check_val_every_n_epoch is None

    trainer = Trainer(
        accelerator='gpu',
        callbacks=callbacks,
        # callbacks=[checkpoint_callback, early_stop_callback],
        enable_checkpointing=True,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        default_root_dir=None,
        devices=gpus,
        gradient_clip_val=config.training.gradient_clip_val,
        gradient_clip_algorithm='value',
        limit_train_batches=config.training.limit_train_batches if not args.trial_run else 4,
        limit_val_batches=config.validation.limit_val_batches if not args.trial_run else 2,
        logger=logger, #if not args.trial_run else None
        log_every_n_steps=config.logging.train.log_every_n_steps,
        plugins=None,
        # UserWarning: 16 is supported for historical reasons but its usage is discouraged. Please set your precision to 16-mixed instead!
        precision=config.training.precision,
        max_epochs=config.training.max_epochs if not args.trial_run else 2,
        max_steps=config.training.max_steps,
        # "auto" is default. None is not an acceptable strategy - TODO DDP
        strategy=strategy if strategy is not None else "auto",
        sync_batchnorm=False if strategy is None else True,
        benchmark=config.reproduce.benchmark,
        deterministic=config.reproduce.deterministic_flag,
        # mod
        reload_dataloaders_every_n_epochs=1
        )

    if args.reload_path: 
        trainer.fit(module, train_loader, validation_loader, ckpt_path=args.reload_path)

    else:
        trainer.fit(module, train_loader, validation_loader)




parser = argparse.ArgumentParser()
parser.add_argument("--walk", type=str, default='ts')
parser.add_argument("--preprocess", type = bool, default=True)

# parser.add_argument("--model", type=str, default='mxlstmvit', choices=['mxlstmvit','rvt']) for outer train module 
parser.add_argument('--config_relpath', type=str, default='configs/mxlstm_cfg.yaml')
parser.add_argument("--use_saved_data", type=bool, default = False)

parser.add_argument("--trial_run", type=bool, default=False)

parser.add_argument("--logger", type=str, default='wandb')
parser.add_argument("--gpu", type=int, default=1)

# RANDOM
parser.add_argument("--reload_path", type=str, default='/home/renaj/Driftiq/driftiq/tsmu500w50/last_epoch=epoch=021-step=step=58454.ckpt') # if not reload empty string
parser.add_argument("--kth_reload", type=int, default=12)

args = parser.parse_args()
if __name__ == "__main__":
    conf = OmegaConf.load(os.path.join(os.getcwd(),args.config_relpath))
    main(conf)