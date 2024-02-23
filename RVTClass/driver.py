# wrapper for RVT > train.py
from envar import *

os.chdir(RVT_FILEPATH)

# in envar 
# import numpy as np

# RVT train.py and https://pytorch.org/docs/master/multiprocessing.html?highlight=sharing%20strategy#sharing-strategies
# and start of https://pytorch.org/docs/stable/notes/cuda.html
# see also https://github.com/pytorch/pytorch/issues/11201
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True

import argparse

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
from pytorch_lightning.strategies import DDPStrategy

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from callbacks.custom import get_ckpt_callback
from callbacks.gradflow import GradFlowLogCallback
from loggers.utils import get_ckpt_path


from dataset import DataSet


parser = argparse.ArgumentParser()
parser.add_argument("--walk", type=str, default='random')

parser.add_argument("--n_frames",type=int,default=300)
parser.add_argument("--timesteps", type=int, default=40)
parser.add_argument("--refrac_pd", type=int, default=0.0)
parser.add_argument("--threshold", type=int, default=0.4)

parser.add_argument("--frame_rate_hz",type=int,default=50)

parser.add_argument("--use_saved_data", type=bool, default = False)

parser.add_argument("--testing", type=bool, default=False)

parser.add_argument("--run_name", type=str, default = 'cutedges40ts40bin96sensorallspikes')

args = parser.parse_args()


import hydra
from omegaconf import DictConfig, OmegaConf
from modules.rnnclass_module import RNNClassModule
from torch import multiprocessing



@hydra.main(config_path='config', config_name='train', version_base='1.2')
def main(config: DictConfig):
    # does not seem to be working 
    torch.cuda.memory._record_memory_history()

    # snippet from RVT.train
    # ---------------------
    # DDP
    # ---------------------
    gpu_config = config.hardware.gpus
    gpus = OmegaConf.to_container(gpu_config) if OmegaConf.is_config(gpu_config) else gpu_config
    gpus = gpus if isinstance(gpus, list) else [gpus]
    distributed_backend = config.hardware.dist_backend
    assert distributed_backend in ('nccl', 'gloo'), f'{distributed_backend=}'
    strategy = DDPStrategy(process_group_backend=distributed_backend,
                           find_unused_parameters=False,
                           gradient_as_bucket_view=True) if len(gpus) > 1 else None
    

    
    # ---------------------
    # Model
    # ---------------------
    # module = fetch_model_module(config=config) MOD 
    module = RNNClassModule(config)
    

        # ---------------------
    # Logging and Checkpoints
    # ---------------------
    # logger = get_wandb_logger(config) MOD 
    if not args.testing: 
        logger = WandbLogger(project='diq',name= args.run_name,job_type='train', log_model='all')
        # if not testing code
        logger.watch(model=module, log='all', log_freq=config.logging.train.log_model_every_n_steps, log_graph=True)
    else:
        logger = None
    
    ckpt_path = None

    if config.wandb.artifact_name is not None:
        ckpt_path = get_ckpt_path(logger, wandb_config=config.wandb)
    
    if ckpt_path is not None and config.wandb.resume_only_weights:
        print('Resuming only the weights instead of the full training state')
        module = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config})
        ckpt_path = None
    # end snippet

    # ---------------------
    # load dataset - move to data module
    # ---------------------
    dataset = DataSet(config,args)
    # snippet from https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    shuffle_dataset = True
    random_seed = 42
    # 50k len(CIFAR)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    split = int(np.floor(TRAIN_SPLIT_SZ * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    # recreate DataLoader with a new sampler in each epoch
    # https://discuss.pytorch.org/t/new-subset-every-epoch/85018
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    # num_workers = 24 rm for debugging - RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, 
                                            sampler=train_sampler, num_workers=0)
    # no need to reshuffle validation 
    validation_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                                    sampler=val_sampler,num_workers=0)
    # end snippet 
    ## end move to data module
    
    # pl_module = RNNClassModule(config)
    # snippets from https://docs.wandb.ai/tutorials/lightning, 
    # https://docs.wandb.ai/guides/integrations/lightning#logger-arguments - see for checkpoint load

    # mod testing
    # if not args.testing:
    #     wandb_logger = WandbLogger(project='diq',name= args.run_name,job_type='train', log_model='all')
    # else:
    #     wandb_logger = None

    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = list()
    # callbacks.append(get_ckpt_callback(config))
    callbacks.append(GradFlowLogCallback(config.logging.train.log_model_every_n_steps))
    if config.training.lr_scheduler.use:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    # if config.logging.train.high_dim.enable or config.logging.validation.high_dim.enable:
    #     viz_callback = get_viz_callback(config=config)
    #     callbacks.append(viz_callback)
    callbacks.append(ModelSummary(max_depth=2))
    # see RVT > callbacks > custom > get_ckpt_callback 
    ckpt_filename = f"{args.run_name}_"+'epoch={epoch:03d}-step={step}- ={ :.2f}'

    checkpoint_callback = ModelCheckpoint(monitor='val_accuracy',mode='max',
                                          every_n_epochs=config.logging.ckpt_every_n_epochs,
                                          filename=ckpt_filename,
                                          save_top_k=1,save_last=True,verbose=True)
    checkpoint_callback.CHECKPOINT_NAME_LAST = 'last_epoch={epoch:03d}-step={step}'
    callbacks.append(checkpoint_callback)
    early_stop_callback = EarlyStopping(monitor="val_loss")
    callbacks.append(early_stop_callback)

    # snippet from RVT.train
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
        limit_train_batches=config.training.limit_train_batches if not args.testing else 4,
        limit_val_batches=config.validation.limit_val_batches if not args.testing else 2,
        logger=logger, #if not args.testing else None
        log_every_n_steps=config.logging.train.log_every_n_steps,
        plugins=None,
        # UserWarning: 16 is supported for historical reasons but its usage is discouraged. Please set your precision to 16-mixed instead!
        precision=config.training.precision,
        max_epochs=config.training.max_epochs if not args.testing else 2,
        max_steps=config.training.max_steps,
        # "auto" is default. None is not an acceptable strategy 
        strategy=strategy if strategy is not None else "auto",
        sync_batchnorm=False if strategy is None else True,
        benchmark=config.reproduce.benchmark,
        deterministic=config.reproduce.deterministic_flag,
        # mod
        reload_dataloaders_every_n_epochs=1
        )


    trainer.fit(module, train_loader, validation_loader)

    torch.cuda.memory._dump_snapshot(f"../logs/{args.run_name}_memsnapshot.pickle")


if __name__ == "__main__":
    main()
    # config = OmegaConf.load('./config/train.yaml')
    # https://stackoverflow.com/questions/72779926/gunicorn-cuda-cannot-re-initialize-cuda-in-forked-subprocess
    # multiprocessing.set_start_method('spawn')
    # p = multiprocessing.Process(target=main, daemon=True)
    # p.start()
    # p.join()
    # main()

    