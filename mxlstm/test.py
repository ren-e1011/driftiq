import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent)) # for pythonpath 
sys.path.append(str(Path.cwd()))
from configs.envar import FILEPATH, CAMERA_RES, COMET_API_KEY
# os.chdir(FILEPATH)
sys.path.append(FILEPATH)

import os
from torch.backends import cuda, cudnn
cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True



from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.loggers import CometLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from Data.dataset import DataSet, Collator

from mxlstm.litmod import MxLSTMClassifier


# parser.add_argument("--use_saved_data", type=bool, default = False)

# RANDOM
# parser.add_argument("--reload_path", type=str, default='/home/renaj/DIQ/diq/mxlstmvitval27test1/checkpoints/last_epoch=epoch=019-step=step=53140.ckpt') # if not reload empty string
# parser.add_argument("--reload_hparams", type=str, default='/home/renaj/DIQ/lightning_logs/version_141/hparams.yaml')


from omegaconf import DictConfig, OmegaConf

def main(config:DictConfig, reload_path: str = None,reload_hparams: str = None, walk: str = None, logger: str = 'wandb', gpu: int = 0):

    reload_path = config.testing.reload_path if reload_path is None else reload_path
    reload_hparams = config.testing.reload_hparams if reload_hparams is None else reload_hparams
    walk = config.testing.walk if walk is None else walk
    logger = None if not logger else logger # default to None instead of config.testing.logger
    gpus = config.testing.gpu if gpu is None else gpu
    use_saved_data = config.testing.use_saved_data
    preprocess = config.testing.preprocess 

    # ---------------------
    # Model
    # ---------------------
    # module = MxLSTMClassifier(config) 
    model = MxLSTMClassifier.load_from_checkpoint(checkpoint_path=reload_path, hparams_file=reload_hparams)


    # ---------------------
    # DDP
    # ---------------------
    # gpus = config.hardware.gpus
    gpus = gpus if isinstance(gpus, list) else [gpus]
    distributed_backend = config.hardware.dist_backend
    assert distributed_backend in ('nccl', 'gloo'), f'{distributed_backend=}'
    strategy = DDPStrategy(process_group_backend=distributed_backend,
                           find_unused_parameters=False,
                           gradient_as_bucket_view=True) if len(gpus) > 1 else None
    
    # ---------------------
    # Logging and Checkpoints
    # ---------------------

    run_name = f"test_mxvit_{walk}walk"
    if logger == 'wandb': 
        logger = WandbLogger(project='diq',name= run_name,job_type='train', log_model='all')
        logger.watch(model=module, log='all', log_freq=config.logging.train.log_model_every_n_steps, log_graph=True)
    elif logger == 'comet':
        logger = CometLogger(api_key=COMET_API_KEY,project_name='driftiq',experiment_name=run_name)
        # if not testing code
    else:
        logger = None
 
    # ---------------------
    # load dataset - move to data module
    # ---------------------
    dataset = DataSet(walk = walk, 
                      architecture= config.model.name,
                      steps = config.time.steps, bins=config.time.bins, 
                      refrac_pd=config.emulator.refrac_pd, threshold= config.emulator.threshold,
                      use_saved_data= use_saved_data, preproc_data= preprocess,
                      frame_hw = (config.input.height,config.input.width), fps=config.time.fps, test=True)
    
    # 10k len(CIFAR_test)
    dataset_size = len(dataset)
    test_indices = list(range(dataset_size))

    # test_sampler = SubsetRandomSampler(test_indices)

    collate_func = Collator()
    test_loader = DataLoader(dataset, batch_size=dataset_size, num_workers=config.hardware.num_workers.eval) # , collate_fn=collate_func

    checkpoint_callback = ModelCheckpoint(reload_path)

    trainer = Trainer(
        accelerator='gpu',
        devices=gpus,
        logger=logger, #if not args.trial_run else None
        # log_every_n_steps=config.logging.train.log_every_n_steps,
        precision=config.training.precision,
        strategy=strategy if strategy is not None else "auto",
        sync_batchnorm=False if strategy is None else True,
        benchmark=config.reproduce.benchmark,
        deterministic=config.reproduce.deterministic_flag,
        checkpoint_callback = False
        )
    
        
    trainer.test(model, dataloaders=test_loader)
    return trainer

if __name__ == "__main__":
    import configargparse as argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_saved_data", type=bool, default = False)
    parser.add_argument('--config_relpath', type=str, default='configs/mxlstm_cfg.yaml')

    # RANDOM
    parser.add_argument("--reload_path", type=str, default='/home/renaj/DIQ/diq/mxlstmvitval27test1/checkpoints/last_epoch=epoch=019-step=step=53140.ckpt') # if not reload empty string
    parser.add_argument("--reload_hparams", type=str, default='/home/renaj/DIQ/lightning_logs/version_141/hparams.yaml')
    args = parser.parse_args()

    conf = OmegaConf.load(os.path.join(os.getcwd(),args.config_relpath))
    main(config=conf, reload_path=args.reload_path,reload_hparams=args.reload_hparams)