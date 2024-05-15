import sys
from configs.envar import FILEPATH, CAMERA_RES, COMET_API_KEY
# os.chdir(FILEPATH)
sys.path.append(FILEPATH)
import yaml
import pickle

from torch.backends import cuda, cudnn
cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True

import configargparse as argparse

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.loggers import CometLogger
from pytorch_lightning.strategies import DDPStrategy

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


from Data.dataset import DataSet, Collator


from mxlstm.litmod import MxLSTMClassifier


parser = argparse.ArgumentParser()
parser.add_argument("--walk", type=str, default='random')
parser.add_argument("--preprocess", type = bool, default=True)

parser.add_argument("--use_saved_data", type=bool, default = False)
parser.add_argument('--config_relpath', type=str, default='configs/mxlstm_cfg.yaml')

parser.add_argument("--logger", type=str, default='wandb')
parser.add_argument("--gpu", type=int, default=0)

# RANDOM
parser.add_argument("--reload_path", type=str, default='/home/renaj/DIQ/diq/mxlstmvitval27test1/checkpoints/last_epoch=epoch=019-step=step=53140.ckpt') # if not reload empty string
parser.add_argument("--reload_hparams", type=str, default='/home/renaj/DIQ/lightning_logs/version_141/hparams.yaml')
parser.add_argument("--kth_reload", type=int, default=0)

args = parser.parse_args()

from omegaconf import DictConfig, OmegaConf

def main(config:DictConfig):

    # ---------------------
    # Model
    # ---------------------
    module = MxLSTMClassifier(config) 

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

    run_name = f"test_mxvit_{args.walk}walk_reload{args.kth_reload}"
    if args.logger == 'wandb': 
        logger = WandbLogger(project='diq',name= run_name,job_type='train', log_model='all')
        logger.watch(model=module, log='all', log_freq=config.logging.train.log_model_every_n_steps, log_graph=True)
    elif args.logger == 'comet':
        logger = CometLogger(api_key=COMET_API_KEY,project_name='driftiq',experiment_name=run_name)
        # if not testing code
    else:
        logger = None
 
    # ---------------------
    # load dataset - move to data module
    # ---------------------
    dataset = DataSet(walk = args.walk, 
                      architecture= config.model.name,
                      steps = config.time.steps, bins=config.time.bins, 
                      refrac_pd=config.emulator.refrac_pd, threshold= config.emulator.threshold,
                      use_saved_data= args.use_saved_data,
                      frame_hw = (config.input.height,config.input.width), fps=config.time.fps, preproc_data=args.preprocess, test=True)
    
    # 10k len(CIFAR_test)
    dataset_size = len(dataset)
    test_indices = list(range(dataset_size))

    test_sampler = SubsetRandomSampler(test_indices)

    collate_func = Collator()
    test_loader = DataLoader(dataset, batch_size=config.batch_size.eval, sampler=test_sampler, num_workers=config.hardware.num_workers.eval, collate_fn=collate_func)

    trainer = Trainer(
        accelerator='gpu',
        enable_checkpointing=True,
        default_root_dir=None,
        devices=gpus,
        logger=logger, #if not args.trial_run else None
        # log_every_n_steps=config.logging.train.log_every_n_steps,
        plugins=None,
        precision=config.training.precision,
        strategy=strategy if strategy is not None else "auto",
        sync_batchnorm=False if strategy is None else True,
        benchmark=config.reproduce.benchmark,
        deterministic=config.reproduce.deterministic_flag,
        )
    
    model = module.load_from_checkpoint(checkpoint_path=args.reload_path, hparams_file=args.reload_hparams)
        
    trainer.test(model, dataloaders=test_loader)

if __name__ == "__main__":
    conf = OmegaConf.load(os.path.join(os.getcwd(),args.config_relpath))
    main(conf)