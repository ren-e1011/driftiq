# wrapper for RVT > train.py
from envar import *

os.chdir(RVT_FILEPATH)

import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary

from pytorch_lightning.strategies import DDPStrategy
import wandb

# from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

from dataset import DataSet



# move to run file
parser = argparse.ArgumentParser()
parser.add_argument("--walk", type=str, default='random')

parser.add_argument("--n_frames",type=int,default=300)
parser.add_argument("--frame_rate_hz",type=int,default=50)

# run list of indices in parallel
# number of data to generate
parser.add_argument("--n_im", type=int, default=4)
# Clean vs noisy for events generation - not relevant for this project
parser.add_argument("--condition",type=str,default="Clean")


args = parser.parse_args()




import hydra
from omegaconf import DictConfig, OmegaConf
from modules.rnnclass_module import RNNClassModule

# HYDRA_FULL_ERROR = 1

# snippet from https://docs.wandb.ai/tutorials/lightning
class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        trainer.logger.experiment.log({"examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                           for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                 preds[:self.num_samples], 
                                                 val_labels[:self.num_samples])]})

# end snippet 

@hydra.main(config_path='config', config_name='train', version_base='1.2')
def main(config: DictConfig):

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
# end snippet


    # t_modeload_start = time.time()
    # trying to add workers
    # torch.multiprocessing.set_start_method('spawn')
    dataset = DataSet(args)

    
    
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

    # Creating PT data samplers and loaders:
    # 
    train_sampler = SubsetRandomSampler(train_indices)
    # val_sampler = SubsetRandomSampler(val_indices)
    val_sampler = SequentialSampler(val_indices)
    # no need to shuffle validation 

    
    
    # num_workers = 24 rm for debugging - RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, 
                                            sampler=train_sampler, num_workers=0)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                                    sampler=val_sampler,num_workers=0)
    # end snippet 
   
    # from RVT > config > modifier.py called in first line of RVT > train.py
    # config.dataset
    
    pl_module = RNNClassModule(config)
    # t_modeload_end = time.time()
    # snippets from https://docs.wandb.ai/tutorials/lightning, 
    # https://docs.wandb.ai/guides/integrations/lightning#logger-arguments - see for checkpoint load

    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    # val_samples = next(iter(validation_loader()))
    # val_imgs, val_labels = val_samples[0], val_samples[1]

    # print('Time for dataset, dataloader, model to load', t_modeload_end - t_modeload_start)
    wandb_logger = WandbLogger(project='diq',name='fewSpikesRandomRun',job_type='train', log_model='all')



    checkpoint_callback = ModelCheckpoint(monitor='val_accuracy',mode='max')
    early_stop_callback = EarlyStopping(monitor="val_loss")


    # snippet from RVT.train
     # ---------------------
    # Training
    # ---------------------

    val_check_interval = config.validation.val_check_interval
    check_val_every_n_epoch = config.validation.check_val_every_n_epoch
    assert val_check_interval is None or check_val_every_n_epoch is None

    trainer = Trainer(
        accelerator='gpu',
        # callbacks=callbacks,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_checkpointing=True,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        default_root_dir=None,
        devices=gpus,
        gradient_clip_val=config.training.gradient_clip_val,
        gradient_clip_algorithm='value',
        limit_train_batches=config.training.limit_train_batches,
        limit_val_batches=config.validation.limit_val_batches,
        logger=wandb_logger,
        log_every_n_steps=config.logging.train.log_every_n_steps,
        plugins=None,
        # UserWarning: 16 is supported for historical reasons but its usage is discouraged. Please set your precision to 16-mixed instead!
        precision=config.training.precision,
        max_epochs=config.training.max_epochs,
        max_steps=config.training.max_steps,
        # "auto" is default. None is not an acceptable strategy 
        strategy=strategy if strategy is not None else "auto",
        sync_batchnorm=False if strategy is None else True,
        # TypeError: Trainer.__init__() got an unexpected keyword argument 'move_metrics_to_cpu'
        # move_metrics_to_cpu=False,
        benchmark=config.reproduce.benchmark,
        deterministic=config.reproduce.deterministic_flag,
        )

    # mod from accelerator='gpu', devices=2, strategy='ddp'
    # trainer = Trainer(reload_dataloaders_every_n_epochs=1,accelerator='gpu',logger=wandb_logger, callbacks=[checkpoint_callback, early_stop_callback], devices=1,strategy='ddp')
    # THIS ONE 
    # trainer = Trainer(reload_dataloaders_every_n_epochs=1,accelerator='gpu',callbacks=[checkpoint_callback, early_stop_callback], devices=1,strategy='ddp', precision=16)

    trainer.fit(pl_module, train_loader, validation_loader)


if __name__ == "__main__":
    # config = OmegaConf.load('./config/train.yaml')
    # torch.multiprocessing.set_start_method('spawn')
    main()

    