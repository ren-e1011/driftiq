# RVT > modules > detection.py

from typing import Any, Optional, Tuple, Union, Dict
# from warnings import warn

import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import CrossEntropyLoss
# import torch.distributed as dist
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning import LightningModule

from models.classification.classifier import Classifier
from envar import * 

from torch.nn.functional import one_hot, log_softmax
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, ConstantLR, SequentialLR

from torchmetrics import Accuracy

# For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
torch.set_float32_matmul_precision('high')




class RNNClassModule(LightningModule):
    def __init__(self,
                 config: DictConfig,
                 num_classes: int = 100
                 ):
        super().__init__()

        # ? config TODO mod 
        self.save_hyperparameters()

        # abstracts both the RVT + classifier head
        self.model = Classifier(config)
        # crossentropy loss combines logsoftmax + bceloss 
        self.loss = CrossEntropyLoss()

        self.num_classes = num_classes  

        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

        self.train_config = config.training

    # def configure_optimizers(self):
    #     opt = torch.optim.AdamW(self.model.parameters(), lr=0.0001)

    #     scheduler0 = LinearLR(opt, start_factor=0.01, total_iters=5)
    #     scheduler1 = ConstantLR(opt, factor=1.0, total_iters=40)
    #     scheduler2 = ExponentialLR(opt, gamma=0.9)
    #     scheduler = SequentialLR(opt, schedulers=[scheduler0,scheduler1, scheduler2], milestones=[2, 10])

    #     return [opt],[scheduler]
        # configure_optimizers from RVT.modules.detection.py
    def configure_optimizers(self):
        lr = self.train_config.learning_rate
        weight_decay = self.train_config.weight_decay
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_params = self.train_config.lr_scheduler
        if not scheduler_params.use:
            return optimizer

        total_steps = scheduler_params.total_steps
        assert total_steps is not None
        assert total_steps > 0
        # Here we interpret the final lr as max_lr/final_div_factor.
        # Note that Pytorch OneCycleLR interprets it as initial_lr/final_div_factor:
        final_div_factor_pytorch = scheduler_params.final_div_factor / scheduler_params.div_factor
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            div_factor=scheduler_params.div_factor,
            final_div_factor=final_div_factor_pytorch,
            total_steps=total_steps,
            pct_start=scheduler_params.pct_start,
            cycle_momentum=False,
            anneal_strategy='linear')
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": 'learning_rate',
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
    

    def _get_preds_loss_accuracy(self, batch):
        # t_pla_s = time.time()
        X, y = batch

        y_oh = one_hot(y.to(torch.int64),self.num_classes)
        # (batch_size,num_classes)
        y_oh = y_oh.squeeze().to(torch.float)

        yhat_logits = self.model(X)

        loss = self.loss(yhat_logits,y_oh)

        labels = y.squeeze().to(torch.float)
        yhat = torch.argmax(yhat_logits,dim=1)
        yhat = yhat.to(torch.float)
        # take sum of all batches and divide by batch_size

        # crossentropy loss combines logsoftmax + bceloss 
        # preds, target
        # RuntimeError: Encountered different devices in metric calculation (see stacktrace for details). This could be due to the metric class not being on the same device as input. Instead of `metric=MulticlassAccuracy(...)` 
        # try to do `metric=MulticlassAccuracy(...).to(device)` where device corresponds to the device of the input.
        
        acc = self.accuracy(yhat,labels)
        # t_pla_e = time.time()
        # print('Time to calculate preds, return loss', t_pla_e - t_pla_s)
        return yhat, loss, acc

    def predict_step(self, batch, batch_idx):
        # t_pred_s = time.time()
        X, y = batch
        yhat, loss = self.model(X,y)
        # t_pred_e = time.time()
        # print('Time to predict step', t_pred_e - t_pred_s)
        return yhat 
    
    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        # t_valsample_start = time.time()
        preds, loss, acc = self._get_preds_loss_accuracy(batch)
        # t_valsample_end = time.time()

        # print('Validation step', t_valsample_end - t_valsample_start)
        # Log loss and metric
        # prog_bar = True? 
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', acc, prog_bar=True)
        return preds

    # def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        # t_samplestep_s = time.time()
        _, loss, acc = self._get_preds_loss_accuracy(batch)
        # t_samplestep_e = time.time()
        # Log loss and metric
        # logger = True?
        # print('Training step,', t_samplestep_e - t_samplestep_s)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_accuracy', acc, on_step=True, on_epoch=True, logger=True)
        
        return loss