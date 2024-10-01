import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from termcolor import colored
import numpy as np
import pickle
import os
from copy import deepcopy

from torchvision.models import resnet
# from torchvision.transforms import v2
from torchvision.transforms.v2 import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    Resize, 
                                    ToImage,
                                    ToDtype)

from mxlstm.layers.EventDropout import EventDropout
from mxlstm.layers.SELayer import SELayer
from mxlstm.layers.MatrixLSTM import MatrixLSTM
# from matrixlstm.classification.layers.MatrixConvLSTM import MatrixConvLSTM
from mxlstm.models.network import Network
from collections import OrderedDict

from transformers import ViTImageProcessor, ViTForImageClassification

from utils.mxlstm_utils import nevents_x_coord


class MatrixLSTMViT(Network):

    # def __init__(self, input_shape, embedding_size,
    #              matrix_hidden_size, matrix_region_shape, matrix_region_stride,
    #              matrix_add_coords_feature, matrix_add_time_feature_mode,
    #              matrix_normalize_relative, matrix_lstm_type, num_classes = 100, 
    #              matrix_keep_most_recent=True, matrix_frame_intervals=1, matrix_frame_intervals_mode=None,
    #              pretrainedvit_base = 'google/vit-base-patch16-224', cifar100labelpath = '/home/renaj/Driftiq/Data/cifar100_labelxix',
            
    #             #  event_dropout=0.25, 
    #             event_dropout=0.0,
    #              frame_dropout=-1, fc_dropout=-1, frame_actfn=None,
    #              lstm_num_layers=1,
    #              ):
    def __init__(self, input_shape, embedding_size,
                 matrix_hidden_size, matrix_region_shape, matrix_region_stride,
                 matrix_add_coords_feature, matrix_add_time_feature_mode, 
                 matrix_normalize_relative, matrix_lstm_type, classes, num_classes = 100, 
                 matrix_keep_most_recent=True, matrix_frame_intervals=1, matrix_frame_intervals_mode=None,
                 pretrainedvit_base = 'google/vit-base-patch16-224',
            
                #  event_dropout=0.25, 
                event_dropout=0.0,
                 frame_dropout=-1, fc_dropout=-1, frame_actfn=None,
                 lstm_num_layers=1,
                 ):
        super().__init__()

        self.frames = None

        self.height, self.width = input_shape
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.use_embedding = self.embedding_size > 0
        self.input_shape = input_shape
        self.lstm_num_layers = lstm_num_layers



        self.matrix_lstm_type = matrix_lstm_type
        self.matrix_hidden_size = matrix_hidden_size
        self.matrix_region_shape = matrix_region_shape
        self.matrix_region_stride = matrix_region_stride
        self.matrix_add_coords_feature = matrix_add_coords_feature
        self.matrix_add_time_feature_mode = matrix_add_time_feature_mode
        self.matrix_normalize_relative = matrix_normalize_relative
        self.matrix_keep_most_recent = matrix_keep_most_recent
        self.matrix_frame_intervals = matrix_frame_intervals
        self.matrix_frame_intervals_mode = matrix_frame_intervals_mode

        self.event_dropout = event_dropout
        self.frame_dropout = frame_dropout
        self.fc_dropout = fc_dropout
        if isinstance(frame_actfn, str):
            self.frame_actfn = getattr(torch, frame_actfn)
        else:
            self.frame_actfn = frame_actfn

        if self.event_dropout > 0:
            self.eventdrop = EventDropout(drop_prob=event_dropout)
        if self.use_embedding:
            self.embedding = nn.Embedding(self.height * self.width, embedding_size)

        matrix_input_size = self.embedding_size + 1 if self.use_embedding else 1
        # MatrixLSTMClass = MatrixConvLSTM if self.matrix_lstm_type == "ConvLSTM" else MatrixLSTM
        MatrixLSTMClass = MatrixLSTM
        self.matrixlstm = MatrixLSTMClass(self.input_shape, self.matrix_region_shape,
                                          self.matrix_region_stride, matrix_input_size,
                                          self.matrix_hidden_size, self.lstm_num_layers,
                                          bias=True, lstm_type=self.matrix_lstm_type,
                                          add_coords_feature=self.matrix_add_coords_feature,
                                          add_time_feature_mode=self.matrix_add_time_feature_mode,
                                          normalize_relative=self.matrix_normalize_relative,
                                          keep_most_recent=self.matrix_keep_most_recent,
                                          frame_intervals=self.matrix_frame_intervals,
                                          frame_intervals_mode=self.matrix_frame_intervals_mode)


        if self.frame_dropout > 0:
            self.framedrop = nn.Dropout(p=self.frame_dropout)


        # can be replaced with filter matching 
                # Adds a 1x1 convolution that maps matrix_hidden_size into 3 channels
        self.project_hidden = nn.Conv2d(self.matrixlstm.out_channels,
                                                3,
                                                kernel_size=1,
                                                stride=1,
                                                bias=True)
        # for increasing (96,96) resolution to (224,224)
        self.im_processor = ViTImageProcessor.from_pretrained(f"{pretrainedvit_base}")
        image_mean = self.im_processor.image_mean
        image_std = self.im_processor.image_std
        size = self.im_processor.size["height"]
        normalize = Normalize(mean=image_mean, std=image_std)

        # Transforms of the MatrixLSTM sensor
        self.transforms = Compose([
            Resize(size, antialias=True),
            # TODO CenterCrop(size) RandomResizedCrop, RandomHorizontalFlip
            ToImage(), 
            ToDtype(torch.float32, scale=True), # ToTensor deprecated
            normalize

        ])

        # # TODO test s
        # id2label_path = os.path.join(cifar100labelpath,'id2label.pkl')
        # try:
        #     with open( id2label_path,'rb') as file:
        #         id2label = pickle.load(file)
        # except FileNotFoundError:
        #     id2label = None
        
        # label2id_path = os.path.join(cifar100labelpath,'label2id.pkl')
        # try:
        #     with open(label2id_path,'rb') as file:
        #         label2id = pickle.load(file)
        # except FileNotFoundError:
        #     label2id = None

        id2label = {idx: label for idx, label in enumerate(classes)}
        label2id = {label: idx for idx, label in enumerate(classes)}

        self.vit = ViTForImageClassification.from_pretrained(f"{pretrainedvit_base}",
                                                              num_labels=num_classes,
                                                              id2label=id2label,
                                                              label2id=label2id,
                                                              ignore_mismatched_sizes=True) # in case you're planning to fine-tune an already fine-tuned checkpoint, like facebook/convnext-tiny-224 (which has already been fine-tuned on ImageNet-1k), then you need to provide the additional argument ignore_mismatched_sizes=True to the from_pretrained method. This will make sure the output head (with 1000 output neurons) is thrown away and replaced by a new, randomly initialized classification head that includes a custom number of output neurons
        # Add a new layer after resnet mapping original 1000 ImageNet classes into num_classes
        # TODO Consider more layers or replacing last later 
        # self.vit = nn.Sequential(OrderedDict([('backbone', self.vit),
        #                                              ('fc', nn.Linear(1000, num_classes))]))


    def init_params(self):
        self.matrixlstm.reset_parameters()
        if self.use_embedding:
            nn.init.normal_(self.embedding.weight, mean=0, std=1)

    def coord2idx(self, x, y):
        return y * self.width + x

    # lengths is a nested list of event spikes for each sample
    def forward(self, events, lengths):

        # Events dropout during training
        # if self.event_dropout > 0:
        #     events, lengths = self.eventdrop(events, lengths)

        # # events.shape = [batch_size, time_size, 4]
        batch_size, time_size, features_size = events.shape # time size is the (max-padded) number of events
        assert(features_size == 4)


        # x = events[:, :, 0].type(torch.int64)
        # y = events[:, :, 1].type(torch.int64)
        # ts = events[:, :, 2].float()
        # p = events[:, :, 3].float()

        # ordering as in preprocess.construct_x for full batch
        # typecasting as above net_matrixlstm_resnet.forward
        ts = events[:,:,0].float()
        x = events[:,:,1].int()
        y = events[:,:,2].int()
        p = events[:,:,3].int()

        if self.use_embedding:
            # Given the event coordinates, retrieves the pixel number
            # associated to the event position
            # embed_idx.shape = [batch_size, time_size]
            embed_idx = self.coord2idx(x, y)

            # Retrieves the actual embeddings
            # [batch_size, time_size, embedding_size]
            # 
            embed = self.embedding(embed_idx)
            # Adds the polarity to each embedding
            # [batch_size, time_size, embedding_size]
            embed = torch.cat([embed, p.unsqueeze(-1)], dim=-1)
        else:
            embed = p.unsqueeze(-1)

        # [batch_size, time_size, hidden_size]
        coords = torch.stack([x, y], dim=-1)

        
        nevents_x_step = deepcopy(lengths)
        # transform events 
        lengths = nevents_x_coord(coords, lengths, self.input_shape[0], self.input_shape[1])

        # out_dense.shape = [batch_size, matrix_out_h, matrix_out_w, matrix_hidden_size]
        # embed in batch_size, time_size, embed_size + bias (17) or 1 if not self.use_embedding
        out_dense = self.matrixlstm(input=(embed, coords, ts.unsqueeze(-1), lengths, nevents_x_step)) # mod pass in nevents_x_step
        out_dense = out_dense.permute(0, 3, 1, 2)

    

        if self.matrixlstm.out_channels != 3 and not self.resnet_replace_first:
            out_dense = self.project_hidden(out_dense)

        if self.frame_actfn is not None:
            out_dense = self.frame_actfn(out_dense)

        # if self.resnet_meanstd_norm:
        #     if self.frame_actfn is None:
        #         # MatrixLSTM uses a tanh as output activation
        #         # we use range normalization with min=-1, max=1
        #         img_min, img_max = -1, 1
        #         out_dense = (out_dense - img_min) / (img_max - img_min)
        #     elif self.frame_actfn is not torch.sigmoid:
        #         # range-normalization
        #         n_pixels = self.matrixlstm.output_shape[0] * self.matrixlstm.output_shape[1]
        #         flat_dense = out_dense.view(-1, self.matrixlstm.hidden_size, n_pixels)
        #         img_max = flat_dense.max(dim=-1)[0].view(-1, self.matrixlstm.hidden_size, 1, 1)
        #         img_min = flat_dense.min(dim=-1)[0].view(-1, self.matrixlstm.hidden_size, 1, 1)
        #         out_dense = (out_dense - img_min) / (img_max - img_min)
        #     # z-normalization
        #     out_dense = (out_dense - self.resnet_mean) / self.resnet_std

        if not self.training:
            self.frames = out_dense

        if self.frame_dropout > 0:
            out_dense = self.framedrop(out_dense)

        
        in_vit = self.transforms(out_dense)

        out_vit = self.vit(in_vit)

        # Computes log probabilities
        # log_probas.shape = [batch_size, num_classes]
        # log_probas = F.log_softmax(out_vit, dim=-1) AttributeError: 'ImageClassifierOutput' object has no attribute 'log_softmax'
        # log_probas = F.log_softmax(out_vit[-1])

        return out_vit.logits

    def loss(self, input, target):
        # RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Float' 
        #         target.dtype
        # torch.float32
        # input.dtype
        # torch.float32
        

        return F.nll_loss(input, target)

    def log_validation(self, logger, global_step):
        if not self.training:
            logframes = self.frames[:, :3]
            logframes_chans = logframes.shape[1]
            if logframes_chans == 3:
                images = vutils.make_grid(logframes, normalize=True, scale_each=True)
                logger.add_image('validation/frames', images, global_step=global_step)
