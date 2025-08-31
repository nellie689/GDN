from functools import partial
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

import os, sys
os.chdir(sys.path[0])
sys.path.append("..")
from utils.utils import instantiate_from_config


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, valid=None, test=None, num_workers=None, *args, **kwargs):
        super().__init__()
        self.batch_size = batch_size


        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        # print(batch_size)
        # print(train)
        # print(validation)
        # print(test)
        # print(predict)
        self.Trainshuffle = kwargs['Trainshuffle'] if 'Trainshuffle' in kwargs else True
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if valid is not None:
            self.dataset_configs["valid"] = valid
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # print(stage)
        # assert 2>124
        # print(self.dataset_configs.keys())
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        # print(self.datasets)
        # assert 3>123

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=self.Trainshuffle)

    def _val_dataloader(self):
        # print("123", self.datasets.keys())
        # assert 3>789
        return DataLoader(self.datasets["valid"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)


