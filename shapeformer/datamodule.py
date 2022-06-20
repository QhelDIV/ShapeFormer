import pytorch_lightning as pl
import torch
import numpy as np
import copy
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from shapeformer import data
from xgutils import *
from xgutils.vis import npfvis


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, test_batch_size=None, val_batch_size=None, num_workers: int = 8,
                 trainset_opt={"class": None, "kwargs": {}},
                 valset_opt={"class": None, "kwargs": {}},
                 testset_opt={"class": None, "kwargs": {}},
                 visualset_opt={"class": None, "kwargs": {}}):
        super().__init__()
        trainset_opt = copy.deepcopy(trainset_opt)
        testset_opt = copy.deepcopy(testset_opt)
        if 'split' not in trainset_opt["kwargs"]:
            trainset_opt["kwargs"]['split'] = 'train'
        if 'split' not in valset_opt["kwargs"]:
            valset_opt["kwargs"]['split'] = 'val'
        if 'split' not in testset_opt["kwargs"]:
            testset_opt["kwargs"]['split'] = 'test'

        self.__dict__.update(locals())
        self.test_batch_size = test_batch_size if test_batch_size is not None else batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else test_batch_size
        self.dims = (1, 1, 1)
        self.pin_memory = False

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_set, self.val_set, self.test_set = None, None, None
        if stage == 'fit' or stage == 'train' or stage == "val" or stage is None:
            self.train_set = sysutil.instantiate_from_opt(self.trainset_opt)
            self.val_set = sysutil.instantiate_from_opt(self.valset_opt)
        if stage == "test" or stage is None or self.val_set is None or self.testset_opt["class"] is not None:
            self.test_set = sysutil.instantiate_from_opt(self.testset_opt)

        if self.valset_opt["class"] is None:
            self.val_set = self.test_set
            self.val_batch_size = self.test_batch_size

        if self.visualset_opt["class"] is None:
            self.visual_set = self.val_set
        else:
            self.visual_set = sysutil.instantiate_from_opt(self.visualset_opt)

    def train_dataloader(self, shuffle=True):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self, shuffle=False):
        return DataLoader(self.val_set, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self, shuffle=False):
        return DataLoader(self.test_set, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def visual_dataloader(self, shuffle=False):
        return DataLoader(self.visual_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=self.pin_memory)


class DataModule2(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, test_batch_size=None, val_batch_size=None, num_workers: int = 8,
                 trainset_opt={"class": None, "kwargs": {}},
                 valset_opt={"class": None, "kwargs": {}},
                 testset_opt={"class": None, "kwargs": {}},
                 visualset_opt={"class": None, "kwargs": {}},
                 dset_opts={}):
        super().__init__()
        if trainset_opt["class"] is not None:
            dset_opts["train"] = trainset_opt
        if valset_opt["class"] is not None:
            dset_opts["val"] = valset_opt
        if testset_opt["class"] is not None:
            dset_opts["test"] = testset_opt
        if visualset_opt["class"] is not None:
            dset_opts["visual"] = visualset_opt
        for key in dset_opts:
            # these dicts may share same fields, so let's replace them with their deepcopies
            dset_opts[key] = copy.deepcopy(dset_opts[key])
            if "split" not in dset_opts["key"]["kwargs"]:
                dset_opts["key"]["kwargs"]["split"] = key
        self.__dict__.update(locals())
        self.test_batch_size = test_batch_size if test_batch_size is not None else batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.dims = (1, 1, 1)
        self.pin_memory = False

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.train_set, self.val_set, self.test_set = None, None, None
        if self.visualset_opt["class"] is None:
            self.visual_set = None
        else:
            self.visual_set = sysutil.instantiate_from_opt(self.visualset_opt)
        if stage == 'fit' or stage == 'train' or stage == "val" or stage is None:
            self.train_set = sysutil.instantiate_from_opt(self.trainset_opt)
            self.val_set = sysutil.instantiate_from_opt(self.valset_opt)
        if stage == "test" or stage is None or self.val_set is None:
            self.test_set = sysutil.instantiate_from_opt(self.testset_opt)
        if self.val_set is None:
            self.val_set = self.test_set
            self.val_batch_size = self.test_batch_size
        if self.visual_set is None:
            self.visual_set = self.test_set

    def train_dataloader(self, shuffle=True):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self, shuffle=False):
        return DataLoader(self.val_set, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self, shuffle=False):
        return DataLoader(self.test_set, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def visual_dataloader(self, shuffle=False):
        return DataLoader(self.visual_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=self.pin_memory)
