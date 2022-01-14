#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import torch
from kinetics import Kinetics

def construct_loader(cfg, split, logger):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg['train']['dataset']
        batch_size = cfg['train']['batch_size']
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg['train']['dataset']
        batch_size = cfg['train']['batch_size']
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg['test']['dataset']
        batch_size = cfg['test']['batch_size']
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = Kinetics(cfg, split, logger)

    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg['data_loader']['num_workers'],
            pin_memory=cfg['data_loader']['pin_memory'],
            drop_last=drop_last,
        )

    return loader