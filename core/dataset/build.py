'''
Copyright (c) 2023 SLAB Group
Author: Tae Ha "Jeff" Park (tpark94@stanford.edu)
'''

import torch
import pytorch3d
import pytorch3d.structures
from pytorch3d.structures import (
    Meshes,
    join_meshes_as_batch,
)
import numpy as np
import random

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .SatReconDataset import SatReconDataset


def mesh_collate_fn(data):
    collated_batch = {}
    # Iterate over each key-value in the dictionary
    for key in data[0]:
        # Gather the data from the batch for the current key
        values = [item[key] for item in data]

        # Check if the key's values are tensors
        if isinstance(values[0], torch.Tensor):
            # If the values are tensors, stack them into a single tensor
            collated_batch[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], pytorch3d.structures.meshes.Meshes):
            collated_batch[key] = join_meshes_as_batch(values)
        else:
            collated_batch[key] = values

    return collated_batch


def _seed_worker(worker_id):
    """ Set seeds for dataloader workers. For more information, see below
    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataset(cfg, split='train', output_mesh=True):
    # TODO: create dedicated image transform script later
    transforms = [
        A.Resize(cfg.DATASET.IMAGE_SIZE[1], cfg.DATASET.IMAGE_SIZE[0]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    transforms = A.Compose(transforms)

    dataset = SatReconDataset(cfg, split, transforms=transforms, output_mesh=output_mesh)

    return dataset


def get_dataloader(cfg, split='train', distributed=False, output_mesh=True):
    # TODO: Temporary
    assert not distributed

    if split == 'train':
        images_per_gpu = cfg.TRAIN.BATCH_SIZE_PER_GPU
        shuffle = cfg.TRAIN.SHUFFLE
        num_workers = min(cfg.TRAIN.BATCH_SIZE_PER_GPU, cfg.TRAIN.WORKERS)
    elif split == 'validation':
        images_per_gpu = cfg.TEST.BATCH_SIZE_PER_GPU
        shuffle = False
        num_workers = min(cfg.TEST.BATCH_SIZE_PER_GPU, cfg.TRAIN.WORKERS)
    elif split == 'all':
        images_per_gpu = 5
        shuffle = False
        num_workers = min(cfg.TEST.BATCH_SIZE_PER_GPU, cfg.TRAIN.WORKERS)
    else:
        images_per_gpu = 1
        shuffle = False
        num_workers = 0

    dataset = build_dataset(cfg, split, output_mesh=output_mesh)

    if output_mesh:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=images_per_gpu,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=_seed_worker,
            collate_fn=mesh_collate_fn,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=images_per_gpu,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=_seed_worker,
        )

    return data_loader
