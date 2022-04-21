#from: github.com/smausenberger/SPaiNN/blob/main/spainn/
import schnetpack as spk
from schnetpack.data import *
import schnetpack.transform as trn
import os
from typing import Optional, List, Dict, Tuple, Union
import torch
import shutil
import pytorch_lightning as pl
from pytorch_lightning.utilities.enums import DeviceType
import fasteners
from copy import copy
from .multidatamodule import *


class SCHNARC(MultiDataModule):
    """
    Load and setup data for training with SPaiNN
    """
    energy = "energy"
    forces = "forces"
    
    def __init__(
        self,
        datapath: str,
        n_states: int,
        batch_size: int,
        num_train: Optional[int] = None,
        num_val: Optional[int] = None,
        num_test: Optional[int] = None,
        split_file: Optional[str] = "split.npz",
        format: Optional[AtomsDataFormat] = AtomsDataFormat.ASE,
        load_properties: Optional[List[str]] = None,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        transforms: Optional[List[torch.nn.Module]] = None,
        train_transforms: Optional[List[torch.nn.Module]] = None,
        val_transforms: Optional[List[torch.nn.Module]] = None,
        test_transforms: Optional[List[torch.nn.Module]] = None,
        num_workers: int = 2,
        num_val_workers: Optional[int] = None,
        num_test_workers: Optional[int] = None,
        property_units: Optional[Dict[str, str]] = None,
        distance_unit: Optional[str] = None,
        data_workdir: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            datapath: path to database (ASE format!)
            batch_size: batch size
            num_train: number of training examples
            num_val: number of validation samples
            num_test: number of test examples
            split_file: path to npz file, if the file exists train, val and test samples are loaded from the existing file.
                        An existing split_file can be a source of error if you train new data.
            format: dataset format
            load_properties: subset of properties to load
            transforms: Transform applied to each system separately before batching.
            train_transforms: Overrides transform_fn for training.
            val_transforms: Overrides transform_fn for validation.
            test_transforms: Overrides transform_fn for testing.
            num_workers: Number of data loader workers.
            num_val_workers: Number of validation data loader workers (overrides num_workers).
            num_test_workers: Number of test data loader workers (overrides num_workers).
            distance_unit: Unit of the atom positions and cell as a string (Ang, Bohr, ...).
            data_workdir: Copy data here as part of setup, e.g. cluster scratch for faster performance.
        """
        super().__init__(
            datapath=datapath,
            n_states=n_states,
            batch_size=batch_size,
            num_train=num_train,
            num_val=num_val,
            num_test=num_test,
            split_file=split_file,
            format=format,
            load_properties=load_properties,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            transforms=transforms,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            num_workers=num_workers,
            num_val_workers=num_val_workers,
            num_test_workers=num_test_workers,
            property_units=property_units,
            distance_unit=distance_unit,
            data_workdir=data_workdir,
            **kwargs,
        )
        self.prepare_data()
        self.setup_multistate()

    def prepare_data(self):
        dataset = load_dataset(self.datapath, self.format)
    
    

