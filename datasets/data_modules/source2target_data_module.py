import torch
import pytorch_lightning as pl
from datasets.data_modules.classes import Source2TargetDataset
from torch.utils.data import random_split, DataLoader
from training.configuration import Configuration
from training.utils import download_dataset

class Source2TargetDataModule(pl.LightningDataModule):

    def __init__(self, config: Configuration) -> None:
        super().__init__()

        # TODO: Research a bit more what prepare_data_per_node does.
        self.prepare_data_per_node = config.dataparams.prepare_data_per_node
        self.config = config

        return

    def prepare_data(self) -> None:
        """
        Downloads the data so we have it to disc
        """
        # this is for single gpu
        download_dataset(self.config.dataparams.source_dataset_name)
        download_dataset(self.config.dataparams.target_dataset_name)
        return

    def setup(self, stage: str) -> None:
        """
        Loads the data downloaded in prepate_data as a pytorch dataset class object
        """
        # this is for multiple gpu as it is called in every gpu on the system.

        if stage == 'fit' or stage == None:
            train_dataset = Source2TargetDataset( self.config, 'train')
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(train_dataset, [train_size, val_size])

        elif stage == 'test':
            self.test_dataset = Source2TargetDataset( self.config, 'test')

        return

    def train_dataloader(self,):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.hyperparams.batch_size,
            num_workers=self.config.general.num_workers,
            shuffle=True,)

    def val_dataloader(self,):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.hyperparams.batch_size,
            num_workers=self.config.general.num_workers,
            shuffle=False,)

    def test_dataloader(self,):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.hyperparams.batch_size,
            num_workers=self.config.general.num_workers,
            shuffle=False,)

    def get_imgs_chs(self,):
        dataset = Source2TargetDataset(
            self.config,
            'train')
        source_imgs_chs = dataset.source_dataset.imgs_chs
        target_imgs_chs = dataset.target_dataset.imgs_chs
        return source_imgs_chs, target_imgs_chs

    def get_total_classes(self,):
        dataset = Source2TargetDataset(
            self.config,
            'train')
        total_classes = dataset.source_dataset.num_classes
        return total_classes

    def get_fixed_data(self, total_imgs: int):
        self.setup(stage='fit')
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=total_imgs,
            num_workers=self.config.general.num_workers,
            shuffle=False)

        dataiter = iter(dataloader)
        source_imgs, target_imgs, _, _ = next(dataiter)
        fixed_noise = torch.randn(total_imgs, self.config.hyperparams.z_dim)

        return [source_imgs, target_imgs, fixed_noise]
