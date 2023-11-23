import os
import torch
import pickle
import torchvision
from torch.utils.data import Dataset
from training.configuration import Configuration

MNISTM_IMG_CHS = 3

class MNISTMDataset(Dataset):

    def __init__(self, split: str, config: Configuration) -> None:
        """
        Loads the data the MNISTM Dataset
        Inputs:
            >> split: (str) train or test.
            >> config: (Configuration) Configuration of the training process.
        Attributes:
            >> imgs_chs: (int) Number of channesl in the images of the dataset.
            >> num_classes: (int) Total number of classes present in the dataset.
            >> imgs_filenames: (List[str]) List containing the path + file name of the images from the dataset.
            >> transforms: (torchvision.transforms) Transformations to be applied to the images before being loaded.
        """
        super().__init__()

        self.num_classes = 10

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(config.dataparams.model_input_size, antialias=True),
            torchvision.transforms.ToTensor(), # pixels value: [0, 255] -> [0, 1]
            torchvision.transforms.Normalize([0.5 for _ in range(MNISTM_IMG_CHS)],[0.5 for _ in range(MNISTM_IMG_CHS)]), # pixels value: [0, 1] -> [-1, 1] ])
        ])

        f = open(f'{os.getcwd()}/datasets/data/MNIST-M/keras_mnistm.pkl', 'rb')
        data = pickle.load(f, encoding='bytes')
        if   split == 'train': imgs = data[b"train"]
        elif split == 'test':  imgs = data[b"test"]

        self.imgs   = imgs
        self.labels = torchvision.datasets.MNIST(root=f'{os.getcwd()}/datasets/data/', train=split=='train', download=True).targets

    def __len__(self,):
        return len(self.imgs)

    def __getitem__(self, idx: int):
        return self.transforms(self.imgs[idx]), self.labels[idx]
