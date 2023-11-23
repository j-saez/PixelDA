import os
import h5py
import torch
import torchvision
from PIL              import Image
from torch.utils.data import Dataset
from training.configuration import Configuration

MNIST_IMG_SIZE = 28
USPS_IMG_CHS = 1
USPS_PATH = f'{os.getcwd()}/datasets/data/USPS/usps.h5'

class USPSDataset(Dataset):

    def __init__(self, split: str, config: Configuration) -> None:
        """
        Loads the data the USPS Dataset
        Inputs:
            >> split: (str) train or test.
            >> config: (Configuration) Configuration of the training process.
        Attributes:
            >> imgs_chs: (int) Number of channesl in the images of the dataset.
            >> num_classes: (int) Total number of classes present in the dataset.
            >> images: (torch.tensor [total_imgs, imgs_chs, imgs_h, imgs_w]) Tensor containing the images of the dataset.
        """
        super().__init__()
        self.num_classes = 10

        transforms = torchvision.transforms.Compose([
            # The images on the h5 file already come within [0,1] values
            torchvision.transforms.Normalize([0.5 for _ in range(USPS_IMG_CHS)],[0.5 for _ in range(USPS_IMG_CHS)]), # pixels value: [0, 1] -> [-1, 1] ]) ])
            torchvision.transforms.Resize(config.dataparams.model_input_size, antialias=True),
        ])

        with h5py.File(USPS_PATH, 'r') as hf:
            images = torch.tensor(hf.get(split).get('data')[:]).view(-1,1,16,16)
            self.images = transforms(images)
            self.labels = torch.tensor(hf.get(split).get('target')[:])

        return

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        """
        Returns the idx-th image of the USPS dataset.
        Inputs:
            >> idx: (int) Index of the image to be loaded.
        Outputs:
            >> img: (torch.tensor [img_chs, img_w, img_h]) Loaded image
        """
        return self.images[idx], self.labels[idx]
