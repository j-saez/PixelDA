import os
import torchvision
from torch.utils.data import Dataset
from training.configuration import Configuration

MNISTM_IMG_CHS = 1
IMG = 0
LABEL = 1

class MNISTDataset(Dataset):

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

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(config.dataparams.model_input_size, antialias=True),
            torchvision.transforms.ToTensor(), # pixels value: [0, 255] -> [0, 1]
            torchvision.transforms.Normalize([0.5 for _ in range(MNISTM_IMG_CHS)],[0.5 for _ in range(MNISTM_IMG_CHS)]), # pixels value: [0, 1] -> [-1, 1] ])
        ])

        self.dataset = torchvision.datasets.MNIST(
            root=f'{os.getcwd()}/datasets/data/',
            train=split=='train',
            transform=transform,
            download=True)

    def __len__(self,):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Returns the label and the image as a torch tensor.
        Inputs:
            >> idx: (int) Idx of the image and the label to be loaded.
        Outputs:
            >> img: (torch.tensor [3, img_w, img_h])
            >> label: (torch.tensor [1])
        """
        img = self.dataset[idx][IMG].repeat(3,1,1)
        label = self.dataset[idx][LABEL]
        return img, label
