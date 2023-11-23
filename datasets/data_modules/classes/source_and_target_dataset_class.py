from torch.utils.data import Dataset
from training.configuration import Configuration
from training.utils import load_dataset

class Source2TargetDataset(Dataset):

    def __init__(self, config: Configuration, split: str)  -> None:
        """
        Loads the source and target domain datasets.
        Inputs:
            >> source_dataset_name: (str) Name of the source domain dataset.
            >> target_dataset_name: (str) Name of the target domain dataset.
            >> split: (str) Whether the dataset will be used during the training or test phase. (Choose between train or test)
        Attributes:
            >> source_dataset: (torch.utils.data.Dataset) Source domain dataset.
            >> target_dataset: (torch.utils.data.Dataset) Target domain dataset.
        """
        super().__init__()
        self.source_dataset = load_dataset(config.dataparams.source_dataset_name, split, config)
        self.target_dataset = load_dataset(config.dataparams.target_dataset_name, split, config)
        return

    def __len__(self):
        return max([len(self.source_dataset), len(self.target_dataset)])

    def __getitem__(self, idx: int):
        """
        Returns the idx-th image of the MNIST dataset and the idx-th image of the MNIST-M Dataaset.
        Inputs:
            >> idx: (int) Index of the image to be loaded.
        Outputs:
            >> source_img: (torch.tensor [img_chs, img_w, img_h]) Source domain loaded image.
            >> target_img: (torch.tensor [img_chs, img_w, img_h]) Target domain loaded image.
            >> source_dom_label: (torch.tensor [1,]) Source domain image label.
            >> target_dom_label: (torch.tensor [1,]) Target domain image labels.
        """
        source_img, source_dom_label = self.source_dataset[idx % len(self.source_dataset)]
        target_img, target_dom_label = self.target_dataset[idx % len(self.target_dataset)]
        return source_img, target_img, source_dom_label, target_dom_label
