import os
import importlib
import zipfile
import gdown
from models import MNISTMClassifier, LineModClassifier
from training.configuration import Configuration
import datasets.data_modules.classes as dataset_classes

DATASETS_PATH = f'{os.getcwd()}/datasets/data'
MNISTM_URL = 'https://drive.google.com/uc?id=1Nr23iNTIwn2BkuQbrE7tgL2U84qVna0j'
MNIST_URL = 'https://drive.google.com/uc?id=18gJ9LsJB4d1fxYjzVJB-YgZ1_dBMqiIp'
USPS_URL = 'https://drive.google.com/uc?id=1oeH-4OMLtEJIEIEq_YZSmMJZvm8Oft9r'
AVAILABLE_DATASETS = ['mnist', 'mnistm', 'usps']
CLASSIFIERS_AVAILABLE = ['mnistm_classifier', 'linemod_classifier']

def load_dataset(dataset_name: str, split: str, config: Configuration):
    """
    Loads the desired dataset
    Inputs:
        >> dataset_name: (Dataparams) Dataset params for training the model.
        >> split: (str) 'train' or 'test'
    Outputs:
        >> dataset: (torch.utils.data.Dataset) Loaded dataset
    """

    dataset = None
    if dataset_name == 'mnist':
        dataset = dataset_classes.MNISTDataset(split, config)
    elif dataset_name == 'mnistm':
        dataset = dataset_classes.MNISTMDataset(split, config)
    elif dataset_name == 'usps':
        dataset = dataset_classes.USPSDataset(split, config)

    return dataset

def download_dataset(dataset_name: str):
    """
    Downloads the dataset and stores it at <root>/datasets/data
    Inputs:
        >> data_params: (Dataparams) Dataset params for training the model.
        >> split: (str) 'train' or 'test'
    Outputs: None
    """
    if dataset_name == 'mnist':
        if not os.path.exists(f'{DATASETS_PATH}/MNIST'):
            gdown.download(MNIST_URL, f'{DATASETS_PATH}/mnist.zip', quiet=True)
            unzip_file(f'mnist.zip')

    elif dataset_name == 'mnistm':
        if not os.path.exists(f'{DATASETS_PATH}/MNIST-M'):
            gdown.download(MNISTM_URL, f'{DATASETS_PATH}/mnistm.zip', quiet=True)
            unzip_file(f'mnistm.zip')

    elif dataset_name == 'usps':
        if not os.path.exists(f'{DATASETS_PATH}/USPS'):
            os.mkdir(f'{DATASETS_PATH}/USPS')
            gdown.download(USPS_URL, f'{DATASETS_PATH}/USPS/usps.h5', quiet=True)
    else:
        raise ValueError(f'{dataset_name} is not a valid dataset. Please choose between: {AVAILABLE_DATASETS}')

    return 

def unzip_file(zip_file: str):
    """
    Unzips the especified file.
    Inputs:
        >> zip_file: (str) Path + file name of the file to be unzipped.
    Outputs: None
    """
    print(f'*******>Extracting {zip_file}')
    with zipfile.ZipFile(f'{DATASETS_PATH}/{zip_file}', 'r') as zip_ref:
        zip_ref.extractall(path=DATASETS_PATH)
        os.remove(f'{DATASETS_PATH}/{zip_file}')
    print(f"*******************> Extracted.")

    return


def load_classifier(classifier_name: str, in_chs: int, img_size: int, num_classes: int):
    """
    Loads the desired classifier model.
    Inputs:
        >> classifier_name: (str) Name of the classifier model to be loaded.
        >> in_chs: (int) Number of channels in the dataset.
        >> img_size: (int) Size of the images present in the dataset.
        >> num_classes: (int) Total classes present in the dataset.
    Outputs:
        >> classifier: (nn.Module) Classifier model.
    """
    classifier = None
    if classifier_name == 'mnistm_classifier':
        classifier = MNISTMClassifier(in_chs, img_size, num_classes)
    elif classifier_name == 'linemod_classifier':
        classifier = LineModClassifier(in_chs, img_size, num_classes)
    else:
        raise ValueError(f'The only valid classifier models are: {CLASSIFIERS_AVAILABLE}')
    return classifier

def load_configuration(config_file_name: str):
    """
    Loads the configuration dict from the especified file.
    The configuration file needs to be stored in the train_configs directory.

    Inputs:
        >> config_file_name: (str) Name of the file containing the configuration.
    Outputs:
        >> configuration: (Configuration) Configuration dict for the training process.

    """
    config_path = os.path.join("train_configs", f"{config_file_name}.py")
    
    spec = importlib.util.spec_from_file_location(config_file_name, config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module.configuration

