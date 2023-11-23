from dataclasses import dataclass

@dataclass
class Generalparams:
    pretrained_weights: str
    test_model_epoch: int
    num_workers: int
    accelerator: str
    devices: list
    log_every_n_steps: int

    def __str__(self):
        output = 'General params:\n'
        output += f'\tUse CPU: {self.accelerator}\n' 
        output += f'\tGPUs to be used: {self.devices}\n' 
        output += f'\tNum workers: {self.num_workers}\n' 
        output += f'\tPretrained weights: {self.pretrained_weights}\n' 
        output += f'\tTest model every n epochs: {self.test_model_epoch}\n'
        output += f'\tLog training info every n steps: {self.log_every_n_steps}\n'
        return output

@dataclass
class Dataparams:
    source_dataset_name: str
    target_dataset_name: str
    model_input_size: int
    prepare_data_per_node: bool

    def __str__(self):
        output = 'Dataparams:\n'
        output += f'\tSource dataset name: {self.source_dataset_name}\n' 
        output += f'\tTarget dataset name: {self.target_dataset_name}\n' 
        output += f'\tModel input img size: ({self.model_input_size},{self.model_input_size})\n'
        output += f'\tPrepapare data per node: {self.prepare_data_per_node}'
        return output

@dataclass
class Hyperparams:
    lr: float
    batch_size: int
    adam_beta_1: float
    adam_beta_2: float
    epochs: int
    classification_model: str
    z_dim: int
    gen_loss_weight: float
    disc_loss_weight: float
    task_loss_weight: float
    gen_res_blocks: int
    weights_decay: float
    adv_loss: str

    def __str__(self):
        output = 'Hyperparams:\n'
        output += f'\tEpochs: {self.epochs}\n' 
        output += f'\tZ dim: {self.z_dim}\n' 
        output += f'\tLearning rate: {self.lr}\n'
        output += f'\tBath size: {self.batch_size}\n' 
        output += f'\tAdam beta 1: {self.adam_beta_1}\n' 
        output += f'\tAdam beta 2: {self.adam_beta_2}\n' 
        output += f'\tClassification model: {self.classification_model}\n' 
        output += f'\tGenerator loss weight: {self.gen_loss_weight}\n' 
        output += f'\tDiscriminator loss weight: {self.disc_loss_weight}\n' 
        output += f'\tTask loss weight: {self.disc_loss_weight}\n' 
        output += f'\tGen residual blocks: {self.gen_res_blocks}\n' 
        output += f'\tWeights decay: {self.weights_decay}\n' 
        output += f'\tAdversarial loss: {self.adv_loss}\n' 
        return output

class Configuration:
    def __init__(self, hyperparams: Hyperparams, dataparams: Dataparams, generalparams: Generalparams, verbose: bool=True) -> None:
        """
        TODO
        """
        self.hyperparams = hyperparams
        self.dataparams = dataparams
        self.general = generalparams

        if verbose:
            print(self.hyperparams)
            print(self.dataparams)
            print(self.general)

        return
