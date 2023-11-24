import os
import argparse
import pytorch_lightning as pl
from models.pixelda            import PixelDA
from datasets.data_modules     import Source2TargetDataModule 
from training.utils            import load_configuration
from pytorch_lightning.loggers import TensorBoardLogger

SOURCE_IMGS_IDX = 0
TARGET_IMGS_IDX = 1

if __name__ == '__main__':

    #################
    # CONFIGURATION #
    #################

    parser = argparse.ArgumentParser(description='Arguments for pix2pix testing.')
    parser.add_argument( '--config-file',        type=str, required=True, help='Path to the configuration file.' )
    args = parser.parse_args()

    config = load_configuration(args.config_file)

    ##################
    # MODEL and DATA #
    ##################

    data_module = Source2TargetDataModule(config)

    model = pl.LightningModule()
    if config.general.pretrained_weights == None:
        raise ValueError(f'The path to the pretrained_weights must be especified in the config file.')

    print(f'===========================================================================')
    print(f'Loading pretrained weights from: {config.general.pretrained_weights}')
    model = PixelDA.load_from_checkpoint(config.general.pretrained_weights)
    print(f'===========================================================================')

    tb_logger = TensorBoardLogger(
        save_dir='runs/tensorboard',
        name='PixelDA',
        version=f"{config.dataparams.source_dataset_name}_2_{config.dataparams.target_dataset_name}_{config.hyperparams.adv_loss}Loss",
        default_hp_metric=False)

    trainer = pl.Trainer(
        logger=tb_logger,
        accelerator=config.general.accelerator,
        devices=config.general.devices,
        max_epochs=config.hyperparams.epochs,
        log_every_n_steps=config.general.log_every_n_steps,)

    trainer.test(model, data_module)
