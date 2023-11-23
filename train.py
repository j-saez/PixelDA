import os
import argparse
import pytorch_lightning as pl
from models.pixelda            import PixelDA
from datasets.data_modules     import Source2TargetDataModule 
from pytorch_lightning.loggers import TensorBoardLogger
from training.utils            import load_configuration

SOURCE_IMGS_IDX = 0
TARGET_IMGS_IDX = 1

if __name__ == '__main__':

    #################
    # CONFIGURATION #
    #################

    parser = argparse.ArgumentParser(description='Arguments for pix2pix training.')
    parser.add_argument( '--config-file',        type=str, required=True, help='Path to the configuration file.' )
    args = parser.parse_args()

    config = load_configuration(args.config_file)

    ##################
    # MODEL and DATA #
    ##################

    data_module = Source2TargetDataModule(config)

    model = pl.LightningModule()
    if config.general.pretrained_weights == None:
        model = PixelDA(
            # Src and tgt datasets must much the number or channels.
            source_imgs_chs=3,
            target_imgs_chs=3,
            img_size=config.dataparams.model_input_size,
            num_classes=data_module.get_total_classes(),
            conf=config,
            fixed_data=data_module.get_fixed_data(total_imgs=8),
            adv_loss=config.hyperparams.adv_loss)

    else:
        print(f'pretrained weights: {config.general.pretrained_weights}')
        model = PixelDA.load_from_checkpoint(config.general.pretrained_weights)

    #############
    # CALLBACKS #
    #############

    checkpoint_filename = f"PixelDA_{config.dataparams.source_dataset_name}2{config.dataparams.target_dataset_name}_{config.hyperparams.adv_loss}Loss"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f'{os.getcwd()}/runs/checkpoints',
        filename=checkpoint_filename+'_e{epoch}_mClsAcc{val_acc:.2f}',
        save_top_k = 1,
        monitor='val_acc',
        mode='max',
        verbose=True,
        save_on_train_epoch_end=False) # Save after validation

    tb_logger = TensorBoardLogger(
        save_dir='runs/tensorboard',
        name='PixelDA',
        version=f"{config.dataparams.source_dataset_name}_2_{config.dataparams.target_dataset_name}_{config.hyperparams.adv_loss}Loss",
        default_hp_metric=False)

    #########
    # Train #
    #########

    trainer = pl.Trainer(
        callbacks = [checkpoint_callback],
        logger=tb_logger,
        accelerator=config.general.accelerator,
        devices=config.general.devices,
        max_epochs=config.hyperparams.epochs,
        log_every_n_steps=config.general.log_every_n_steps,
        )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)
