import torch
import itertools
import training
import torchvision
import torchmetrics
import torch.nn          as nn
import pytorch_lightning as torch_lightning
from models                 import Generator, Discriminator
from training.configuration import Configuration

FIXED_SRC_IMGS_IDX = 0
FIXED_TGT_IMGS_IDX = 1
FIXED_NOISE_IDX    = 2

GEN_CLS_OPT        = 0
DISC_OPT           = 1

class PixelDA(torch_lightning.LightningModule):

    def __init__(
        self,
        source_imgs_chs: int,
        target_imgs_chs: int,
        img_size: int,
        num_classes: int,
        conf: Configuration,
        fixed_data: list,
        adv_loss: str) -> None:
        """
        Class of the PixelDA model described in the original paper.
        Inputs:
            >> source_imgs_chs: (int) Number of channels in the source images.
            >> target_imgs_chs: (int) Number of channels in the target images.
            >> img_size: (int) Number of channels in the images (source and target domain images must match in size).
            >> num_classes: (int) Number of classes present in the dataset.
            >> config: (Configuration) Configuration for the training process and some parameters of the model.
            >> fixed_data: (torch.tensor) Fixed data to show the evolution of the generated images during training.
            >> adv_loss: (str) Loss function to be used during the training process for the GAN network.
        Attributes:
            >> conf:(Configuration) Configuration for the training process and some parameters of the model.
            >> generator: (nn.Module) Generator model.
            >> discriminator: (nn.Module) Discriminator model.
            >> adv_loss: (nn.BCELoss)
            >> task_loss: (nn.CELoss)
            >> accuracy: (torchmetrics.Accuracy) 
        """
        super().__init__()
        self.conf = conf

        # Set to manual optimization as we are working with multiple optimizers at the same time
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator     = Generator(source_imgs_chs, target_imgs_chs, conf.hyperparams.z_dim, img_size, conf.hyperparams.gen_res_blocks)
        self.discriminator = Discriminator(target_imgs_chs, img_size)
        self.classifier    = training.utils.load_classifier(
            conf.hyperparams.classification_model,
            target_imgs_chs,
            img_size,
            num_classes)

        init_weights(self.generator)
        init_weights(self.discriminator)
        init_weights(self.classifier)

        """
        MSE Emphasizes Pixel-Level Accuracy: MSE loss measures the pixel-wise difference 
        between the generated image and the target image. It tends to emphasize the optimization 
        of individual pixel values, which may lead to blurry or less visually appealing generated images.
        GANs are typically used for tasks like image generation, where capturing high-level structures,
        textures, and details is crucial.
        """
        self.adv_loss = None
        if adv_loss != 'bce' and adv_loss != 'mse':
            raise ValueError(f'The only loss functions that can be selected are bce or mse, not {adv_loss}')
        elif adv_loss == 'bce':
            self.adv_loss = torch.nn.BCELoss()
        elif adv_loss == 'mse':
            self.adv_loss = torch.nn.MSELoss()

        self.task_loss  = torch.nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.fixed_data = fixed_data 

        return

    def training_step(self, batch: list, batch_idx: int):
        """
        Peforms a one training step for the Generator, Discriminator and Classifier models
        Inputs:
            >> batch: (list of torch.tensors) It contains the [source_dom_img, targe_dom_img].
            >> batch_idx: (int) Index of the current batch.
        Outputs:
            >> loss_dict: (dict) Dict of loss containing 'gen_loss', 'disc_loss' and 'task_loss'.
        """
        gen_loss, disc_loss, task_loss = self.__common_step__(batch, batch_idx, train=True)
        loss_dict = {
            "train_gen_loss": gen_loss,
            "train_disc_loss": disc_loss,
            "train_task_loss": task_loss}
        # on_step = True --> Logs the metric at the current step
        # on_epoch = True --> Automatically accumlates and logs at the end of the epoch
        self.log_dict(loss_dict, on_step=False, on_epoch=True)

        ## Evaluate cls performance on real tgt domain
        src_dom_imgs, tgt_dom_imgs, source_dom_labels, tgt_dom_labels = batch
        cls_preds = self.classifier(tgt_dom_imgs)
        train_acc = self.accuracy(cls_preds, tgt_dom_labels) 
        # on_step = True --> Logs the metric at the current step
        # on_epoch = True --> Automatically accumlates and logs at the end of the epoch
        self.log('train_acc', train_acc, on_step=False, on_epoch=True)

        return loss_dict 

    def validation_step(self, batch: list, batch_idx: int):
        """
        Peforms a one validation step for the Generator, Discriminator and Classifier models
        Inputs:
            >> batch: (list of torch.tensors) It contains the [source_dom_img, targe_dom_img].
            >> batch_idx: (int) Index of the current batch.
        Outputs:
            >> loss_dict: (dict) Dict of loss containing 'gen_loss', 'disc_loss' and 'task_loss'.
        """
        gen_loss, disc_loss, task_loss = self.__common_step__(batch, batch_idx, train=False)
        loss_dict = {
            "val_gen_loss": gen_loss,
            "val_disc_loss": disc_loss,
            "val_task_loss": task_loss}
        # on_step = True --> Logs the metric at the current step
        # on_epoch = True --> Automatically accumlates and logs at the end of the epoch
        self.log_dict(loss_dict, on_step=False, on_epoch=True)

        ## Evaluate cls performance on real tgt domain
        src_dom_imgs, tgt_dom_imgs, source_dom_labels, tgt_dom_labels = batch
        cls_preds = self.classifier(tgt_dom_imgs)
        val_acc = self.accuracy(cls_preds, tgt_dom_labels) 
        # on_step = True --> Logs the metric at the current step
        # on_epoch = True --> Automatically accumlates and logs at the end of the epoch
        self.log('val_acc', val_acc, on_step=False, on_epoch=True)

        return loss_dict 

    def test_step(self, batch: list, batch_idx: int):
        """
        Peforms a one test step for the Generator, Discriminator and Classifier models
        Inputs:
            >> batch: (list of torch.tensors) It contains the [source_dom_img, targe_dom_img].
            >> batch_idx: (int) Index of the current batch.
        Outputs: None
        """

        ## Evaluate cls performance on real tgt domain
        src_dom_imgs, tgt_dom_imgs, source_dom_labels, tgt_dom_labels = batch
        cls_preds = self.classifier(tgt_dom_imgs)
        test_acc = self.accuracy(cls_preds, tgt_dom_labels) 
        # on_step = True --> Logs the metric at the current step
        # on_epoch = True --> Automatically accumlates and logs at the end of the epoch
        self.log('test_acc', test_acc, on_step=False, on_epoch=True)
        return

    def configure_optimizers(self,):
        """
        Configures the optimizer that will be used during the training process
        Inputs: None
        Outputs:
            >> optimizers_list: (list) Contains the optimizers for the generator, discriminator and classifier.
            >> lr_schedulers_list: (list) Contains the lr schedulers for the generator, discriminator and classifier.
        """
        generator_cls_optim = torch.optim.Adam(
            itertools.chain(
                self.generator.parameters(),
                self.classifier.parameters()),
            lr=self.conf.hyperparams.lr,
            weight_decay=self.conf.hyperparams.weights_decay,
            betas=(self.conf.hyperparams.adam_beta_1, self.conf.hyperparams.adam_beta_2))

        discriminator_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.conf.hyperparams.lr,
            weight_decay=self.conf.hyperparams.weights_decay,
            betas=(self.conf.hyperparams.adam_beta_1,
                   self.conf.hyperparams.adam_beta_2))

        gen_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                generator_cls_optim,
                step_size=20000,
                gamma=0.95)

        disc_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                generator_cls_optim,
                step_size=20000,
                gamma=0.95)

        optimizers_list = [generator_cls_optim, discriminator_optim]
        lr_schedulers_list = [gen_lr_scheduler, disc_lr_scheduler]

        return optimizers_list, lr_schedulers_list

    def __common_step__(self, batch: list, batch_idx: int, train: bool=True):
        """
        Peforms a common step for the training, validation and test data splits.
        Inputs:
            >> batch: (list of torch.tensors) It contains the [source_dom_img, targe_dom_img].
            >> batch_idx: (int) Index of the current batch.
            >> train: (bool) Indicates whether this step is running for training the model or not.
        Outputs:
            >> loss_dict: (dict) Containg 'gen_loss', 'disc_loss' and 'task_loss'
        """
        source_dom_imgs, real_target_dom_imgs, source_dom_labels, _ = batch
        batch_size = source_dom_imgs.shape[0]
        noise = 2 * torch.rand(batch_size, self.conf.hyperparams.z_dim, device=self.device) - 1 # Noise from a Normal dist [-1,1]

        disc_loss = self.train_discriminator(source_dom_imgs, real_target_dom_imgs, noise, train)
        gen_loss, task_loss = self.train_generator(source_dom_imgs, source_dom_labels, noise, train)

        return gen_loss, disc_loss, task_loss

    def train_generator(self, source_dom_imgs: torch.tensor, source_dom_labels: torch.tensor, noise: torch.tensor, train: bool=True):
        """
        Trains the generator model
        Inputs:
            >> source_dom_imgs: (torch.tensor [batch, img_chs, img_size, img_size]) Images of the source domain.
            >> source_dom_labels: (torch.tensor [batch, 1]) Labels of the images in the source domain.
            >> noise: (torch.tensor [batch, config.hyperparams.z_dim]) Noise needed for the generator.
            >> train: (bool) Wether to run the backpropagation step and modify the weights of the model.
        Outputs:
            >> gen_loss: (float) Loss value in the current batch for the generator model.
            >> task_loss: (float) Task (classification) loss for the current batch
        """
        # Train generator: min log(1-D(G(x_s)) <--> max log(D(G(x_s)))
        fake_disc_output = self.discriminator(self.generator(source_dom_imgs, noise))
        gen_loss = self.adv_loss(fake_disc_output, torch.ones_like(fake_disc_output))
        task_loss = (self.task_loss(self.classifier(self.generator(source_dom_imgs, noise)), source_dom_labels) + self.task_loss(self.classifier(source_dom_imgs), source_dom_labels)) / 2.0
        dom_loss = self.conf.hyperparams.gen_loss_weight * gen_loss + self.conf.hyperparams.task_loss_weight * task_loss

        if train:
            self.optimizers()[GEN_CLS_OPT].zero_grad()
            self.manual_backward(dom_loss)
            self.optimizers()[GEN_CLS_OPT].step()
            self.lr_schedulers()[GEN_CLS_OPT].step()

        return gen_loss, task_loss

    def train_discriminator(self, source_dom_imgs: torch.tensor, real_target_dom_imgs: torch.tensor, noise: torch.tensor, train: bool=True) -> float:
        """
        Trains the discriminator model
        Inputs: 
            >> source_dom_imgs: (torch.tensor [batch, img_chs, img_size, img_size]) Images of the source domain.
            >> real_target_dom_imgs: (torch.tensor [batch, img_chs, img_size, img_size]) Realimages of target domain.
            >> noise: (torch.tensor [batch, config.hyperparams.z_dim]) Noise needed for the generator.
            >> train: (bool) Wether to run the backpropagation step and modify the weights of the model.
        Outputs:
            >> disc_loss: (float) Loss value in the current batch for the discriminator model.
        """
        # Train discriminator: max log(D(x_t)) + log(1-D(G(x_s)))
        real_disc_output = self.discriminator(real_target_dom_imgs)
        loss_real_disc   = self.adv_loss(real_disc_output, torch.ones_like(real_disc_output))

        fake_disc_output = self.discriminator(self.generator(source_dom_imgs, noise))
        loss_fake_disc   = self.adv_loss(fake_disc_output, torch.zeros_like(real_disc_output))
        disc_loss = self.conf.hyperparams.disc_loss_weight * ((loss_real_disc + loss_fake_disc) / 2.0)

        if train:
            self.optimizers()[DISC_OPT].zero_grad()
            self.manual_backward(disc_loss)
            self.optimizers()[DISC_OPT].step()
            self.lr_schedulers()[DISC_OPT].step()

        return disc_loss / self.conf.hyperparams.disc_loss_weight

    def on_train_epoch_end(self):
        if self.current_epoch % 1 == 0:
            fixed_noise = self.fixed_data[FIXED_NOISE_IDX].to(self.device)
            source_imgs = self.fixed_data[FIXED_SRC_IMGS_IDX].to(self.device)
            target_imgs = self.fixed_data[FIXED_TGT_IMGS_IDX].to(self.device)

            source_imgs_grid = torchvision.utils.make_grid(source_imgs)
            real_target_imgs_grid = torchvision.utils.make_grid(target_imgs)
            fake_target_imgs_grid = torchvision.utils.make_grid(self.generator(source_imgs, fixed_noise))

            self.logger.experiment.add_image('src imgs', source_imgs_grid, self.current_epoch)
            self.logger.experiment.add_image('real tgt imgs', real_target_imgs_grid, self.current_epoch)
            self.logger.experiment.add_image('fake tgt imgs', fake_target_imgs_grid, self.current_epoch)

        return

    def predict_step(self, batch: list, batch_idx: int):
        """
        TODO
        Inputs:
            >> : ()
        Outputs:
            >> : ()
        """
        return

def init_weights(model: torch.nn.Module):
    """
    Initialises the weights of the input model following a zero centered Gaussian with stddev 0.02.
    Inputs:
        >> model: (torch.nn.Module)
    Outputs: None
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
    return


