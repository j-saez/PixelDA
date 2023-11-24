from training.configuration import Hyperparams, Dataparams, Generalparams, Configuration

hyperparams = Hyperparams(
    lr = 1e-3,                                  # learning rate
    z_dim = 10,                                 # Dimension for the random noise
    batch_size = 32,                            # Learning rate
    adam_beta_1 = 0.5,                          # Beta 1 for the Adam optimizer
    adam_beta_2 = 0.999,                        # Beta 2 for the Adam optimizer
    epochs = 50,                                # Total number of training epochs
    classification_model = "mnistm_classifier", # At the moment only mnistm_classifier can be selected 
    gen_loss_weight = 10,                       # Generator loss weight
    disc_loss_weight = 1,                       # Discriminator loss weight
    task_loss_weight = 1,                       # Classifier loss weight
    gen_res_blocks = 6,                         # Number residual blocks in the generator
    weights_decay = 1e-5,                       # Weight decay for the models parameters (from the paper)
    adv_loss = 'mse',                           # Loss function to be used for training the GAN network. Choose between mse or bce
)

dataparams = Dataparams(
    source_dataset_name = "mnist",  # Source domain dataset
    target_dataset_name = "mnistm", # Traget domain dataset
    model_input_size = 28,          # Input size for the model
    prepare_data_per_node = True,   # Get the data at each node where the model will be trained.
)

generalparams = Generalparams(
    accelerator = 'cuda',      # Hardware acceleration for training
    num_workers=12,            # Number of workers
    devices = [0,],            # Comma separated list containing the gpu/s where the code will be executed. (Example 1 = 0 -- Example 2 = 0,1)
    pretrained_weights = 'runs/checkpoints/PixelDA_mnist2mnistm_mseLoss_eepoch=40_mClsAccval_acc=0.97.ckpt', # Path to the pretrained_weights
    test_model_epoch = 5,      # Sets the number of epochs to be trained before testing the model.
    log_every_n_steps = 500,   # Sets the number of steps to be executed before logging training info.
)

configuration = Configuration(hyperparams, dataparams, generalparams, verbose=True)
