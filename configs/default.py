'''Default Hyperparameter configuration.'''

import ml_collections


def get_config():
    '''Get the default hyperparameter configuration.'''
    config = ml_collections.ConfigDict()

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.name = 'InvalidModel'

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.optimizer = 'adamw'
    training.learning_rate = 0.001
    training.lr_schedule = 'const'  # 'cosine'/'cos', 'const'
    training.weight_decay = 0.0
    training.adam_b1 = 0.9
    training.adam_b2 = 0.999
    # training.warmup_epochs = 200
    training.momentum = 0.9
    training.batch_size = 512
    training.shuffle_buffer_size = 16 * 1024
    training.prefetch = 10
    training.num_epochs = 4000
    training.log_per_step = 100
    training.log_per_epoch = -1
    # training.eval_per_epoch = 100
    training.visualize_per_epoch = 1
    training.checkpoint_per_epoch = 200
    training.steps_per_eval = -1
    training.half_precision = False
    training.seed = 0  # init random seed
    training.wandb = True
    training.noise_level = 0.0
    return config
