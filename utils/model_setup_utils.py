import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

def init_weights(model):
    """Initialize the network parameters"""
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight)
            nn.init.constant_(layer.bias, 0)


def linear_warmup_step_decay_scheduler(
    optimizer, lr_warmup_perc, total_epochs, step_size, gamma
):
    num_warmup_steps = lr_warmup_perc * total_epochs

    def lr_lambda(epoch):
        if epoch < num_warmup_steps:
            return epoch / (lr_warmup_perc * total_epochs)  # Linear warmup (0 to 1)
        else:
            decay_epochs = epoch - num_warmup_steps
            return gamma ** (decay_epochs // step_size)  # Step decay

    return LambdaLR(optimizer, lr_lambda)
