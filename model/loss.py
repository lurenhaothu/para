import torch

def BCELossWithClassWeight(weight):
    def loss(mask, pred):
        return - weight[1] * mask * torch.log(pred[:, 1:2, :, :]) - weight[0] * (1 - mask) * torch.log(pred[:, 0:1, :, :])
    return loss

def BCELoss():
    return torch.nn.BCELoss()

def HomeMadeBCE():
    def loss(mask, pred):
        return - mask * torch.log(pred[:, 1:2, :, :]) - (1 - mask) * torch.log(pred[:, 0:1, :, :])
    return loss