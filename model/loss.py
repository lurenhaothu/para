import torch

def BCELoss(mask, pred, w_map):
    return torch.mean(- mask * torch.log(pred + 1e-7) - (1 - mask) * torch.log(1 - pred + 1e-7))

def BCE_withClassBalance(mask, pred, w_map): #20570595.0 84287005.0
    count1 = 20570595
    count0 = 84287005
    w1 = (count1 + count0) / count1 / 2
    w0 = (count1 + count0) / count0 / 2
    return torch.mean(- w1 * mask * torch.log(pred + 1e-7) - w0 * (1 - mask) * torch.log(1 - pred + 1e-7))

def DiceLoss(mask, pred, w_map):
    sum_p12 = torch.sum(torch.pow(mask, 2))
    sum_g12 = torch.sum(torch.pow(pred, 2))
    sum_p1g1 = torch.sum(mask * pred)
    B, C, H, W = mask.shape
    return - 2 * sum_p1g1 / (sum_p12 + sum_g12 + 1e-7)

def Unet_Weight_BCE(mask, pred, w_map):
    count1 = 20570595
    count0 = 84287005
    w1 = (count1 + count0) / count1 / 2
    w0 = (count1 + count0) / count0 / 2
    return torch.mean(- (w_map + w1) * mask * torch.log(pred + 1e-7) - w0 * (1 - mask) * torch.log(1 - pred + 1e-7))




