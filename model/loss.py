import torch

class BCELoss(torch.nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, mask, pred, w_map, class_weight, epoch=None):
        return torch.mean(- mask * torch.log(pred + 1e-7) - (1 - mask) * torch.log(1 - pred + 1e-7))

class BCE_withClassBalance(torch.nn.Module):
    def __init__(self):
        super(BCE_withClassBalance, self).__init__()

    def forward(self, mask, pred, w_map, class_weight, epoch=None): #20570595.0 84287005.0
        return torch.mean(- class_weight[:,1:2,:] * mask * torch.log(pred + 1e-7) - class_weight[:,0:1,:] * (1 - mask) * torch.log(1 - pred + 1e-7))

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, mask, pred, w_map, class_weight, epoch=None):
        sum_p12 = torch.sum(torch.pow(mask, 2))
        sum_g12 = torch.sum(torch.pow(pred, 2))
        sum_p1g1 = torch.sum(mask * pred)
        B, C, H, W = mask.shape
        return - 2 * sum_p1g1 / (sum_p12 + sum_g12 + 1e-7)

class Unet_Weight_BCE(torch.nn.Module):
    def __init__(self):
        super(Unet_Weight_BCE, self).__init__()

    def forward(self, mask, pred, w_map, class_weight, epoch=None):
        return torch.mean(- (w_map + class_weight[:,1:2,:]) * mask * torch.log(pred + 1e-7) - class_weight[:,0:1,:] * (1 - mask) * torch.log(1 - pred + 1e-7))
    
class Weight_Map_BCE(torch.nn.Module):
    def __init__(self):
        super(Weight_Map_BCE, self).__init__()

    def forward(self, mask, pred, w_map, class_weight, epoch=None):
        if w_map != None:
            return torch.mean(- (w_map + class_weight[:,1:2,:]) * mask * torch.log(pred + 1e-7) - (w_map + class_weight[:,0:1,:]) * (1 - mask) * torch.log(1 - pred + 1e-7))
        else:
            return torch.mean(- class_weight[:,1:2,:] * mask * torch.log(pred + 1e-7) - class_weight[:,0:1,:] * (1 - mask) * torch.log(1 - pred + 1e-7))




