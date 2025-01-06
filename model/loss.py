import torch

# def BCELoss():
#     return torch.nn.BCELoss()

def HomeMadeBCE_withClassBalance(): #20570595.0 84287005.0
    def loss(mask, pred):
        return torch.mean(- 2.5 * mask * torch.log(pred) - 0.6 * (1 - mask) * torch.log(1 - pred))
    return loss

def HomeMadeBCE():
    def loss(mask, pred):
        with torch.no_grad():
            error = (mask != (pred > 0.5).to(torch.int8)).to(torch.int8)
        return torch.mean((0.5 + 0.5 * error) * (- 2.5 * mask * torch.log(pred) - 0.6 * (1 - mask) * torch.log(1 - pred)))
    return loss

def DiceLoss():
    def loss(mask, pred):
        sum_p12 = torch.sum(torch.pow(mask, 2))
        sum_g12 = torch.sum(torch.pow(pred, 2))
        sum_p1g1 = torch.sum(mask * pred)
        B, C, H, W = mask.shape
        return - 2 * sum_p1g1 / (sum_p12 + sum_g12 + 1e-7)
    return loss




