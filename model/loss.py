import torch

def BCELossWithClassWeight(weight):
    def loss(mask, pred):
        return - weight[1] * mask * torch.log(pred) - weight[0] * (1 - mask) * torch.log(1 - pred)
    return loss

# def BCELoss():
#     return torch.nn.BCELoss()

def HomeMadeBCE_withClassBalance():
    def loss(mask, pred):
        return torch.mean(- 2.5 * mask * torch.log(pred) - 0.6 * (1 - mask) * torch.log(1 - pred))
    return loss

def HomeMadeBCE():
    def loss(mask, pred):
        with torch.no_grad():
          error = (mask != (pred > 0.5).to(torch.int8)).to(torch.int8)
        return torch.mean((0.5 + 0.5 * error) * (- 2.5 * mask * torch.log(pred) - 0.6 * (1 - mask) * torch.log(1 - pred)))
    return loss

def VILoss():
    def loss(mask, pred):
        with torch.no_grad():
            pred_fix = (pred > 0.5).type(torch.int8)
        sum_p1 = torch.sum(mask)
        sum_g1 = torch.sum(pred)
        sum_g1_fix = torch.sum(pred_fix)
        sum_p1g1 = torch.sum(mask * pred)
        sum_p0 = torch.sum(1 - mask)
        sum_g0 = torch.sum(1 - pred)
        sum_g0_fix = torch.sum(1 - pred_fix)
        sum_p0g0 = torch.sum((1 - mask) * (1 - pred))
        sum_p0g1 = torch.sum((1 - mask) * pred)
        sum_p1g0 = torch.sum(mask * (1 - pred))
        B, C, H, W = mask.shape
        N = H * W * B
        l11 = sum_p1 * sum_g1_fix / N / N * (torch.log(sum_p1 * sum_g1 / sum_p1g1 + 1e-7))
        l00 = sum_p0 * sum_g0_fix / N / N * (torch.log(sum_p0 * sum_g0 / sum_p0g0 + 1e-7))
        l01 = sum_p0 * sum_g1_fix / N / N * (torch.log(sum_p0 * sum_g1 / sum_p0g1 + 1e-7))
        l10 = sum_p1 * sum_g0_fix / N / N * (torch.log(sum_p1 * sum_g0 / sum_p1g0 + 1e-7))
        return torch.log(sum_p1 * sum_g1 / sum_p1g1 + 1e-7)
    return loss

def DiceLoss():
    def loss(mask, pred):
        sum_p12 = torch.sum(torch.pow(mask, 2))
        sum_g12 = torch.sum(torch.pow(pred, 2))
        sum_p1g1 = torch.sum(mask * pred)
        B, C, H, W = mask.shape
        return - 2 * sum_p1g1 / (sum_p12 + sum_g12 + 1e-7)
    return loss




