import torch
import torchvision
from model.MI.steerable import SteerablePyramid
#from steerable import SteerablePyramid

from PIL import Image
import matplotlib.pyplot as plt

_POS_ALPHA = 5e-4

class SPMILoss(torch.nn.Module):
    def __init__(self, imageSize, spN = 4, spK=6):
        super(MILoss, self).__init__()
        self.sp = SteerablePyramid(imgSize=imageSize, N=spN, K=spK)

    def forward(self, mask, pred, _):
        sp_mask = self.sp(mask)
        sp_pred = self.sp(pred)
        mi_output = []
        for i in range(self.sp.N):
            mi_output.append(self.mi(sp_mask[i + 1].squeeze(1), sp_pred[i + 1].squeeze(1)))
        return torch.sum(mi_output)

    def mi(self, mask, pred):
        print(mask.shape)
        B, C, H, W = mask.shape
        mask_flat = mask.view(B, C, H * W).type(torch.cuda.DoubleTensor)
        mask_mean = torch.mean(mask_flat, dim=2)
        mask_centered = mask_flat - mask_mean.unsqueeze(-1)

        pred_flat = pred.view(B, C, H * W).type(torch.cuda.DoubleTensor)
        pred_mean = torch.mean(pred_flat, dim=2)
        pred_centered = pred_flat - pred_mean.unsqueeze(-1)

        var_mask = torch.matmul(mask_centered, torch.permute(mask_centered, (0, 2, 1)))
        var_pred = torch.matmul(pred_centered, torch.permute(pred_centered, (0, 2, 1)))
        cov_mask_pred = torch.matmul(mask_centered, torch.permute(pred_centered, (0, 2, 1)))

        print(var_mask, var_pred, cov_mask_pred)

        diag_matrix = torch.eye(C).unsqueeze(0)
        inv_cov_pred = torch.inverse(var_pred + diag_matrix.type_as(var_pred) * _POS_ALPHA)

        cond_cov_mask_pred = var_mask - torch.matmul(torch.matmul(cov_mask_pred, inv_cov_pred), torch.permute(cov_mask_pred, (0, 2, 1)))

        chol = torch.linalg.cholesky(cond_cov_mask_pred)
        return 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)
    
if __name__ == "__main__":
    mask_image_1 = torchvision.transforms.ToTensor()(Image.open('data/masks/000.png')).unsqueeze(0).cuda()
    mask_image_2 = torchvision.transforms.ToTensor()(Image.open('data/masks/099.png')).unsqueeze(0).cuda()

    loss = SPMILoss(imageSize=1024)
    print(loss(mask_image_1, mask_image_2))
