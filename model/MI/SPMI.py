import torch
import torchvision
# from model.MI.steerable import SteerablePyramid
#from steerable import SteerablePyramid

from ComplexSteerablePyramid import ComplexSteerablePyramid
import model.loss as loss

from PIL import Image
import matplotlib.pyplot as plt
import math

from model.MI.SPMap import SPMap

_POS_ALPHA = 5e-4

# 1-12-25 tested: 0.0001 SPMI + 0.01 BCE

class SPMILoss(torch.nn.Module):
    def __init__(self, imageSize, spN = 4, spK=4, beta=0.25, lamb=0.5, mag=1, map_method=2, map_weight=1, ffl: bool=False):
        super(SPMILoss, self).__init__()
        self.sp = ComplexSteerablePyramid(Complex=imgSize=imageSize, N=spN, K=spK)
        self.beta = beta
        self.ffl = ffl
        self.lamb = lamb
        self.mag = mag
        self.map_method = map_method
        self.sp_map = SPMap()
        self.map_weight = map_weight
        self.BCEW = loss.Weight_Map_BCE()

    def forward(self, mask, pred, w_map, class_weight, epoch=None):
        if epoch == 0:
            return self.BCEW(mask, pred, None, class_weight)
        sp_mask = self.sp(mask)
        sp_pred = self.sp(pred)
        mi_output = []
        for i in range(self.sp.N):
            if not self.ffl:
                mi_output.append(torch.mean(self.mi(sp_mask[i + 1], sp_pred[i + 1])))
            else:
                mi_output.append(torch.mean(torch.log(torch.norm(sp_mask[i + 1] - sp_pred[i + 1], dim=1))))
        with torch.no_grad():
            if self.map_method == 2:
                w_map = self.sp_map(sp_mask) + self.sp_map(sp_pred)
                w_map = w_map * self.map_weight
            elif self.map_method == 1:
                w_map = self.sp_map(sp_mask) * self.map_weight
            else:
                w_map = None
        loss = self.BCEW(mask, pred, None, class_weight) * self.lamb
        #for i in range(self.sp.N):
        #    loss += math.pow(self.beta, self.sp.N - i - 1) * mi_output[i] * self.mag
        return loss

    def mi(self, mask, pred):
        # print(mask.shape)
        B, C, H, W = mask.shape
        mask_flat = mask.view(B, C, H * W).type(torch.cuda.DoubleTensor)
        mask_mean = torch.mean(mask_flat, dim=2, keepdim=True)
        mask_centered = mask_flat - mask_mean

        pred_flat = pred.view(B, C, H * W).type(torch.cuda.DoubleTensor)
        pred_mean = torch.mean(pred_flat, dim=2, keepdim=True)
        pred_centered = pred_flat - pred_mean

        var_mask = torch.matmul(mask_centered, torch.permute(mask_centered, (0, 2, 1)))
        var_pred = torch.matmul(pred_centered, torch.permute(pred_centered, (0, 2, 1)))
        cov_mask_pred = torch.matmul(mask_centered, torch.permute(pred_centered, (0, 2, 1)))

        # print(var_mask, var_pred, cov_mask_pred)

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
