import torch
from sklearn.model_selection import KFold
from dataset import SNEMI3DDataset
from torch.utils.data import random_split, DataLoader
from unet import UNet
from PIL import Image
import numpy as np
import metrics
import os
import pandas as pd
import matplotlib.pyplot as plt

expriment_name = "BCE_First_trial"
cwd = os.getcwd()
curResultDir = cwd + "/results/" + expriment_name + "/"
os.makedirs(curResultDir, exist_ok=True)

numFile = 100
fileList = [i for i in range(numFile)]

fold_num = 3
kf = KFold(n_splits=fold_num, shuffle=True)

batch_size = 4

# TODO: Normalize whole dataset by its mean and std

# What to do each eoch
# 1. train, log training loss
# 2. validate, log metrics, save model in a early stopping manner, log all metrics
# What to do each experiment:
# Set experiment name
# save all in the parameter name
# Generate illustration if needed

for fold, (train_list, test_list) in enumerate(kf.split(fileList)):

    train_size = int(0.8 * len(train_list))
    val_size = len(train_list) - train_size
    train_dataset, val_dataset = random_split(SNEMI3DDataset(train_list, augmentation=True), [train_size, val_size])
    test_dataset = SNEMI3DDataset(test_list, augmentation=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    epoch_num = 50

    unet = UNet()

    unet.cuda()

    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    vi_record = []
    curmin_vi = None

    for i in range(epoch_num):
        print("epoch: ", i)
        unet.train()
        for image, mask in train_dataloader:
            image = image.cuda()
            mask = mask.cuda()
            pred = torch.softmax(unet(image), 1)
            # print(pred.shape, mask.shape)
            loss = torch.nn.BCELoss()(pred[:, 0:1, :, :], mask)
            print(loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 5)
            optimizer.step()
        scheduler.step()

        torch.save(unet.state_dict(), "epoch_" + str(i) + ".pth")

        vi = 0.
        
        for index, (image, mask) in enumerate(val_dataloader):
            unet.eval()
            with torch.no_grad():
                pred = torch.softmax(unet(image.cuda()), 1)[:, 0:1, :, :]
                vi += metrics.vi(mask.squeeze().numpy(), pred.cpu().squeeze().numpy())

                if index == 0:
                    fig, axes = plt.subplots(1,3)
                    axes[0].imshow(image.squeeze().numpy())
                    axes[1].imshow(mask.squeeze().numpy())
                    axes[2].imshow(pred.squeeze().numpy())
                    axes[4].imshow((pred.squeeze().numpy() > 0).astype(int))
                    plt.show()
        vi /= len(val_dataloader)
        
        vi_record.append(vi)
        result = pd.DataFrame({
            "Epoch": [i],
            "VI": [vi],
        })
        if not os.path.exists(curResultDir + "val_result.csv"):
            result.to_csv(curResultDir + "val_result.csv", index=False)
        else:
            result.to_csv(curResultDir + "val_result.csv", mode='a', header=False, index=False)

        if curmin_vi == None or vi < curmin_vi:
            curmin_vi = vi
            torch.save(unet.state_dict(), curResultDir + "best_model_state.pth")

        print("epoch: " + str(i) + " VI: " + str(vi))

    break
