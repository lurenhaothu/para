import torch
from sklearn.model_selection import KFold
from model.dataset import SNEMI3DDataset
from torch.utils.data import DataLoader
from model.unet import UNet
from PIL import Image
import numpy as np
import model.metrics as metrics
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
import model.loss as loss
import time
from concurrent.futures import ThreadPoolExecutor

expriment_name = "Weight_BCE_b_10_1-6-2025"
lossFunc = loss.HomeMadeBCE_withClassBalance()

# 100 0.5053152359607173 0.19049978520827424 20570595.0 84287005.0

cwd = os.getcwd()
curResultDir = cwd + "/results/" + expriment_name + "/"
os.makedirs(curResultDir, exist_ok=True)

numFile = 100
fileList = [i for i in range(numFile)]

fold_num = 3
kf = KFold(n_splits=fold_num, shuffle=True)

batch_size = 10

# TODO: Normalize whole dataset by its mean and std

# What to do each eoch
# 1. train, log training loss
# 2. validate, log metrics, save model in a early stopping manner, log all metrics
# What to do each experiment:
# Set experiment name
# save all in the parameter name
# Generate illustration if needed

for fold, (train_and_val_list, test_list) in enumerate(kf.split(fileList)):

    print("fold: " + str(fold))

    train_size = int(0.8 * len(train_and_val_list))
    val_size = len(train_and_val_list) - train_size

    print(type(train_and_val_list))

    train_list = random.sample(train_and_val_list.tolist(), train_size)
    val_size = [i for i in train_and_val_list if i not in train_list]

    train_dataset = SNEMI3DDataset(train_list, augmentation=True)
    val_dataset = SNEMI3DDataset(train_list, augmentation=False)
    test_dataset = SNEMI3DDataset(test_list, augmentation=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    epoch_num = 50

    unet = UNet()

    unet.cuda()

    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    vi_record = []
    curmin_vi = None

    for epoch in range(epoch_num):
        print("epoch: ", epoch)

        t1 = time.time()

        unet.train()
        for image, mask in train_dataloader:
            image = image.cuda()
            mask = mask.cuda()
            pred = torch.softmax(unet(image), 1)[:, 1:2, :, :]
            # print(pred.shape, mask.shape)
            loss = lossFunc(mask, pred)
            print(loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 5)
            optimizer.step()
        scheduler.step()

        t2 = time.time()

        print("train time: ", t2 - t1)

        # torch.save(unet.state_dict(), "epoch_" + str(i) + ".pth")

        # validation
        vi = 0.
        images = []
        masks = []
        preds = []
        preds_bin = []
        vis = []
        for index, (image, mask) in enumerate(val_dataloader):
            unet.eval()
            with torch.no_grad():
                pred = torch.softmax(unet(image.cuda()), 1)[:, 1:2, :, :]
                images.append(image.squeeze().numpy())
                masks.append(mask.squeeze().numpy())
                preds.append(pred.cpu().squeeze().numpy())
                preds_bin.append((preds[-1] > 0.5).astype(int))

            # TODO: calculate loss grad on pixels
            
            if epoch % 2 == 0 and index == 0:
                fig, axes = plt.subplots(2,3)
                axes[0][0].imshow(image.squeeze().numpy())
                axes[0][1].imshow(mask.squeeze().numpy())
                axes[0][2].imshow(pred.cpu().squeeze().numpy())
                axes[1][0].imshow((pred.cpu().squeeze().numpy() > 0.5).astype(int))
                axes[1][1].imshow(mask.squeeze().numpy() - (pred.cpu().squeeze().numpy() > 0.5).astype(int))

                pred.requires_grad = True
                l = lossFunc(mask.cuda(), pred)
                grad = torch.autograd.grad(l, pred)
                print(grad[0].shape)
                axes[1][2].imshow(grad[0].cpu().squeeze().numpy())

                plt.show()

        with ThreadPoolExecutor(max_workers=10) as executor:
            vis = list(executor.map(metrics.vi, preds_bin, masks))
        vi = np.mean(vis)

        t3 = time.time()
        print("val time: ", t3 - t2)

        vi_record.append(vi)
        result = pd.DataFrame({
            "Epoch": [epoch],
            "VI": [vi],
        })
        if not os.path.exists(curResultDir + "fold_" + str(fold) + "_val_result.csv"):
            result.to_csv(curResultDir + "fold_" + str(fold) + "_val_result.csv", index=False)
        else:
            result.to_csv(curResultDir + "fold_" + str(fold) + "_val_result.csv", mode='a', header=False, index=False)

        if curmin_vi == None or vi < curmin_vi:
            curmin_vi = vi
            torch.save(unet.state_dict(), curResultDir + "fold_" + str(fold) + "_best_model_state.pth")

        print("epoch: " + str(epoch) + " VI: " + str(vi))

    #test
    best_state_dict = torch.load(curResultDir + "fold_" + str(fold) + "_best_model_state.pth")
    unet.load_state_dict(best_state_dict)
    unet.eval()
    
    images_test = []
    masks_test = []
    preds_test = []
    preds_bin_test = []
    vis_test = []
    for index, (image, mask) in enumerate(test_dataloader):
        unet.eval()
        with torch.no_grad():
            pred = torch.softmax(unet(image.cuda()), 1)[:, 1:2, :, :]
            images_test.append(image.squeeze().numpy())
            masks_test.append(mask.squeeze().numpy())
            preds_test.append(pred.cpu().squeeze().numpy())
            preds_bin_test.append((preds[-1] > 0.5).astype(int))

    with ThreadPoolExecutor(max_workers=10) as executor:
        vis_test = list(executor.map(metrics.vi, preds_bin_test, masks_test))
    vi_test = np.mean(vis_test)

    test_result = pd.DataFrame({
        "Fold": [fold],
        "VI": [vi_test],
    })
    if not os.path.exists(curResultDir + "test_result.csv"):
        test_result.to_csv(curResultDir + "test_result.csv", index=False)
    else:
        test_result.to_csv(curResultDir + "test_result.csv", mode='a', header=False, index=False)

    print("---------------------------------------------------------")
    print("-----------------------fold finished---------------------")
    print("fold: " + str(fold) + " VI: " + str(vi_test))
    print("---------------------------------------------------------")  