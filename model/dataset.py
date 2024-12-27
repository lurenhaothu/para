import torch
import os
from PIL import Image
import numpy as np
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt

class SNEMI3DDataset(torch.utils.data.Dataset):
    def __init__(self, indices, augmentation):
        self.indices = indices
        cwd = os.getcwd()
        self.images_dir = cwd + "/data/images/"
        self.masks_dir = cwd + "/data/masks/"

        mean = [0.5053152359607174]
        std = [0.16954360899089577]
        
        if augmentation:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True), # scale=True: 0-255 to 0-1
                # v2.Normalize(mean=mean, std=std),
                v2.RandomCrop(size=(512, 512)),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip()
            ])
        else:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomCrop(size=(512, 512)),
                # v2.Normalize(mean=mean, std=std)
            ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        image = Image.open(self.images_dir + str(self.indices[idx]).zfill(3) + '.png')
        mask = Image.open(self.masks_dir + str(self.indices[idx]).zfill(3) + '.png')

        return self.transform((image, mask))
    
# test
if __name__ == "__main__":
    dataset = SNEMI3DDataset([1])
    print(len(dataset))
    img, msk = dataset[0]
    print(img.shape)
    print(msk)
    print(torch.min(msk), torch.max(msk))
    print(torch.min(img), torch.max(img))
    
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img.squeeze())
    axes[1].imshow(msk.squeeze())
    plt.show()
    