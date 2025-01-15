import torch
import torchvision
from model.MI.steerable import SteerablePyramid
from model.MI.SPMap import SPMap
from PIL import Image
import matplotlib.pyplot as plt


path = "C:/Users/Renhao Lu/Desktop/Skea_topo-main/Para/data/masks/000.png"

imgg = torchvision.transforms.Grayscale(num_output_channels=1)(torchvision.transforms.ToTensor()(Image.open(path)))

imgg = imgg.reshape((1,1,1024,1024)).cuda()

# imgg = torch.cat((imgg, imgg), dim=0)

print("imgg shape: ", imgg.shape)

SP = SteerablePyramid(imgSize=1024, K=4, N=4)

sp_output = SP(imgg)

print("layer 0 shape", sp_output[0].shape)
print("layer 1 shape", sp_output[1].shape)
print("layer 1 device", sp_output[1].device)

sp_map = SPMap()

a = sp_map(sp_output)

plt.imshow(a.cpu().squeeze().numpy())
plt.show()

