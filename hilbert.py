import torch
import torchvision
from model.MI.steerable import SteerablePyramid
#from steerable import SteerablePyramid
import model.loss as loss

from PIL import Image
import matplotlib.pyplot as plt
import math

from model.MI.SPMap import SPMap

import numpy as np
'''
path = "C:/Users/Renhao Lu/Desktop/dwt/test.jpg"

imgg = torchvision.transforms.Grayscale(num_output_channels=1)(torchvision.transforms.ToTensor()(Image.open(path)))

C, H, W = imgg.shape

sp = SteerablePyramid(imgSize=400, K=4, N=4)

print(sp.hl0.shape)

fig, axe = plt.subplots(2,2)
axe[0][0].imshow(sp.hl0[0,0,0,0,...].squeeze().cpu().numpy())
axe[0][1].imshow(sp.hl0[0,0,0,1,...].squeeze().cpu().numpy())
axe[1][0].imshow(torch.sum(torch.pow(sp.hl0, 2), dim=3).squeeze().cpu().numpy())
#axe[1][1].imshow(im.angle().squeeze().cpu().numpy())
plt.show()


x = np.linspace(0, 2*np.pi, 1000)
y = np.power(x, 2)
filter1 = np.linspace(0, 1, 1000)
#filter2 = np.sqrt(1 - np.power(filter1, 2))
filter2 = 1 - filter1
#filter1 = 0.5
#filter2 = 0.5
f1 = np.fft.fft(y) * filter2
f2 = np.fft.fft(y) * filter1

fig, axe = plt.subplots(3,3)
axe[0][0].plot(x, filter1)
axe[0][1].plot(x, filter2)
axe[0][2].plot(x, np.power(filter1,2) + np.power(filter2, 2))
axe[1][0].plot(x, np.abs(f1))
axe[1][1].plot(x, np.abs(f2))
axe[1][2].plot(x, np.fft.fft(y))
axe[2][0].plot(x, np.abs(np.fft.ifft(f2)))
axe[2][1].plot(x, np.abs(np.fft.ifft(f1) + np.fft.ifft(f2)))
axe[2][2].plot(x, np.abs(np.fft.ifft(np.fft.fft(y))))
plt.show()
'''

