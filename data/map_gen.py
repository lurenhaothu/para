from PIL import Image
import os
import numpy as np
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
from matplotlib import pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor

cwd = os.getcwd()
mask_dir = cwd + "/data/masks/"
map_dir = cwd + "/data/map/"
os.makedirs(map_dir, exist_ok=True)

w0 = 10
sigma = 5
single = True

def get_dis_map(index, mask_label):
    item_map = (mask_label != index).astype(int)
    return index, distance_transform_edt(item_map)

if single:
    single_arr = np.zeros((100, 1024, 1024))

for i in range(100):
    mask = np.array(Image.open(mask_dir + str(i).zfill(3) + ".png"))
    mask = (mask == 255).astype(int)
    mask_label, num = label(mask, background=1, connectivity=1, return_num=True)

    dis_map = np.zeros((num - 1, 1024, 1024))

    t = time.time()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(get_dis_map, j, mask_label) for j in range(1, num)]

        for future in futures:
            j, dis_map_j = future.result()
            dis_map[j - 1,:,:] = dis_map_j

    print(str(j) + ' time: ' + str(time.time() - t))
    
    t = time.time()

    sorted_dis_map = np.sum(np.partition(dis_map * mask, 2, axis=0)[0:2, :, :], axis=0)

    print('sort time: ' + str(time.time() - t))

    weight_map = w0 * np.exp( - np.power(sorted_dis_map , 2) / 2 / sigma / sigma) * mask

    np.save(map_dir + str(i).zfill(3) + ".npy", weight_map)

    if single:
        single_arr[i,:,:] = weight_map

if single:
    np.save(map_dir + "maps.npy", single_arr)

