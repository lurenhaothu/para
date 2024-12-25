import skimage.metrics as metrics
from skimage.measure import label
import numpy as np

def vi(mask: np.array, pred: np.array):
    mask_label = label(mask, background=0, connectivity=1)
    pred_label = label((pred > 0.5).astype(int), background=0, connectivity=1)
    merger_error, split_error = metrics.variation_of_information(pred_label, mask_label)

    return merger_error + split_error

if __name__ == "__main__":
    # test vi
    mask = np.random.rand(512, 512) > 0.5
    pred = np.random.rand(512, 512)
    print(vi(mask, pred))