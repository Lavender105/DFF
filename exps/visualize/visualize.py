###########################################################################
# Created by: Yuan Hu
# Email: huyuan@radi.ac.cn
# Copyright (c) 2019
###########################################################################

import numpy as np
from skimage import io


def apply_mask(image, mask, color):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] + color[c],
                                  image[:, :, c])
    return image

def visualize_prediction(dataset, path, pred):
    n, h, w = pred.shape
    image = np.zeros((h, w, 3))
    # image = image.astype(np.uint32)

    if dataset == 'cityscapes':
      colors = [[128, 64, 128],
               [244, 35, 232],
               [70, 70, 70],
               [102, 102, 156],
               [190, 153, 153],
               [153, 153, 153],
               [250, 170, 30],
               [220, 220, 0],
               [107, 142, 35],
               [152, 251, 152],
               [70, 130, 180],
               [220, 20, 60],
               [255, 0, 0],
               [0, 0, 142],
               [0, 0, 70],
               [0, 60, 100],
               [0, 80, 100],
               [0, 0, 230],
               [119, 11, 32]]
    else:
      assert dataset == 'sbd'
      colors = [[128, 0, 0],
               [0, 128, 0],
               [128, 128, 0],
               [0, 0, 128],
               [128, 0, 128],
               [0, 128, 128],
               [128, 128, 128],
               [64, 0, 0],
               [192, 0, 0],
               [64, 128, 0],
               [192, 128, 0],
               [64, 0, 128],
               [192, 0, 128],
               [64, 128, 128],
               [192, 128, 128],
               [0, 64, 0],
               [128, 64, 0],
               [0, 192, 0],
               [128, 192, 0],
               [0, 64, 128]]

    pred = np.where(pred >= 0.5, 1, 0).astype(np.bool)
    edge_sum = np.zeros((h, w))

    for i in range(n):
      color = colors[i]
      edge = pred[i,:,:]
      edge_sum = edge_sum + edge
      masked_image = apply_mask(image, edge, color)

    edge_sum = np.array([edge_sum, edge_sum, edge_sum])
    edge_sum = np.transpose(edge_sum, (1, 2, 0))
    idx = edge_sum > 0
    masked_image[idx] = masked_image[idx]/edge_sum[idx]
    masked_image[~idx] = 255
    
    io.imsave(path, masked_image/255)