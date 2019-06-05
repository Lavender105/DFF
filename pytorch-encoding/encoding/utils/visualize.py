import os
import sys
import random
import itertools
import colorsys

import numpy as np
from skimage.measure import find_contours
from skimage import io #hy added


def class_specific_color(class_id, bright=True):
    """
    Generate class specific color.
    """
    brightness = 1.0 if bright else 0.7
    hsv = (class_id / 20, 1, brightness)
    color = colorsys.hsv_to_rgb(*hsv)
    return color


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def visualize_prediction(path, pred):
	n, h, w = pred.shape
	image = np.ones((h, w, 3)) * 255
	image = image.astype(np.uint32)

	pred = np.where(pred >= 0.5, 1, 0).astype(np.bool)
	for i in range(n):
		color = class_specific_color(i+1)
		edge = pred[i,:,:]
		masked_image = apply_mask(image, edge, color, alpha=0.8)

	io.imsave(path, masked_image)
