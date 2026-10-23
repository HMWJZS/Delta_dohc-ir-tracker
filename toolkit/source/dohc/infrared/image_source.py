"""Native-coordinate BGR patches; hardware frames stay YUV until a patch is used."""
import numpy as np


def crop_bgr(image, rectangle):
    x0, y0, x1, y1 = rectangle
    h, w = image.shape[:2]
    if (any(not isinstance(v, (int, np.integer)) for v in rectangle) or
            not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h):
        raise ValueError('BGR crop must be an in-bounds native integer rectangle')
    if isinstance(image, np.ndarray):
        return image[y0:y1, x0:x1]
    return image.crop_bgr(rectangle)


def full_bgr(image):
    return crop_bgr(image, [0, 0, image.shape[1], image.shape[0]])
