import cv2
import numpy as np

def glcm(image_array):
 
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array

    resized = cv2.resize(gray, (64, 64))
    mean = np.mean(resized)
    std = np.std(resized)
    min_val = np.min(resized)
    max_val = np.max(resized)

    return [mean, std, min_val, max_val]


def haralick_feat(image_array):

    return np.mean(image_array, axis=(0, 1))


def bitdesc_feat(image_array):
  
    return np.var(image_array, axis=(0, 1))


def concat(image_array):
    return glcm(image_array) + haralick_feat(image_array).tolist() + bitdesc_feat(image_array).tolist()


def glcm_rgb(path_img):
    img = cv2.imread(path_img)
    img_resized = cv2.resize(img, (64, 64))
    features = []

    for i in range(3):
        channel = img_resized[:, :, i]
        mean = np.mean(channel)
        std = np.std(channel)
        min_val = np.min(channel)
        max_val = np.max(channel)

        features.extend([mean, std, min_val, max_val])

    return features


__all__ = ["glcm", "haralick_feat", "bitdesc_feat", "concat", "glcm_rgb"]
