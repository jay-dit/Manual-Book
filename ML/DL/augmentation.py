import cv2
import numpy as np
def upsampling(path, width = 256):
    img = cv2.imread(path)
    delta = width - img.shape[0]
    zeros_up = np.zeros((delta//2, 128, 3), dtype = 'uint8')
    zeros_down = np.zeros((delta//2 + delta%2, 128, 3),
                          dtype = 'uint8')
    return np.concatenate((zeros_up, img, zeros_down), axis = 0)

def flip_blur(img):
    return cv2.blur(img[:,::-1,:], (3, 3))

def noise(img):
    for i in range(6000):
        x = int(np.random.uniform(0, 128, 1))
        y = int(np.random.uniform(0, 256, 1))
        img[y, x,:] = np.array([0,0,0])
    return img


def flip_sdvig(img):
    x = int(np.random.uniform(-50, 50, 1))
    y = int(np.random.uniform(-50, 50, 1))
    h, w = img.shape[:2]
    translation_matrix = np.float32([[1, 0, y],[0, 1, x]])
    return cv2.warpAffine(img[:,::-1,:], translation_matrix, (w, h))

def flip_blur_sdvig(img):
    x = int(np.random.uniform(-50, 50, 1))
    y = int(np.random.uniform(-50, 50, 1))
    h, w = img.shape[:2]
    translation_matrix = np.float32([[1, 0, y],[0, 1, x]])
    return cv2.blur(cv2.warpAffine(img[:,::-1,:], 
                    translation_matrix, 
                    (w, h)), 
                    (3,3))
