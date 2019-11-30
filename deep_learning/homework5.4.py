import numpy as np
import cv2


def gasuss_noise(image, mean=0, var=9):
    image = np.array(image/255, dtype=float)
    nosie = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + nosie
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out*255)
    return out


img = cv2.imread('blur1.jpg', 2)
gasuss_img = gasuss_noise(img)
cv2.imwrite('blur1_v1.jpg', gasuss_img)
cv2.imwrite('blur1.jpg', img)
cv2.imshow('G_noise', gasuss_img)
cv2.waitKey(0)
