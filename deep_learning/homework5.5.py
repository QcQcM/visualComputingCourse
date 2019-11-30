import cv2
import numpy as np
import math
import scipy.signal as signal
from pip._vendor.distlib.compat import raw_input


#   读入待处理的三幅图像
img_synth1 = cv2.imread('synth1.jpg', 2)
img_blur = cv2.imread('blur1.jpg', 2)
img_blur_v1 = cv2.imread('blur1_v1.jpg', 2)
img_list = [img_synth1, img_blur, img_blur_v1]
#   保存计算所得的第三个矩阵，即水平垂直边缘检测后再二值化的结果
dst3_l = []

#   提取垂直边缘的水平Sobel算子
hx_kernel = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=np.float32)

threshold = int(raw_input('请输入边缘检测阈值：'))

for k in range(len(img_list)):
    #   与水平算子求卷积
    dst1 = signal.convolve2d(img_list[k], hx_kernel, mode='same')
    #   与垂直算子求卷积
    dst2 = signal.convolve2d(img_list[k], hx_kernel.T, mode='same')
    #   平方和相加
    dst3 = list(map(lambda x, y: x+y, list(map(lambda x: x ** 2, list(dst1))),
                    list(map(lambda x: x ** 2, list(dst2)))))
    #   转化为矩阵方便变形
    dst3 = np.array(dst3)
    rows, cols = dst3.shape[:2]
    #   遍历dst3，求平方并用固定阈值二值化
    for i in range(rows):
        for j in range(cols):
            dst3[i, j] = math.sqrt(dst3[i, j])
            if dst3[i, j] > threshold:
                dst3[i, j] = 255
            else:
                dst3[i, j] = 0
    dst3 = dst3.reshape(dst1.shape)
    dst3_l.append(dst3)
cv2.imshow('fixed threshold'+str(threshold), np.hstack((dst3_l[0], dst3_l[1], dst3_l[2])))
cv2.waitKey(0)