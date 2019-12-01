#   如下图，找到3个圆环的内环，然后填充成(180,215,215)这种颜色
#   实现思路是找内层轮廓，然后填充
import cv2
import numpy as np

img = cv2.imread('circle_ring.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
res, thre = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

image, contours, hierarchy = cv2.findContours(thre, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# hierarchy的形状为(1,6,4)，使用np.squeeze压缩一维数据，变成(6,4)
hierarchy = np.squeeze(hierarchy)

for i in range(len(hierarchy)):
    if hierarchy[i, 3] != -1:
        cv2.drawContours(img, contours, i, (180, 215, 215), -1)

cv2.imshow('fill', img)
cv2.waitKey(0)

