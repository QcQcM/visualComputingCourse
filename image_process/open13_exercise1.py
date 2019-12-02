import cv2
import numpy as np


img = cv2.imread('./img/handwriting1.jpg', 0)
res, thre = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
img, contours, hierarchy = cv2.findContours(thre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

cnt = contours[2]
cv2.drawContours(color_img, [cnt], -1, (0, 0, 255), 2)

x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

rect = cv2.minAreaRect(cnt)
box = np.int0(cv2.boxPoints(rect))
cv2.drawContours(color_img, [box], 0, (255, 0, 0), 2)

(x, y), radius = cv2.minEnclosingCircle(cnt)
(x, y, radius) = np.int0((x, y, radius))
cv2.circle(color_img, (x, y), radius, (0, 128, 255), 2)

cv2.imshow('exercise1', color_img)
cv2.waitKey(0)

print(cv2.contourArea(cnt))
print(cv2.arcLength(cnt, True))
M = cv2.moments(cnt)
cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
print(cx, cy)


