import cv2


#   打开lena.jpg，将帽子部分（高：25，宽：220）的红色通道截取出来并显示。
img = cv2.imread('lena.jpg')
hat_r = img[25:120, 50:220, 2]
cv2.imshow('hat_red', hat_r)
cv2.waitKey(0)