#   尝试在视频中同时提取红色、蓝色、绿色的物体
import cv2
import numpy as np
#   颜色划定可能有问题，不能很好地显示出来


capture = cv2.VideoCapture(0)
while True:
    #   从视频中读取一帧出来
    ret, frame = capture.read()
    #   把这一帧图像从BGR转换成HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #   设置红色、蓝色、绿色上下限范围
    upper_blue = np.array([130, 255, 255])
    lower_blue = np.array([100, 110, 110])

    upper_red = np.array([179, 255, 255])
    lower_red = np.array([160, 120, 120])

    upper_green = np.array([70, 255, 255])
    lower_green = np.array([40, 90, 90])

    mask = cv2.inRange(hsv, lower_blue, upper_blue) + cv2.inRange(hsv, lower_red, upper_red) \
        + cv2.inRange(hsv, lower_green, upper_green)

    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    if cv2.waitKey(1) == ord('q'):
        break
