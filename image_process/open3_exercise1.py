#   实现一个可以拖动滑块播放视频的功能
import cv2


def nothing(x):
    pass


capture = cv2.VideoCapture('demo_video.mp4')
while capture.isOpened():
    cv2.namedWindow('frame')
    cv2.createTrackbar('position', 'frame', 1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)-1), nothing)
    while True:
        ret, frame = capture.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        pos = cv2.getTrackbarPos('position', 'frame')
        capture.set(cv2.CAP_PROP_POS_FRAMES, pos)
    break

