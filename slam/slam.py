#!/usr/bin/env python3
import cv2

from display import Display

H = 1080 // 2
W = 1920 // 2


display = Display(W, H)


def process_frame(img):
    img = cv2.resize(img, (W, H))
    display.draw(img)


def process_video(filepath):
    cap = cv2.VideoCapture(filepath)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process_frame(frame)
        if cv2.waitKey(1) and 0xFF == ord("q"):
            break


if __name__ == "__main__":
    FILEPATH = "./data/854669-hd_1920_1080_30fps.mp4"
    process_video(FILEPATH)
