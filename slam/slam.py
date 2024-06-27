#!/usr/bin/env python3
import cv2
from display import Display
from extractor import Extractor
H = 1080 // 2
W = 1920 // 2


display = Display(W, H)


ext = Extractor()


def process_frame(img):
    img = cv2.resize(img, (W, H))
    matches = ext.extract(img)
    print(len(matches))
    for pt1, pt2 in matches:
        u1, v1 = map(lambda x: int(round(x)), pt1)
        u2, v2 = map(lambda x: int(round(x)), pt2)
        cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

    display.draw(img)


def process_video(filepath):
    cap = cv2.VideoCapture(filepath)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process_frame(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    FILEPATH = "../data/3.mp4"
    process_video(FILEPATH)
