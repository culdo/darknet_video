import glob
import threading
import time

import cv2


class CvVideo:
    def __init__(self):
        self.lock = threading.Lock()
        self.raw = None
        self.yolo_raw = None
        self.detections = []

    def capture_stream(self, url="0", video_size=(1920, 1080)):
        self.url = url

        if url.endswith("*.jpg") or url.endswith("*.png"):
            img_files = sorted(glob.glob(url))
        else:
            if url.isnumeric():
                cap = cv2.VideoCapture(int(url))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_size[1])
            else:
                cap = cv2.VideoCapture(url)

        print("Start capture.")
        if url.endswith("*.jpg") or url.endswith("*.png"):
            for i, im in enumerate(img_files):
                with self.lock:
                    print("%s: %s" % (i, im))
                    im = cv2.imread(im)
                    self._process(im)
                time.sleep(0.001)
            self.raw = None
        else:
            try:
                while cap.isOpened():
                    ret, raw = cap.read()
                    self._process(raw)
            finally:
                cap.release()

    def _process(self, raw):
        self.raw = raw
        if self.url.startswith("/") or self.url.startswith("."):
            time.sleep(1 / 30)
