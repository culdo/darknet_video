import glob
import os
import threading
import time

import cv2


class CvVideo:
    def __init__(self):
        self.lock = threading.Lock()
        self.raw = None
        self.yolo_raw = None
        self.detections = [], []
        self.labeling_fps = None
        self.frame_i = None
        self.track_box = None
        self.is_stop = False

    def capture_stream(self, url="0", video_size=(1920, 1080)):
        self.url = url

        if url.endswith("*.jpg") or url.endswith("*.png"):
            if not os.path.exists(os.path.dirname(url)):
                url = os.path.join("../../darknet/data", url)
            img_files = sorted(glob.glob(url))
            print(img_files)
            self._read_files(img_files)
        else:
            if url.isnumeric():
                cap = cv2.VideoCapture(int(url))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_size[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_size[1])
            else:
                cap = cv2.VideoCapture(url)

            print("Start capture.")
            try:
                while cap.isOpened() and not self.is_stop:
                    ret, raw = cap.read()
                    if raw is None:
                        break
                    # raw = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)
                    self._process(raw)
                    if self.labeling_fps:
                        with self.lock:
                            self.frame_i = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
                            print("self.frame_i: %s" % self.frame_i)
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_i + (fps / self.labeling_fps))
                        time.sleep(0.001)
            finally:
                self.is_stop = True
                cap.release()

    def _read_files(self, img_files):
        for i, im in enumerate(img_files):
            with self.lock:
                print("%s: %s" % (i, im))
                im = cv2.imread(im)
                self._process(im)
            time.sleep(0.001)
        self.raw = None

    def _process(self, raw):
        self.raw = raw
        if self.labeling_fps is None and (self.url.startswith("/") or self.url.startswith(".")):
            time.sleep(1 / 30)
