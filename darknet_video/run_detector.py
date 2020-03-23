import time
from threading import Thread

import cv2

from darknet_video.utils import gui_threading
from multiprocessing import Manager
from darknet_video.detector import YOLO
from darknet_video.video import CvVideo


class ThreadingDetector:
    def __init__(self, url, **kwargs):
        self.kwargs = kwargs
        self.url = url

        self.run()

    def run(self):
        cap_th = self._captrue_stream()
        detect_th = self._detect_frame()
        detect_th.start()
        cap_th.start()
        # Thread(target=gui_threading, args=(self.stream,)).start()
        cap_th.join()
        detect_th.join()

    def _detect_frame(self):
        def thread():
            yolo = YOLO(self.stream, **self.kwargs)
            yolo.detect_stream()
        return Thread(target=thread)

    def _captrue_stream(self):
        self.stream = CvVideo()

        def thread():
            self.stream.capture_stream(self.url, video_size=(1920, 1080))
        return Thread(target=thread)

