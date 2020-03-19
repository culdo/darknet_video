from threading import Thread

from darknet_python.detector import YOLO
from darknet_python.video import CvVideo


class ThreadingDetector:
    def __init__(self, url, weights="../../darknet/bin/csresnext50-panet-spp-original-optimal.weights"):
        self.weights = weights
        self.url = url

        cap_th = self._captrue_stream()
        detect_th = self._detect_frame()
        detect_th.start()
        cap_th.start()
        cap_th.join()
        detect_th.join()

    def _detect_frame(self):
        def thread():
            yolo = YOLO(self.weights, self.stream)
            yolo.detect_stream()
        return Thread(target=thread)

    def _captrue_stream(self):
        self.stream = CvVideo()

        def thread():
            self.stream.capture_stream(self.url, video_size=(1920, 1080))
        return Thread(target=thread)
