from threading import Thread
from darknet_video.detector import YOLODetector
from darknet_video.mjpeg_server import ThreadingHTTPServer
from darknet_video.video import CvVideo


class ThreadingDetector:
    def __init__(self, url, is_forever=False, **kwargs):
        self.kwargs = kwargs
        self.url = url
        self.stream = CvVideo()
        self.yolo = YOLODetector(self.stream, **self.kwargs)
        self.server = ThreadingHTTPServer()
        self.run(is_forever)

    def run(self, forever):
        cap_th = self._captrue_stream()
        detect_th = self._detect_frame()
        server_th = self._stream_yolo()
        detect_th.start()
        cap_th.start()
        server_th.start()
        # Thread(target=gui_threading, args=(self.stream,)).start()
        if forever:
            cap_th.join()
            detect_th.join()
            server_th.join()

    def _detect_frame(self):
        def thread():
            self.yolo.detect_stream(**self.kwargs)

        return Thread(target=thread)

    def _captrue_stream(self):
        def thread():
            self.stream.capture_stream(self.url)

        return Thread(target=thread)

    def _stream_yolo(self):
        def thread():
            self.server.stream = self.stream
            self.server.serve_forever()

        return Thread(target=thread)
