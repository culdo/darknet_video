from threading import Thread

from darknet_video.detector import YOLODetector
from darknet_video.mjpeg_server import ThreadingHTTPServer
from darknet_video.video import CvVideo


class ThreadingDetector:
    def __init__(self, url, is_stream_result=False, **kwargs):
        self.kwargs = kwargs
        self.is_stream_result = is_stream_result
        self.url = url
        self.stream = CvVideo()
        self.yolo = YOLODetector(self.stream, **self.kwargs)
        self._run()

    def _run(self):
        cap_th = self._captrue_stream()
        cap_th.start()
        detect_th = self._detect_frame()
        detect_th.start()
        if self.is_stream_result:
            self.server = ThreadingHTTPServer()
            server_th = self._result_stream()
            server_th.start()
            server_th.join()
        detect_th.join()
        cap_th.join()

    def _detect_frame(self):
        def thread():
            self.yolo.detect_stream(**self.kwargs)

        return Thread(target=thread)

    def _captrue_stream(self):
        def thread():
            self.stream.capture_stream(self.url)

        return Thread(target=thread)

    def _result_stream(self):
        def thread():
            self.server.stream = self.stream
            self.server.serve_forever()

        return Thread(target=thread)
