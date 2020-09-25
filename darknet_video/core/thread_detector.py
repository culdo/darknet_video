import time
from threading import Thread

from darknet_video.core.detector import YOLODetector
from darknet_video.core.video import CvVideo
from darknet_video.utils.mjpeg_server import ThreadingHTTPServer


class ThreadingDetector:
    def __init__(self, url, is_stream_result=False, blocking=True, **kwargs):
        self.blocking = blocking
        self.is_stream_result = is_stream_result
        self.stream = CvVideo(url, **kwargs)
        self.yolo = YOLODetector(self.stream, is_stream_result=is_stream_result, **kwargs)
        self.server = None
        self._run()

    def _run(self):
        cap_th = self._captrue_stream()
        cap_th.start()
        detect_th = self._detect_frame()
        detect_th.start()
        if self.is_stream_result:
            server_th = self._stream_result()
            server_th.start()
        if self.blocking:
            detect_th.join()
            cap_th.join()
        self._after_clear()

    def _detect_frame(self):
        def thread():
            self.yolo.detect_stream()

        return Thread(target=thread)

    def _captrue_stream(self):
        def thread():
            self.stream.capture_stream()

        return Thread(target=thread)

    def _stream_result(self):
        self.server = ThreadingHTTPServer()

        def thread():
            self.server.stream = self.stream
            self.server.serve_forever()

        return Thread(target=thread)

    def _after_clear(self):
        def thread():
            while not self.stream.is_stop:
                time.sleep(0.01)
            if self.server:
                self.server.shutdown()
        Thread(target=thread).start()
