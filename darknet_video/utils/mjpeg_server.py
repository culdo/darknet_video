import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):

        # Get the min intra frame delay
        if self.server.maxfps != 0:
            minDelay = 1.0 / self.server.maxfps
        else:
            minDelay = 0

        # Send headers
        self.send_response(200)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=--myboundary")
        self.end_headers()

        o = self.wfile

        # Send image files in a loop
        lastFrameTime = 0
        while self.server.stream.yolo_jpg is not None and not self.server.stream.is_stop:

            contents = self.server.stream.yolo_jpg

            # Wait if required so we stay under the max FPS
            if lastFrameTime != 0:
                now = time.time()
                delay = now - lastFrameTime
                if delay < minDelay:
                    time.sleep(minDelay - delay)

            buff = "Content-Length: %s \r\n" % str(len(contents))
            # logging.debug( "Serving frame %s", imageFile )
            o.write(b"--myboundary\r\n")
            o.write(b"Content-Type: image/jpeg\r\n")
            o.write(buff.encode("utf8"))
            o.write(b"\r\n")
            o.write(contents)
            o.write(b"\r\n")

            lastFrameTime = time.time()


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    def __init__(self, port=8091):
        super().__init__(("0.0.0.0", port), RequestHandler)
        print("Listening on Port " + str(port) + "...")
        self.maxfps = 0
