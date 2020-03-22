from threading import Thread

import cv2


class CvVideo:
    def __init__(self, cv_window_size=(1280, 720)):
        self.got_frame = None

        # cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Demo', *cv_window_size)

    def capture_stream(self, url=None, save_video=False, video_size=(1280, 720)):

        # cv2.resizeWindow('Demo', *video_size)
        if url is None:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(url)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_size[1])

        if save_video:
            out = cv2.VideoWriter(
                "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0, self.input_size)
        print("Starting the YOLO loop...")

        while cap.isOpened():
            ret, self.got_frame = cap.read()
            # cv2.imshow('Demo', self.got_frame)
            # cv2.waitKey(1)

        cap.release()
        if save_video:
            out.release()


