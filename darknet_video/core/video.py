import glob
import os
import random
import time
from queue import Queue

import cv2


class CvVideo:
    def __init__(self, url="0", video_size=(1920, 1080), is_rotate=False, labeling_fps=30, is_lock=False,
                 val_split=0.2, limit_frames=None, start_frame=0, is_labeling=False, is_realtime=True, **kwargs):
        self.is_realtime = is_realtime
        self.val_split = val_split
        self.is_lock = is_lock
        self.is_rotate = is_rotate
        # self.lock = threading.Lock()
        self.raw = None
        self.yolo_raw = None
        self.labeling_fps = labeling_fps
        self.frame_i = 0
        self.queue = Queue(maxsize=1)
        self.is_stop = False
        self.limit_frames = limit_frames
        self.start_frame = start_frame

        # TODO(190716): Move below props to suitable detector class.
        self.detections = []
        self.track_box = None
        self.is_previous = False
        self.manual_roi = None
        self.is_labeling = is_labeling

        if isinstance(url, int):
            url = str(url)
        self.url = url
        # Use filename to assure training and validation set.
        if is_labeling:
            random.seed(os.path.basename(url))
        self._init_stream(video_size)

    def _init_stream(self, video_size=(1920, 1080)):
        if self.url.endswith("*.jpg") or self.url.endswith("*.png"):
            if not os.path.exists(os.path.dirname(self.url)):
                self.url = os.path.join("../../darknet/data", self.url)
            img_files = sorted(glob.glob(self.url))
            print(img_files)
            self._read_files(img_files)
        else:
            if self.url.isnumeric():
                self.cap = cv2.VideoCapture(int(self.url))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_size[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_size[1])
            else:
                self.cap = cv2.VideoCapture(self.url)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            print("VIDEO FPS: %d" % self.fps)

            frame_count = round(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("Total frames: %d" % frame_count)
            if self.limit_frames:
                # if frame_count < self.limit_frames:
                #     raise AssertionError("frame_count < limit_frames")
                frame_count = self.limit_frames
                print("Limited frames: %d" % frame_count)
            if self.is_labeling:
                self.val_set = random.choices(range(frame_count - 1), k=round(frame_count * self.val_split))
            self.frame_count = frame_count

            self.jump_frames = round(self.fps / self.labeling_fps)

    def capture_stream(self):

        print("Start capture.")
        try:
            while (self.cap.isOpened() and not self.is_stop) and (
                    not self.limit_frames or self.frame_i < self.limit_frames - 1):
                self._read_frame()
                if self.is_lock:
                    self.queue.put("check1")
                    self.queue.put("check2")
                    self.queue.put(self.raw)
        finally:
            self.is_stop = True
            self.cap.release()

    def _read_frame(self):
        if self.is_previous:
            self._back_frame()
        ret, raw = self.cap.read()
        if raw is None:
            self.is_stop = True
            return
        if self.is_rotate:
            raw = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)
        self.raw = raw
        self.frame_i = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
        self.frame_window = [self.frame_i, raw]
        if self.is_labeling:
            self._next_frame()
        elif not self.is_realtime:
            time.sleep(1 / self.fps)

    def _next_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_i + self.jump_frames)

    def _back_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_i - self.jump_frames)
        self.is_previous = False

    def _read_files(self, img_files):
        for i, im in enumerate(img_files):
            with self.lock:
                print("%s: %s" % (i, im))
                im = cv2.imread(im)
                self._process(im)
            time.sleep(0.001)
        self.raw = None
