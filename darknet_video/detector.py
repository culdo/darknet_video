import os
import re
import subprocess
import time

import cv2
import numpy as np

from darknet_video.utils import cv_draw_boxes
from . import darknet


class MetaMain:
    def __init__(self):
        self.classes = None
        self.names = None


class YOLODetector:
    def __init__(self, stream, weights_path,
                 config_path=None,
                 meta_path=None,
                 meta_file="coco.data",
                 show_gui=False,
                 is_tracking=False,
                 darknet_dir="../../darknet",
                 **kwargs):
        self.darknet_dir = darknet_dir
        self.kwargs = kwargs
        self.show_gui = show_gui

        self.stream = stream
        self.config_path = config_path
        self.weights_path = weights_path
        self.meta_path = meta_path
        self.meta_file = meta_file
        self.is_tracking = is_tracking
        if is_tracking:
            from .tracking import SiamMask
            self.sm = SiamMask()
            self.is_pressed_shift = False

        self._check_path()

        # Load neural network
        self.netMain = darknet.load_net_custom(self.config_path.encode(
            "ascii"), self.weights_path.encode("ascii"), 0, 1)  # batch size = 1
        self.input_size = (darknet.network_width(self.netMain), darknet.network_height(self.netMain))

        self._load_meta()
        # Create an image we reuse for each detect
        self.darknet_image = darknet.make_image(darknet.network_width(self.netMain),
                                                darknet.network_height(self.netMain), 3)
        self._reset_pick_classes()

        if os.name != 'nt':
            subprocess.call("notify-send -i %s -t %d %s %s"
                            % ("/home/lab-pc1/nptu/lab/computer_vision/darknets/service/darknet_notext.png",
                               3000, "Darknet服務", "已載入類神經網路模型"), shell=True)

    def _load_meta(self):
        self.metaMain = MetaMain()

        with open(self.meta_path) as metaFH:
            metaContents = metaFH.read()
            names = re.search("names *= *(.*)$", metaContents,
                              re.IGNORECASE | re.MULTILINE)
            classes = re.search("classes *= *(.*)$", metaContents,
                                re.IGNORECASE | re.MULTILINE)
            if classes:
                self.metaMain.classes = int(classes.group(1))
            if names:
                result = names.group(1)
                result = os.path.join(os.path.dirname(__file__), result)
                print(result)
                with open(result, encoding='utf8') as namesFH:
                    namesList = namesFH.read().strip().split("\n")
                    self.metaMain.names = [x.strip() for x in namesList]

    def _check_path(self):
        package_dir = os.path.dirname(__file__)
        self.darknet_dir = os.path.join(package_dir, self.darknet_dir)

        if self.meta_path is None:
            self.meta_path = os.path.join(package_dir, "data", self.meta_file)

        if self.config_path is None:
            weight_name = os.path.basename(self.weights_path)
            self.config_path = "%s.cfg" % os.path.splitext(weight_name)[0]

        if os.path.dirname(self.config_path) == "":
            self.config_path = os.path.join(package_dir, "cfg", self.config_path)

        self.weights_path = os.path.join(self.darknet_dir, self.weights_path)
        print(self.weights_path)

        if not os.path.exists(self.config_path):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(self.config_path) + "`")
        if not os.path.exists(self.weights_path):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(self.weights_path) + "`")
        if not os.path.exists(self.meta_path):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(self.meta_path) + "`")

    def _pick_class(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_LBUTTONDBLCLK:
            for d in self.stream.detections:
                xmin, ymin, xmax, ymax = d["box_xy"]
                if xmin < x < xmax and ymin < y < ymax:
                    self.picked_det = d
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.picked_det is not None:
                self.picked_classes.add(self.picked_det["name"])
        if event == cv2.EVENT_LBUTTONDBLCLK and self.is_tracking and self.picked_det is not None:
            c = self.picked_det["coord"]
            self.sm.init_roi(self.stream.raw, c["x"], c["y"], c["w"], c["h"])
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._reset_pick_classes()

    def _reset_pick_classes(self):
        self.picked_classes = set()
        self.picked_det = None

    def detect_stream(self, save_video=False, pseudo_label=False, video_size=(1280, 720),
                      fps=30.0, fpath="output", interval=1, **kwargs):

        if self.show_gui:
            cv2.namedWindow('Detected', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detected", *video_size)
            cv2.setMouseCallback('Detected', self._pick_class)
        if save_video:
            out = cv2.VideoWriter(
                "%s.mp4" % fpath, cv2.VideoWriter_fourcc(*'MP4V'), fps, video_size)
        while self.stream.raw is None:
            time.sleep(0.001)

        print("Start detecting.")
        try:
            while self.stream.raw is not None:
                with self.stream.lock:
                    prev_time = time.time()
                    frame = np.copy(self.stream.raw)
                    self.stream.detections = self.detect_image(frame)
                    dets = self._check_whitelist()
                    if self.is_tracking and self.picked_det:
                        self.sm.track(frame)
                    print("\nFPS:   %.2f" % (1 / (time.time() - prev_time)))
                    yolo_raw = cv_draw_boxes(dets, frame)
                    self.stream.yolo_raw = cv2.imencode(".jpeg", frame, (cv2.IMWRITE_JPEG_QUALITY, 90))[1]

                    if save_video:
                        # h, w, _ = yolo_raw.shape
                        # if w >= h:
                        #     factor = 1280 / w
                        # else:
                        #     factor = 720 / h
                        # vid_f = cv2.resize(yolo_raw, (1280, 720))

                        out.write(yolo_raw)
                    if self.show_gui:
                        cv2.imshow("Detected", yolo_raw)
                        key = cv2.waitKey(interval)
                        if self.is_tracking and key == 225:
                            self.select_roi()
                        elif key == ord('q'):
                            self.stream.yolo_raw = "quit"
                            break

                time.sleep(0.001)
        finally:
            if save_video:
                out.release()

    def select_roi(self):
        self.picked_det = True
        frame = self.stream.raw.copy()
        x, y, w, h = cv2.selectROI('Detected', frame, False, False)
        self.sm.init_roi(frame, x + w / 2, y + h / 2, w, h)
        cv2.setMouseCallback('Detected', self._pick_class)

    def _check_whitelist(self):
        for d in self.stream.detections:
            if d["name"] in self.picked_classes or \
                    len(self.picked_classes) == 0:
                yield d

    def detect_image(self, img):
        self.frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(self.frame_rgb, self.input_size,
                                   interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())
        detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, self.frame_rgb.shape,
                                          **self.kwargs)
        return detections

    def _pseudo_label(self, label, frame):
        b = self.stream.detections["coord"]
        label_path = os.path.join("data", label, frame)
        cv2.imsave("%s.jpg" % label_path, self.frame_rgb)
        with open("%s.txt" % label_path, 'w') as f:
            f.write("%s %.4f %.4f %.4f %.4f" % (label, b["x"], b["y"], b["w"], b["h"]))
