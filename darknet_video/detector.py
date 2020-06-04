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


package_dir = os.path.dirname(__file__)
cfg_dir = os.path.join(package_dir, "cfg")


class YOLO:
    def __init__(self, stream, weights_path,
                 config_path=None,
                 meta_path=None,
                 meta_file="coco.data",
                 thresh=0.25,
                 white_list=None,
                 show_gui=False):
        self.thresh = thresh
        self.show_gui = show_gui
        if meta_path is None:
            meta_path = os.path.join(cfg_dir, meta_file)
        if isinstance(white_list, str):
            self.white_list = [white_list]
        elif isinstance(white_list, list) or white_list is None:
            self.white_list = white_list
        else:
            raise AssertionError("Only Accept str and list.")

        self.stream = stream
        self.config_path = config_path
        self.weights_path = weights_path
        self.meta_path = meta_path

        self._check_path()

        self.netMain = darknet.load_net_custom(self.config_path.encode(
            "ascii"), weights_path.encode("ascii"), 0, 1)  # batch size = 1
        self.input_size = (darknet.network_width(self.netMain), darknet.network_height(self.netMain))

        self._load_meta()
        # Create an image we reuse for each detect
        self.darknet_image = darknet.make_image(darknet.network_width(self.netMain),
                                                darknet.network_height(self.netMain), 3)
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
                result = os.path.join(package_dir, result)
                print(result)
                if os.path.exists(result):
                    with open(result) as namesFH:
                        namesList = namesFH.read().strip().split("\n")
                        self.metaMain.names = [x.strip() for x in namesList]
            else:
                raise AssertionError("Not found names file.")

    def _check_path(self):
        if self.config_path is None:
            weight_name = os.path.basename(self.weights_path)
            self.config_path = os.path.join(cfg_dir, "%s.cfg" % os.path.splitext(weight_name)[0])
        else:
            self.config_path = os.path.join(cfg_dir, "%s.cfg" % self.config_path)

        if not os.path.exists(self.config_path):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(self.config_path) + "`")
        if not os.path.exists(self.weights_path):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(self.weights_path) + "`")
        if not os.path.exists(self.meta_path):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(self.meta_path) + "`")

    def detect_stream(self, save_video=False, pseudo_label=False, video_size=(1280, 720), fps=30.0, fpath="output",
                      **kwargs):

        if self.show_gui:
            cv2.namedWindow('Detected', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detected", *video_size)
        if save_video:
            out = cv2.VideoWriter(
                "%s.mp4" % fpath, cv2.VideoWriter_fourcc(*'mp4v'), fps, video_size)
        while self.stream.raw is None:
            time.sleep(0.001)

        print("Start detecting.")
        while self.stream.raw is not None:
            with self.stream.lock:
                prev_time = time.time()
                frame = np.copy(self.stream.raw)
                self.stream.yolo_raw, self.stream.detections = self.detect_image(frame)
                print("\nFPS:   %.2f" % (1 / (time.time() - prev_time)))
                if save_video:
                    out.write(self.stream.yolo_raw)
                if self.show_gui:
                    cv2.imshow("Detected", self.stream.yolo_raw)
                    key = cv2.waitKey(1)
            time.sleep(0.001)

    def detect_image(self, img):
        self.frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(self.frame_rgb, self.input_size,
                                   interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())
        detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, self.frame_rgb.shape,
                                          thresh=self.thresh, white_list=self.white_list)
        result_gbr = cv_draw_boxes(detections, img)
        return result_gbr, detections

    def _pseudo_label(self, label, frame):
        b = self.stream.detections["relative_coordinates"]
        x, y, w, h = b["center_x"], \
                     b["center_y"], \
                     b["width"], \
                     b["height"]
        label_path = os.path.join("data", label, frame)
        cv2.imsave("%s.jpg" % label_path, self.frame_rgb)
        with open("%s.txt" % label_path, 'w') as f:
            f.write("%s %.4f %.4f %.4f %.4f" % (label, x, y, w, h))
