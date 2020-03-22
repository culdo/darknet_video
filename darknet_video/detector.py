import os
import re
import time

import cv2

from darknet_video.utils import cvDrawBoxes
from . import darknet


class MetaMain:
    def __init__(self):
        self.classes = None
        self.names = None


class SharedObject:
    def __init__(self):
        image = None

package_dir = os.path.dirname(__file__)
cfg_dir = os.path.join(package_dir, "cfg")


class YOLO:
    def __init__(self, stream, weight_path,
                 config_path=None,
                 meta_path=os.path.join(cfg_dir, "coco.data"), cv_window_size=(1100, 960)):

        self.stream = stream
        self.config_path = config_path
        self.weight_path = weight_path
        self.meta_path = meta_path

        self._check_path()

        self.netMain = darknet.load_net_custom(self.config_path.encode(
            "ascii"), weight_path.encode("ascii"), 0, 1)  # batch size = 1
        self.input_size = (darknet.network_width(self.netMain), darknet.network_height(self.netMain))

        self._load_meta()
        # Create an image we reuse for each detect
        self.darknet_image = darknet.make_image(darknet.network_width(self.netMain),
                                                darknet.network_height(self.netMain), 3)
        cv2.namedWindow('Detected', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detected', *cv_window_size)

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
            weight_name = os.path.basename(self.weight_path)
            self.config_path = os.path.join(cfg_dir, "%s.cfg" % os.path.splitext(weight_name)[0])
        if not os.path.exists(self.config_path):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(self.config_path) + "`")
        if not os.path.exists(self.weight_path):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(self.weight_path) + "`")
        if not os.path.exists(self.meta_path):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(self.meta_path) + "`")

    def detect_stream(self):
        while self.stream.got_frame is None:
            time.sleep(0.001)
        cv2.resizeWindow('Detected', *reversed(self.stream.got_frame.shape[:2]))
        while self.stream.got_frame is not None:
            prev_time = time.time()
            _, result_image = self.detect_image(self.stream.got_frame)
            print("FPS: %.2f" % (1 / (time.time() - prev_time)))
            cv2.imshow('Detected', result_image)
            cv2.waitKey(1)

    def detect_image(self, frame_read, thresh=0.5):
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.input_size,
                                   interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())
        detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, frame_rgb.shape,
                                          thresh=thresh)
        result_img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        result_gbr = cvDrawBoxes(detections, result_img)
        return detections, result_gbr
