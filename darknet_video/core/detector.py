import os
import re
import subprocess
import time

import cv2
import numpy as np

from darknet_video.core import darknet
from darknet_video.utils.common import cv_draw_boxes, all_nms, cv_draw_text, sort_confid
from darknet_video.utils.labeling import pseudo_label, prewrite_label


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
                 is_stream_result=False,
                 is_labeling=False,
                 only_tracking=False,
                 autoplay=1,
                 is_write_path=True,
                 data_name=None, label_empty=False, label_subset=1, only_one=False, labels_map=(), **kwargs):
        self.labels_map = labels_map
        self.only_one = only_one
        self.label_subset = label_subset
        self.label_empty = label_empty
        self.is_write_path = is_write_path
        self.data_name = data_name
        self.only_tracking = only_tracking
        self.autoplay = autoplay
        self.is_stream_result = is_stream_result
        self.darknet_dir = darknet_dir
        self.kwargs = kwargs
        self.show_gui = show_gui

        self.stream = stream
        if is_labeling or autoplay == 0:
            self.stream.is_lock = True
        if is_labeling:
            self.prev_coord = None
        self.config_path = config_path
        self.weights_path = weights_path
        self.meta_path = meta_path
        self.meta_file = meta_file
        self.is_tracking = is_tracking
        self.is_labeling = is_labeling
        if is_labeling:
            prewrite_label(self.stream, data_name, list(labels_map.values())[0], label_subset, is_write_path)

        if not label_empty:
            if is_tracking:
                from .tracking import SiamMask
                self.sm = SiamMask()
                self._is_pressed_shift = False

            if not only_tracking:
                self._init_yolo()

        self._reset_pick_classes()

        if os.name != 'nt':
            subprocess.call("notify-send -i %s -t %d %s %s"
                            % ("/home/lab-pc1/nptu/lab/computer_vision/darknets/service/darknet_notext.png",
                               3000, "Darknet服務", "已載入類神經網路模型"), shell=True)

    def _init_yolo(self):
        self._check_path()
        # Load neural network
        self.netMain = darknet.load_net_custom(self.config_path.encode(
            "ascii"), self.weights_path.encode("ascii"), 0, 1)  # batch size = 1
        self.input_size = (darknet.network_width(self.netMain), darknet.network_height(self.netMain))
        self._load_meta()
        # Create an image we reuse for each detect
        self.darknet_image = darknet.make_image(darknet.network_width(self.netMain),
                                                darknet.network_height(self.netMain), 3)

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
                result = os.path.join(os.path.dirname(__file__), "..", result)
                print(result)
                with open(result, encoding='utf8') as namesFH:
                    namesList = namesFH.read().strip().split("\n")
                    self.metaMain.names = [x.strip() for x in namesList]

    def _check_path(self):
        package_dir = os.path.join(os.path.dirname(__file__), "..")
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
            self.sm.init_target(self.stream.raw, c["x"], c["y"], c["w"], c["h"])
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._reset_pick_classes()

    def _reset_pick_classes(self):
        self.picked_classes = set()
        self.picked_det = None

    def detect_stream(self, save_video=False, video_size=(1280, 720), overlap_thresh=0.45,
                      fps=30.0, fpath="output", **kwargs):

        if self.show_gui:
            cv2.namedWindow('Detected', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detected", *video_size)
            cv2.setMouseCallback('Detected', self._pick_class)
        if save_video:
            self.out = cv2.VideoWriter(
                "%s.mp4" % fpath, cv2.VideoWriter_fourcc(*'MP4V'), fps, video_size)

        print("Start detecting.")
        try:
            while not self.stream.is_stop:
                # TODO(190716): Figure out better sync between detector and streamer.
                prev_time = time.time()
                self._get_frame()
                # Copy for edit
                img = np.copy(self.frame)
                if not self.label_empty:
                    self._yolo_detect(img, overlap_thresh)
                    self._tracking(img)
                fps = 1 / (time.time() - prev_time)
                print("\nFPS:   %.2f" % fps)
                if self.is_labeling:
                    self.yolo_raw = cv_draw_text(str(self.stream.frame_i), img, offset=70)
                else:
                    self.yolo_raw = cv_draw_text("FPS %.1f" % fps, img)
                if self.is_stream_result:
                    self.stream.yolo_jpg = cv2.imencode(".jpeg", self.yolo_raw, (cv2.IMWRITE_JPEG_QUALITY, 90))[1]

                if save_video:
                    self._save_video()
                if self.show_gui and self._check_gui() == "q":
                    break
                if self.is_labeling:
                    pseudo_label(self.stream, self.data_name, list(self.labels_map.values())[0], self.is_write_path,
                                 is_empty=self.label_empty, labels_map=self.labels_map, label_subset=self.label_subset)
        finally:
            if save_video:
                self.out.release()

    def _tracking(self, img):
        # TODO(190715): Move SiamMask tracking to another thread to accelerate speed
        if self.is_tracking and self.picked_det:
            self.stream.track_box = self.sm.track(img, draw_mask=not self.is_labeling)

    def _yolo_detect(self, img, overlap_thresh):
        if not self.only_tracking:
            dets, nms_box = self.detect_image()
            if len(dets) > 0:
                if self.only_one:
                    dets = sort_confid(dets, nms_box)
                elif overlap_thresh:
                    dets = all_nms(dets, nms_box, overlap_thresh)
                dets = self._check_whitelist(dets)
                cv_draw_boxes(dets, img)
            self.stream.detections = dets
            if self.is_labeling:
                self._check_dets(img)

    # Copy for frame buffer
    def _get_frame(self):

        if self.stream.is_lock:
            # Check1
            self.stream.queue.get()
            # Check2
            self.stream.queue.get()
            self.frame = self.stream.queue.get()
        else:
            self.frame = np.copy(self.stream.raw)

    def _check_gui(self):
        cv2.imshow("Detected", self.yolo_raw)
        key = cv2.waitKey(self.autoplay)
        print("pressed key: %d" % key)
        # enter
        if self.is_tracking:
            if key == ord("t"):
                self.only_tracking = False
                self._tracking_roi()
            elif key == ord("o"):
                self.only_tracking = True
                self._tracking_roi()

        # arrow left
        if key == 81:
            self.stream.is_previous = True
        elif self.is_labeling and key == ord("m"):
            print("No detected boxes, please selecting one manually. ")
            if key == ord("m"):
                self.stream.detections = []
                self.manual_roi()
        elif key == ord(' '):
            if len(self.stream.detections) == 0 and not self.only_tracking and self.is_labeling:
                self.stream.manual_roi = self.prev_coord
            if self.autoplay == 1:
                self.autoplay = 0
            else:
                self.autoplay = 1
        elif key == ord('q'):
            self.stream.is_stop = True
            return "q"

    def _save_video(self):
        h, w, _ = self.yolo_raw.shape
        if w >= h:
            factor = 1280 / w
        else:
            factor = 720 / h
        yolo_raw = cv2.resize(self.yolo_raw, (1280, 720))
        self.out.write(yolo_raw)

    def _tracking_roi(self):
        self.picked_det = True
        self.stream.track_box = cv2.selectROI('Detected', self.frame, False, False)
        if self.stream.track_box != (0, 0, 0, 0):
            x, y, w, h = self.stream.track_box
            self.sm.init_target(self.frame, x + w / 2, y + h / 2, w, h)
            cv2.setMouseCallback('Detected', self._pick_class)

    def manual_roi(self):
        manual_roi = cv2.selectROI('Detected', self.frame, False, False)
        if manual_roi != (0, 0, 0, 0):
            self.stream.manual_roi = manual_roi
            self.prev_coord = manual_roi
        else:
            self.stream.manual_roi = None

    def _check_whitelist(self, dets):
        result = []
        for d in dets:
            if d["name"] in self.picked_classes or \
                    len(self.picked_classes) == 0:
                result.append(d)
        return result

    def detect_image(self):
        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, self.input_size,
                                   interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())
        detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, frame_rgb.shape,
                                          **self.kwargs)
        return detections

    def _check_dets(self, img, color=(0, 255, 0)):
        if len(self.stream.detections) > 0:
            coord = self.stream.detections[0]["coord"]
            x, y, w, h = coord["x"], coord["y"], coord["w"], coord["h"]
            x = round(x - w / 2)
            y = round(y - h / 2)
            self.prev_coord = x, y, w, h
        else:
            cv2.rectangle(img, self.prev_coord, color, 2)
            self.autoplay = 0
