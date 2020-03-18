import os
import re
import time

import cv2
import numpy as np
import requests

from . import darknet


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0], \
                     detection[2][1], \
                     detection[2][2], \
                     detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0] +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


package_dir = os.path.dirname(__file__)
cfg_dir = os.path.join(package_dir, "cfg")


class MetaMain:
    def __init__(self):
        self.classes = None
        self.names = None


class YOLO:
    def __init__(self, weight_path,
                 config_path=None,
                 meta_path=os.path.join(cfg_dir, "coco.data")):

        self.config_path = config_path
        self.weight_path = weight_path
        self.meta_path = meta_path

        self._check_path()

        self.netMain = darknet.load_net_custom(self.config_path.encode(
            "ascii"), weight_path.encode("ascii"), 0, 1)  # batch size = 1
        self._load_meta()
        # Create an image we reuse for each detect
        self.darknet_image = darknet.make_image(darknet.network_width(self.netMain),
                                                darknet.network_height(self.netMain), 3)
        cv2.namedWindow('Demo', cv2.WINDOW_NORMAL)

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

    def video_capture(self, save_video=False):
        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture("test.mp4")
        cap.set(3, 1280)
        cap.set(4, 720)
        if save_video:
            out = cv2.VideoWriter(
                "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
                (darknet.network_width(self.netMain), darknet.network_height(self.netMain)))
        print("Starting the YOLO loop...")

        while True:
            ret, frame_read = cap.read()
            prev_time = time.time()
            detections, result_image = self.detect_image(frame_read)
            print(1 / (time.time() - prev_time))
            cv2.imshow('Demo', result_image)
            cv2.waitKey(3)
        cap.release()
        if save_video:
            out.release()

    def detect_image(self, frame_read, thresh=0.95):
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(self.netMain),
                                    darknet.network_height(self.netMain)),
                                   interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())
        detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, frame_rgb.shape,
                                          thresh=thresh)
        print(detections)
        result_img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        result_gbr = cvDrawBoxes(detections, result_img)
        return detections, result_gbr

    def detect_mjpeg(self, img_url):
        r = requests.get(img_url, stream=True)

        if r.status_code == 200:
            mybytes = bytes()
            for chunk in r.iter_content(chunk_size=512):
                mybytes += chunk
                a = mybytes.find(b'\xff\xd8')
                b = mybytes.find(b'\xff\xd9')

                if a != -1 and b != -1:
                    if not a < (b + 2):
                        # flush to head flag to find correct range
                        mybytes = mybytes[a:]
                    else:
                        jpg = mybytes[a:b + 2]
                        frame_read = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        prev_time = time.time()
                        detections, result_gbr = self.detect_image(frame_read)
                        cv2.imshow('Demo', result_gbr)
                        cv2.waitKey(1)
                        print("FPS: %.2f" % (1 / (time.time() - prev_time)))

                        # Clear mybytes buffer to prevent internal bound shift
                        mybytes = bytes()


if __name__ == "__main__":
    yolo = YOLO()
    lab_door_cam = "http://192.168.0.52:8081/"
    yolo.detect_mjpeg(lab_door_cam)
