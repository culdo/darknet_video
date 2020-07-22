import os
import time

import cv2
import numpy as np
from PIL import ImageColor

from darknet_video.utils.nms import py_cpu_nms

box_colormap = ["NAVY", "AQUA", "TEAL", "OLIVE", "GREEN", "LIME", "ORANGE", "RED", "MAROON",
                "FUCHSIA", "PURPLE", "GRAY", "BLUE", "SILVER"]
color_len = len(box_colormap)
ft = cv2.freetype.createFreeType2()
if os.name == 'nt':
    ft.loadFontData(fontFileName="C:\\Windows\\Fonts\\kaiu.ttf", id=0)
else:
    ft.loadFontData(fontFileName='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', id=0)


def convert_back(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def convert_to(xmin, ymin, xmax, ymax):
    x = int(round((xmin + xmax)/2))
    y = int(round((ymin + ymax)/2))
    w = int(round(xmax - xmin))
    h = int(round(ymax - ymin))
    return x, y, w, h


def cv_draw_boxes(detections, img, box_color=None, use_uid=False):
    for detection in detections:
        b = detection["coord"]
        xmin, ymin, xmax, ymax = detection["box_xy"]
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        color = _choose_color(box_color, detection, use_uid)
        cv2.rectangle(img, pt1, pt2, color, 2)
        ft.putText(img=img,
                   text="%s [%.2f] [%d]" % (detection["name"], detection["confidence"] * 100, b["w"] * b["h"]),
                   org=(pt1[0], pt1[1] - 5),
                   fontHeight=30,
                   color=color,
                   thickness=-1,
                   line_type=cv2.LINE_AA,
                   bottomLeftOrigin=True)
    return img


def cv_draw_text(text, img, offset=160):
    ft.putText(img=img,
               text=text,
               org=(img.shape[1] - offset, 0),
               fontHeight=40,
               color=(0, 0, 255),
               thickness=-1,
               line_type=cv2.LINE_AA,
               bottomLeftOrigin=False)
    return img


def all_nms(dets, box_arr, overlapThresh):
    if len(dets) > 0:
        box_arr = np.array(box_arr)
        pick = py_cpu_nms(box_arr, overlapThresh)
        for i, det in enumerate(dets):
            if i in pick:
                yield det


def _choose_color(box_color, detection, use_uid):
    if box_color is None:
        if not use_uid:
            uid = detection["class_id"]
        else:
            uid = detection.state.uid
        color = ImageColor.getrgb(box_colormap[uid % color_len])
    else:
        color = ImageColor.getrgb(box_color)
    color = list(reversed(color))
    return color


def gui_threading(stream):
    for win_name in ["yolo_raw"]:
        while stream.__dict__[win_name] is None:
            time.sleep(0.01)
        img = stream.__dict__[win_name]
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, *reversed(img.shape[:2]))
    while True:
        for win_name in ["yolo_raw"]:
            img = stream.__dict__[win_name]
            cv2.imshow(win_name, img)
        cv2.waitKey(1)
