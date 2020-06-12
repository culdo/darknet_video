import os
import time

import cv2
from PIL import ImageColor

box_colormap = ["NAVY", "BLUE", "AQUA", "TEAL", "OLIVE", "GREEN", "LIME", "ORANGE", "RED", "MAROON",
                "FUCHSIA", "PURPLE", "GRAY", "SILVER"]
color_len = len(box_colormap)
ft = cv2.freetype.createFreeType2()  # 需要安装freetype模块 cv2' has no attribute 'freetype'
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
