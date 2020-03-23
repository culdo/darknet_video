import time

import cv2
from PIL import ImageColor

box_colormap = ["NAVY", "BLUE", "AQUA", "TEAL", "OLIVE", "GREEN", "LIME", "ORANGE", "RED", "MAROON",
                "FUCHSIA", "PURPLE", "GRAY", "SILVER"]
color_len = len(box_colormap)
ft = cv2.freetype.createFreeType2()  # 需要安装freetype模块 cv2' has no attribute 'freetype'
ft.loadFontData(fontFileName='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', id=0)


def convert_back(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cv_draw_boxes(detections, img, box_color=None):
    for detection in detections:
        b = detection["relative_coordinates"]
        x, y, w, h = b["center_x"], \
                     b["center_y"], \
                     b["width"], \
                     b["height"]
        xmin, ymin, xmax, ymax = convert_back(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        if box_color is None:
            color = ImageColor.getrgb(box_colormap[detection["class_id"] % color_len])
        else:
            color = ImageColor.getrgb(box_color)
        cv2.rectangle(img, pt1, pt2, color, 1)
        ft.putText(img=img,
                   text="%s [%s]" % (detection["name"], round(detection["confidence"] * 100, 2)),
                   org=(pt1[0], pt1[1] - 5),
                   fontHeight=20,
                   color=color,
                   thickness=1,
                   line_type=cv2.LINE_AA,
                   bottomLeftOrigin=True)
    return img


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
