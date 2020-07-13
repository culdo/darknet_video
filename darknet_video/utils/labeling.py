import os
from random import random

import cv2

from darknet_video.utils.common import convert_to


def pseudo_label(stream, data_name, label_id, labels_map=(), val_split=0.2):
    if random() >= val_split:
        data_subset = "train"
    else:
        data_subset = "val"
    data_dir = os.path.join(os.path.dirname(__file__), "../../../darknet", "data", data_name)
    data_list = os.path.join(data_dir, data_subset) + ".txt"
    label_path = os.path.join(data_dir, data_subset, "%s_%s" % (label_id, int(stream.frame_i)))
    cv2.imwrite(label_path + ".jpg", stream.raw)
    im_h, im_w = stream.raw.shape[:2]
    with open(label_path + ".txt", 'w') as f:
        for b in stream.detections:
            class_id = b["class_id"]
            x, y, w, h = (b["coord"]["x"] / im_w, b["coord"]["y"] / im_h,
                          b["coord"]["w"] / im_w, b["coord"]["h"] / im_h)
            if class_id in labels_map:
                class_id = labels_map[class_id]
            f.write("%d %.4f %.4f %.4f %.4f\n" % (class_id, x, y, w, h))
        if stream.track_box:
            x, y, w, h = convert_to(*stream.track_box)
            f.write("%d %.4f %.4f %.4f %.4f\n" % (label_id, x, y, w, h))

    darknet_label_path = os.path.join("data", data_name, data_subset, "%s_%s" % (label_id, int(stream.frame_i)))
    with open(data_list, 'a') as f:
        f.write(darknet_label_path + ".jpg\n")


def recreate_dir(data_dir):
    if os.path.exists(data_dir):
        os.rmdir(data_dir)
    os.makedirs(data_dir)
