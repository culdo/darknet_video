import os

import cv2


def pseudo_label(stream, data_name, label_id, is_empty=False, labels_map=(), label_subset=1):
    if stream.frame_i in stream.val_set:
        data_subset = "val"
    else:
        data_subset = "train"
    data_dir = os.path.join(os.path.dirname(__file__), "../../../darknet", "data", data_name)
    label_path = os.path.join(data_subset, "%s-%s_%s" % (label_id, label_subset, int(stream.frame_i)))
    # TODO(190717): Prevent to duplicated write.
    write_path = os.path.join(data_dir, label_path)
    cv2.imwrite(write_path + ".jpg", stream.raw)
    im_h, im_w = stream.raw.shape[:2]

    # TODO(190716): Make priority of  manual selecting ROI ,tracking ROI and YOLO box have better user experience
    with open(write_path + ".txt", 'w') as f:
        if is_empty:
            f.write("")
        elif stream.manual_roi:
            label_rect(f, im_h, im_w, label_id, stream.manual_roi)
            stream.manual_roi = None
        elif stream.track_box:
            label_rect(f, im_h, im_w, label_id, stream.track_box)
        elif len(stream.detections) > 0:
            for b in stream.detections:
                class_id = b["class_id"]
                # convert back to label format that values between 0 ~ 1
                x, y, w, h = (b["coord"]["x"] / im_w, b["coord"]["y"] / im_h,
                              b["coord"]["w"] / im_w, b["coord"]["h"] / im_h)
                if class_id in labels_map:
                    class_id = labels_map[class_id]
                f.write("%d %.4f %.4f %.4f %.4f\n" % (class_id, x, y, w, h))
        else:
            f.write("")


def prewrite_label(stream, data_name, label_id, label_subset, is_write_path):
    data_dir = os.path.join(os.path.dirname(__file__), "../../../darknet", "data", data_name)
    if is_write_path:
        def _write_path(data_subset):
            data_list = os.path.join(data_dir, data_subset) + ".txt"
            with open(data_list, "a+") as f:
                for i in range(stream.frame_count):
                    if i % stream.jump_frames == 0 and (i in stream.val_set and data_subset == "val") or (
                            i not in stream.val_set and data_subset == "train"):
                        # Write label path
                        darknet_label_path = os.path.join("data", data_name, data_subset,
                                                          "%s-%s_%s" % (label_id, label_subset, i) + ".jpg")
                        f.write(darknet_label_path + "\n")

        _write_path("train")
        _write_path("val")
        print("Pre-wrote label.")


def label_rect(f, im_h, im_w, label_id, roi):
    x, y, w, h = roi
    x = x + w / 2
    y = y + h / 2
    f.write("%d %.4f %.4f %.4f %.4f\n" % (label_id, x / im_w, y / im_h, w / im_w, h / im_h))


def recreate_dir(data_dir):
    if os.path.exists(data_dir):
        os.rmdir(data_dir)
    os.makedirs(data_dir)
