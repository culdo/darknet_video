from darknet_video.core.thread_detector import ThreadingDetector


def coco_yolo(url, weights, show_gui=True, **kwargs):
    ThreadingDetector(url,
                      weights_path=weights,
                      meta_file="coco_cht.data",
                      show_gui=show_gui,
                      **kwargs)


def hands_yolo(url, weights, **kwargs):
    ThreadingDetector(url,
                      weights_path=weights,
                      meta_file="hands.data",
                      config_path="yolov4-hands.cfg",
                      **kwargs)


def hand_yolo(url, weights, **kwargs):
    ThreadingDetector(url,
                      weights_path=weights,
                      meta_file="hand.data",
                      config_path="cross-hands.cfg",
                      **kwargs)


def mango_yolo(url, weights, **kwargs):
    ThreadingDetector(url,
                      weights_path=weights,
                      meta_file="mango.data",
                      config_path=mango_config,
                      **kwargs)


if __name__ == '__main__':
    outdoor_lab = "rtsp://192.168.0.61:554/user=admin&password=&channel=1&stream=0.sdp?real_stream--rtp-caching=1"
    # indoor_lab = "rtsp://203.64.134.168:554/user=admin&password=&channel=1&stream=0.sdp?real_stream--rtp-caching=1"
    indoor_lab = "rtsp://192.168.0.60:554/user=admin&password=&channel=1&stream=0.sdp?real_stream--rtp-caching=1"
    op3_camera = "http://192.168.0.101:8080/stream?topic=/usb_cam_node/image_raw&type=ros_compressed"
    gg_phone_ip = "http://203.64.134.168:8000/video"
    phone_ip = "http://192.168.0.249:8080/video"
    wan_phone_ip = "http://203.64.134.168:8082/video"

    v4_weights = "bin/yolov4.weights"
    v4tiny_weights = "bin/yolov4-tiny.weights"
    v3tiny_weights = "bin/yolov3-tiny.weights"
    enet_weights = "bin/enet-coco.weights"
    # hand_video = "/home/lab-pc1/nptu/lab/ip_cam/videos/hands/0.mp4"
    mango_img = "mango_dev/*.jpg"
    mango_weights = "backup/yolov4-mango_bak.weights"
    mango_config = "enet-mango.cfg"
    mango_data = "mango.data"
    hand_num = -1
    label_subset = 1
    labeling = False
    hands_video = "/home/lab-pc1/nptu/lab/videos/hands/hover/hand_%s-%s.mp4" % (hand_num, label_subset)
    hands_weights = "backup/yolov4-hands_best.weights"
    hand_weights = "bin/cross-hands.weights"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if not labeling:
        hands_yolo(op3_camera, hands_weights, thresh=0.25, show_gui=True, only_one=True)
        # hands_yolo(op3_camera, hands_weights, thresh=0.25, show_gui=True)
        # hands_yolo(hands_video, hands_weights, thresh=0.25, show_gui=True, is_rotate=False, obj_size=[100000, 1000000],
        #           overlap_thresh=0.15, autoplay=0, is_tracking=False, labeling_fps=5,
        #           is_labeling=False, data_name="hand_test",
        #           limit_frames=1448)
    else:
        # coco_yolo(hands_video, v4_weights, thresh=0.10, show_gui=True,
        #           overlap_thresh=0.01, is_rotate=False, autoplay=0, is_tracking=False, only_tracking=False,
        #           labeling_fps=30, obj_size=[150000, 1000000],
        #           is_labeling=True, data_name="hand_test", white_list="人", labels_map={0: hand_num-1},
        #           label_empty=False, label_subset=label_subset,
        #           limit_frames=558, is_write_path=True)
        hand_yolo(hands_video, hand_weights, thresh=0.10, show_gui=True,
                  overlap_thresh=0.01, is_rotate=False, autoplay=0, is_tracking=False, only_tracking=False,
                  labeling_fps=30, only_one=True,
                  is_labeling=True, data_name="hand_test", white_list="hand", labels_map={0: hand_num - 1},
                  label_empty=False, label_subset=label_subset,
                  limit_frames=558, is_write_path=True)
    # mango_yolo(wan_phone_ip, mango_weights, thresh=0.25, show_gui=False, is_stream_result=True, obj_size=[60000, 70000],
    #           overlap_thresh=0.45)
