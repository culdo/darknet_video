from darknet_video.core.thread_detector import ThreadingDetector


def coco_yolo(url, weights, show_gui=True, **kwargs):
    ThreadingDetector(url,
                      weights_path=weights,
                      meta_file="coco_cht.data",
                      show_gui=show_gui,
                      **kwargs)


def hand_yolo(url, weights, **kwargs):
    ThreadingDetector(url,
                      weights_path=weights,
                      meta_file="hands.data",
                      config_path="yolov4-hands.cfg",
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
    hand_weights = "backup/yolov4-hands_bak.weights"
    # hand_video = "/home/lab-pc1/nptu/lab/ip_cam/videos/hands/0.mp4"
    mango_img = "mango_dev/*.jpg"
    mango_weights = "backup/yolov4-mango_best.weights"
    mango_config = "enet-mango.cfg"
    mango_data = "mango.data"
    hand_num = 3
    hand_video = "/home/lab-pc1/nptu/lab/videos/hands/hover/hand_%d.mp4" % hand_num
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # hand_yolo(phone_ip, hand_weights, thresh=0.25, show_gui=True)
    hand_yolo(0, hand_weights, thresh=0.25, show_gui=True, is_rotate=True, obj_size=[100000, 1000000],
              overlap_thresh=0.15, autoplay=0, is_tracking=False,
              is_labeling=False, data_name="hand_test", labels_map={0: hand_num - 1, 1: hand_num - 1, 2: hand_num - 1},
              limit_frames=592)
    # coco_yolo(hand_video, v4_weights, thresh=0.10, show_gui=True, obj_size=[150000, 1000000],
    #           overlap_thresh=0.01, is_rotate=True, autoplay=1, is_tracking=False,
    #           is_labeling=True, data_name="hand_test", white_list="人", labels_map={0: hand_num-1},
    #           limit_frames=558)
    # mango_yolo(wan_phone_ip, mango_weights, thresh=0.25, show_gui=False, is_stream_result=True, obj_size=[60000, 70000],
    #           overlap_thresh=0.45)
