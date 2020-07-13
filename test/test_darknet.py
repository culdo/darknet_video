from darknet_video.thread_detector import ThreadingDetector


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
    hand_weights = "backup/yolov4-hands_best.weights"
    # hand_video = "/home/lab-pc1/nptu/lab/ip_cam/videos/hands/0.mp4"
    mango_img = "mango_dev/*.jpg"
    mango_weights = "backup/yolov4-mango_best.weights"
    mango_config = "enet-mango.cfg"
    mango_data = "mango.data"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # hand_weights = darknet_path % "backup/enet-coco-obj_22000.weights"
    hand_yolo(op3_camera, hand_weights, thresh=0.25, show_gui=True)
    # hand_video = "/home/lab-pc1/nptu/lab/videos/hands/hover/hand_3.mp4"
    # coco_yolo(hand_video, enet_weights, thresh=0.25, show_gui=True, obj_size=[100000, 1000000],
    #           overlap_thresh=0.45, is_labeling=True, data_name="hand_test",
    #           white_list="人", labels_map={0: 2})
    # mango_yolo(wan_phone_ip, mango_weights, thresh=0.25, show_gui=False, is_stream_result=True, obj_size=[60000, 70000],
    #           overlap_thresh=0.45)
