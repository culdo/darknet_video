from darknet_video.thread_detector import ThreadingDetector

outdoor_lab = "rtsp://192.168.0.61:554/user=admin&password=&channel=1&stream=0.sdp?real_stream--rtp-caching=1"
# indoor_lab = "rtsp://203.64.134.168:554/user=admin&password=&channel=1&stream=0.sdp?real_stream--rtp-caching=1"
indoor_lab = "rtsp://192.168.0.60:554/user=admin&password=&channel=1&stream=0.sdp?real_stream--rtp-caching=1"
op3_camera = "http://203.64.134.168:8084/stream?topic=/usb_cam_node/image_raw&type=ros_compressed"
gg_phone_ip = "http://192.168.0.170:8080/video"
phone_ip = "http://192.168.0.249:8080/video"

v4_weights = "bin/yolov4.weights"
v4tiny_weights = "bin/yolov4-tiny.weights"
v3tiny_weights = "bin/yolov3-tiny.weights"
enet_weights = "bin/enet-coco.weights"
hand_weights = "backup/enet-hand.weights"
# hand_video = "/home/lab-pc1/nptu/lab/ip_cam/videos/hands/0.mp4"
hand_video = "hands/%s.mp4"
mango_img = "mango_dev/*.jpg"
mango_weights = "backup/enet-mango.weights"
mango_config = "enet-mango.cfg"
mango_data = "mango.data"


def coco_yolo(url, weights=v4_weights, show_gui=True, **kwargs):
    ThreadingDetector(url,
                      weights_path=weights,
                      meta_file="coco_cht.data",
                      show_gui=show_gui,
                      **kwargs)


def hand_yolo(url, weights=hand_weights, **kwargs):
    ThreadingDetector(url,
                      weights_path=weights,
                      meta_file="hands.data",
                      config_path="enet-coco-obj.cfg",
                      **kwargs)


def mango_yolo(url, weights=mango_weights, **kwargs):
    ThreadingDetector(url,
                      weights_path=weights,
                      meta_file="mango.data",
                      config_path=mango_config,
                      **kwargs)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # hand_weights = darknet_path % "backup/enet-coco-obj_22000.weights"
    # hand_yolo(op3_camera, hand_weights, thresh=0.2, show_gui=True, is_tracking=True)
    coco_yolo(phone_ip, v4_weights, thresh=0.25, show_gui=True, is_stream_result=True, obj_size=[60000, 70000],
              overlap_thresh=0.45)
    # mango_yolo(gg_phone_ip, mango_weights, thresh=0.25, show_gui=True)
