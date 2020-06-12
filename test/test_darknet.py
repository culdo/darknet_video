import os

from darknet_video.thread_detector import ThreadingDetector

lab_camera = "rtsp://192.168.0.60:554/user=admin&password=&channel=1&stream=0.sdp?real_stream--rtp-caching=1"
op3_camera = "http://203.64.134.168:8080/stream?topic=/usb_cam_node/image_raw&type=ros_compressed"
phone_rtsp = "rtsp://192.168.137.41:8080/h264_ulaw.sdp"
phone_ip = "http://192.168.137.124:8080/video"
darknet_path = "../../darknet/%s"

v4_weights = darknet_path % "bin/yolov4.weights"
v3tiny_weights = darknet_path % "bin/yolov3-tiny.weights"
enet_weights = darknet_path % "bin/enet-coco.weights"
hand_weights = darknet_path % "backup/enet-coco-obj_final.weights"
# hand_video = "/home/lab-pc1/nptu/lab/ip_cam/videos/hands/0.mp4"
hand_video = darknet_path % "data/hands/%s.mp4"
mango_img = darknet_path % "data/img/C1-P1_Train/*.jpg"


def coco_yolo(url, weights=v4_weights, show_gui=True, **kwargs):
    ThreadingDetector(url,
                      forever=True,
                      weights_path=weights,
                      meta_file="coco_cht.data",
                      show_gui=show_gui,
                      **kwargs)


def hand_yolo(url, weights=hand_weights, **kwargs):
    ThreadingDetector(url,
                      forever=True,
                      weights_path=weights,
                      meta_file="hands.data",
                      config_path="enet-coco-obj",
                      show_gui=True,
                      **kwargs)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # hand_weights = darknet_path % "backup/enet-coco-obj_22000.weights"
    # hand_yolo("0",hand_weights, thresh=0.2)
    coco_yolo(phone_ip, enet_weights, thresh=0.25, show_gui=True, is_tracking=True)
    # coco_yolo("0", enet_weights, thresh=0.25, show_gui=True, is_tracking=True)
