import os

from darknet_video.thread_detector import ThreadingDetector

lab_camera = "rtsp://192.168.0.60:554/user=admin&password=&channel=1&stream=0.sdp?real_stream--rtp-caching=1"
op3_camera = "http://203.64.134.168:8080/stream?topic=/usb_cam_node/image_raw&type=ros_compressed"
phone_rtsp = "rtsp://192.168.137.41:8080/h264_ulaw.sdp"
phone_ip = "http://192.168.0.249:8080/video"

v4_weights = "yolov4.weights"
v3tiny_weights = "yolov3-tiny.weights"
enet_weights = "enet-coco.weights"
hand_weights = "backup/enet-coco-obj_final.weights"
# hand_video = "/home/lab-pc1/nptu/lab/ip_cam/videos/hands/0.mp4"
hand_video = "hands/%s.mp4"
mango_img = "mango_dev/*.jpg"
mango_weights = "mango/608v4weights/yolov4_final.weights"
mango_config = "yolov4-mango.cfg"
mango_data = "mango.data"


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
                      config_path="enet-coco-obj.cfg",
                      **kwargs)


def mango_yolo(url, weights=mango_weights, **kwargs):
    ThreadingDetector(url,
                      forever=True,
                      weights_path=weights,
                      meta_file="mango.data",
                      config_path=mango_config,
                      **kwargs)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # hand_weights = darknet_path % "backup/enet-coco-obj_22000.weights"
    # hand_yolo("0",hand_weights, thresh=0.2)
    # coco_yolo(phone_ip, enet_weights, thresh=0.25, show_gui=True, is_tracking=True)
    mango_yolo(mango_img, mango_weights, save_video=True)
