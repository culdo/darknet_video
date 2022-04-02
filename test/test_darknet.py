from darknet_video.core.thread_detector import ThreadingDetector
from darknet_video.utils.get_yt import get_yt_vid


def coco_yolo(url, weights, meta_file="coco_cht.data",show_gui=True, **kwargs):
    ThreadingDetector(url,
                      weights_path=weights,
                      meta_file=meta_file,
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

    phone_ip = "http://192.168.13.246:8080/video"
    yt_video = get_yt_vid("https://www.youtube.com/watch?v=9XPBNaLXzPo")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if not labeling:
        coco_yolo(yt_video, enet_weights, is_realtime=True, is_tracking=False,
                  meta_file="coco_cht.data", thresh=0.25, show_gui=True, is_stream_result=False)
    else:
        pass
