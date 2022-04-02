from darknet_video.core.thread_detector import ThreadingDetector
from darknet_video.utils.get_yt import get_yt_vid


def coco_yolo(url, weights, meta_file="coco_cht.data", show_gui=True, **kwargs):
    ThreadingDetector(url,
                      weights_path=weights,
                      meta_file=meta_file,
                      show_gui=show_gui,
                      **kwargs)


if __name__ == '__main__':

    v4_weights = "yolov4.weights"
    v4tiny_weights = "yolov4-tiny.weights"
    v3tiny_weights = "yolov3-tiny.weights"
    enet_weights = "enet-coco.weights"
    labeling = False

    phone_ip = "http://192.168.13.246:8080/video"
    yt_video = get_yt_vid("https://www.youtube.com/watch?v=9XPBNaLXzPo")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if not labeling:
        coco_yolo(yt_video, enet_weights, is_realtime=True, is_tracking=False,
                  meta_file="coco_cht.data", thresh=0.25, show_gui=True, is_stream_result=False)
    else:
        pass
