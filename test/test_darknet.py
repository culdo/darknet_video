from darknet_video.run_detector import ThreadingDetector

if __name__ == '__main__':
    door_cam = "http://192.168.0.52:8081/"
    IP_camera = "rtsp://192.168.0.60:554/user=admin&password=&channel=1&stream=0.sdp?real_stream--rtp-caching=1"
    weights_file = "../../darknet/bin/csresnext50-panet-spp-original-optimal.weights"
    ThreadingDetector(IP_camera, weights_path=weights_file, meta_file="coco_cht.data")
