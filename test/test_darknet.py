from darknet_python.video import YOLO

yolo = YOLO("../../darknet/bin/csresnext50-panet-spp-original-optimal.weights")
lab_door_cam = "http://192.168.0.52:8081/"
yolo.detect_mjpeg(lab_door_cam)
