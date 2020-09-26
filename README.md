# Darknet Video
* Extract python APIs from [darknet](https://github.com/AlexeyAB/darknet) to maintain python wrapper easier.
## Object Detection and Tracking
<img src="docs/demo.gif" width="80%"/>

## Robot Interaction
Use with [python-OP3](https://github.com/culdo/python-OP3)
### Gesture Recognition
<img src="docs/robot_gesture.gif" width="80%"/>

### Head Tracking
<img src="docs/robot_tracking.gif" width="80%"/>

# Feature
* Video object tracking using [SiamMask](https://github.com/foolwood/SiamMask)
* Global NMS for all object (not each class)
* Object size threshold and class filtering
* MJPEG streaming for YOLO+SiamMask result on port 8091 
* Auto labeling powered by YOLO+SiamMask.
# Requirement
* Ubuntu 18.04 or 16.04 
* Ubuntu 16.04 need compiled opencv for unicode labels on result image
* Put this project in same directory as [darknet](https://github.com/AlexeyAB/darknet)
## Development Installation
`pip3 install --user -e .`
## Usage
```
from darknet_python.run_detector import ThreadingDetector
# First argument same as cv2.captureVideo() argument, here we use web camera.
# If not pass config_path, loading config of basename of weights by default.
# By default loading meta file of coco.data.
ThreadingDetector(0, weights_path='yolov3.weights')
```
You can find additional examples on `test/test_darknet.py`. 

## Labeling assisted by YOLO-SiamMask
Automating labeling use native detection ability of YOLO and SiamMask. 
### To-do list
- [ ] click to select yolo box to candidate label
- [x] frame by frame human check 
- [x] manual bbox selecting
- [ ] adjust yolo predict box
