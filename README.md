# Darknet Video
Extract files from [darknet](https://github.com/AlexeyAB/darknet) to  maintain python wrapper easier.
# Feature
* Video object tracking using [SiamMask](https://github.com/foolwood/SiamMask)
* Global NMS (not each class)
* Object size threshold and 
# Requirement
* Ubuntu 18.04 or 16.04 (16.04 need compiled opencv for unicode labels on result image)
* Put this project in same directory as [darknet](https://github.com/AlexeyAB/darknet)
## Installation
`pip3 install --user -e .`
## Usage
```
from darknet_python.run_detector import ThreadingDetector
# First argument same as cv2.captureVideo() argument, then we use web camera.
# If not pass config_path, loading config of basename of weights by default.
# By default loading meta file of coco.data.
ThreadingDetector(0, weights_path='yolov3.weights')
```

## Labeling assisted by YOLO-SiamMask
Automating labeling use native detection ability of YOLO and SiamMask. 
### To-do list
- [ ] click to select yolo box to candidate label
- [x] frame by frame human check 
- [x] manual bbox selecting
- [ ] adjust yolo predict box