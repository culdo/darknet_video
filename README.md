# Darknet Video
Extract files from [darknet](https://github.com/culdo/darknet) to  maintain python wrapper easier.  
* Only test on Ubuntu 16.04.
* Default linux lib(libdarknet.so) compiled at CUDA-10.0 CuDnn-7.5.0.56
## Installation
`pip install -e .`
## Usage
```
from darknet_python.run_detector import ThreadingDetector
# First argument same as cv2.captureVideo() argument, then we use web camera.
# If not pass config_path, loading config of basename of weights by default.
# By default loading meta file of coco.data.
ThreadingDetector(0, weights_path='yolov3.weights')
```
