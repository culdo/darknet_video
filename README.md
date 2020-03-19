# Darknet Python Wrapper
Extract files from [darknet](https://github.com/culdo/darknet) to  maintain python wrapper easier.  
Only test on Ubuntu 16.04.
## Installation
`pip install -e .`
## Usage
```
from darknet_python.video import YOLO
yolo = YOLO("../../darknet/bin/csresnext50-panet-spp-original-optimal.weights")
yolo.detect_video()
```
