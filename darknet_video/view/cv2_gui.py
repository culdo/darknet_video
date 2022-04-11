import cv2
def check_cv2_gui(self):
    cv2.imshow("Detected", self.yolo_raw)
    key = cv2.waitKey(self.autoplay)
    print("pressed key: %d" % key)
    # enter
    if self.is_tracking:
        if key == ord("t"):
            self.only_tracking = False
            self._select_tracking_roi()
        elif key == ord("o"):
            self.only_tracking = True
            self._select_tracking_roi()

    # arrow left
    if key == 81:
        self.stream.is_previous = True
    elif self.is_labeling and key == ord("m"):
        print("No detected boxes, please selecting one manually. ")
        if key == ord("m"):
            self.stream.detections = []
            self.manual_roi()
    elif key == ord(' '):
        if len(self.stream.detections) == 0 and not self.only_tracking and self.is_labeling:
            self.stream.manual_roi = self.prev_coord
        if self.autoplay == 1:
            self.autoplay = 0
        else:
            self.autoplay = 1
    elif key == ord('q'):
        self.stream.is_stop = True
        return "q"