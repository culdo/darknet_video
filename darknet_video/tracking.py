import os

import SiamMask.experiments.siammask_sharp as model
from SiamMask.experiments.siammask_sharp.custom import Custom
from SiamMask.tools.test import *
from SiamMask.utils.config_helper import load_config
from SiamMask.utils.load_helper import load_pretrain


class SiamMask:
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
        model_path = os.path.dirname(model.__file__)
        parser.add_argument('--resume', default=os.path.join(model_path, 'SiamMask_DAVIS.pth'), type=str,
                            metavar='PATH', help='path to latest checkpoint (default: none)')
        parser.add_argument('--config', dest='config', default=os.path.join(model_path, 'config_davis.json'),
                            help='hyper-parameter of SiamMask in json format')
        parser.add_argument('--start-frame', default=0, type=int, help='start frame')
        parser.add_argument('--cpu', action='store_true', help='cpu mode')
        args = parser.parse_args()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True
        # Setup Model
        self.cfg = load_config(args)
        siammask = Custom(anchors=self.cfg['anchors'])
        self.siammask = load_pretrain(siammask, args.resume)

        self.siammask.eval().to(self.device)
        self.state = None

    def init_target(self, frame, x, y, w, h):
        target_pos = np.array([x, y])
        target_sz = np.array([w, h])
        self.state = siamese_init(frame, target_pos, target_sz, self.siammask, self.cfg['hp'], device=self.device)

    def track(self, img, draw_rect=False):
        if self.state is not None:
            self.state = siamese_track(self.state, img, mask_enable=True, refine_enable=True,
                                       device=self.device)  # track
            location = self.state['ploygon'].flatten()
            mask = self.state['mask'] > self.state['p'].seg_thr
            img[:, :, 2] = (mask > 0) * 255 + (mask == 0) * img[:, :, 2]
            if draw_rect:
                bbox = cv2.boundingRect(np.uint8(mask))
                cv2.rectangle(img, bbox, (0, 255, 0))
                return bbox

    def _cb_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x, y, w, h = cv2.selectROI('SiamMask', self.frame, False, False)
            self.init_target(self.frame, x, y, w, h)

    def run_along(self, url):
        cap = cv2.VideoCapture(url)

        cv2.namedWindow('SiamMask', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('SiamMask', self._cb_mouse)

        while cap.isOpened():
            self.frame = cap.read()
            self.track(self.frame)
            cv2.imshow('SiamMask', self.frame)
            key = cv2.waitKey(1)
            if key > 0:
                break


if __name__ == '__main__':
    phone_ip = "http://192.168.0.249:8080/video"
    sm = SiamMask()
    sm.run_along(phone_ip)
