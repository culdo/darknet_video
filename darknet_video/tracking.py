import os

import torch
from SiamMask.tools.test import *
import SiamMask.experiments.siammask_sharp as model


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
        from experiments.siammask_sharp.custom import Custom
        siammask = Custom(anchors=self.cfg['anchors'])
        self.siammask = load_pretrain(siammask, args.resume)

        self.siammask.eval().to(self.device)
        self.state = None

    def init_roi(self, frame, x, y, w, h):
        target_pos = np.array([x, y])
        target_sz = np.array([w, h])
        self.state = siamese_init(frame, target_pos, target_sz, self.siammask, self.cfg['hp'], device=self.device)

    def track(self, img):
        if self.state is not None:
            self.state = siamese_track(self.state, img, mask_enable=True, refine_enable=True, device=self.device)  # track
            location = self.state['ploygon'].flatten()
            mask = self.state['mask'] > self.state['p'].seg_thr
            img[:, :, 2] = (mask > 0) * 255 + (mask == 0) * img[:, :, 2]
