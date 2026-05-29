import numpy as np
import random
import torchvision.transforms as T
import cv2
import math
from data import transform as base_transform
from utils import is_list, is_dict, get_valid_args


class NoOperation():
    def __call__(self, x):
        return x


class BaseSilTransform():
    def __init__(self, divsor=255.0, img_shape=None):
        self.divsor = divsor
        self.img_shape = img_shape

    def __call__(self, x):
        if self.img_shape is not None:
            s = x.shape[0]
            _ = [s] + [*self.img_shape]
            x = x.reshape(*_)
        return x / self.divsor


class BaseParsingCuttingTransform():
    def __init__(self, divsor=255.0, cutting=None):
        self.divsor = divsor
        self.cutting = cutting

    def __call__(self, x):
        if self.cutting is not None:
            cutting = self.cutting
        else:
            cutting = int(x.shape[-1] // 64) * 10
        if cutting != 0: 
            x = x[..., cutting:-cutting]
        if x.max() == 255 or x.max() == 255.:
            return x / self.divsor
        else:
            return x / 1.0


class BaseSilCuttingTransform():
    def __init__(self, divsor=255.0, cutting=None):
        self.divsor = divsor
        self.cutting = cutting

    def __call__(self, x):
        if self.cutting is not None:
            cutting = self.cutting
        else:
            cutting = int(x.shape[-1] // 64) * 10
        if cutting != 0: 
            x = x[..., cutting:-cutting]
        return x / self.divsor


# **************** Data Agumentation ****************


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            return seq[..., ::-1]


class RandomErasing(object):
    def __init__(self, prob=0.5, sl=0.05, sh=0.2, r1=0.3, per_frame=False):
        self.prob = prob
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.per_frame = per_frame

    def __call__(self, seq):
        if not self.per_frame:
            if random.uniform(0, 1) >= self.prob:
                return seq
            else:
                for _ in range(100):
                    seq_size = seq.shape
                    area = seq_size[1] * seq_size[2]

                    target_area = random.uniform(self.sl, self.sh) * area
                    aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))

                    if w < seq_size[2] and h < seq_size[1]:
                        x1 = random.randint(0, seq_size[1] - h)
                        y1 = random.randint(0, seq_size[2] - w)
                        seq[:, x1:x1+h, y1:y1+w] = 0.
                        return seq
            return seq
        else:
            self.per_frame = False
            frame_num = seq.shape[0]
            ret = [self.__call__(seq[k][np.newaxis, ...])
                   for k in range(frame_num)]
            self.per_frame = True
            return np.concatenate(ret, 0)


class RandomRotate(object):
    def __init__(self, prob=0.5, degree=10):
        self.prob = prob
        self.degree = degree

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            dh, dw = seq.shape[-2:]
            # rotation
            degree = random.uniform(-self.degree, self.degree)
            M1 = cv2.getRotationMatrix2D((dh // 2, dw // 2), degree, 1)
            # affine
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 2, 3, 1)
            seq = [cv2.warpAffine(_[0, ...], M1, (dw, dh))
                for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                 for _ in seq], 0)
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 3, 1, 2)
            return seq


class RandomPerspective(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            h, w = seq.shape[-2:]
            cutting = int(w // 44) * 10
            x_left = list(range(0, cutting))
            x_right = list(range(w - cutting, w))
            TL = (random.choice(x_left), 0)
            TR = (random.choice(x_right), 0)
            BL = (random.choice(x_left), h)
            BR = (random.choice(x_right), h)
            srcPoints = np.float32([TL, TR, BR, BL])
            canvasPoints = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            perspectiveMatrix = cv2.getPerspectiveTransform(
                np.array(srcPoints), np.array(canvasPoints))
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 2, 3, 1)
            seq = [cv2.warpPerspective(_[0, ...], perspectiveMatrix, (w, h))
                   for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                 for _ in seq], 0)
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 3, 1, 2)
            return seq


class RandomAffine(object):
    def __init__(self, prob=0.5, degree=10):
        self.prob = prob
        self.degree = degree

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            dh, dw = seq.shape[-2:]
            # rotation
            max_shift = int(dh // 64 * 10)
            shift_range = list(range(0, max_shift))
            pts1 = np.float32([[random.choice(shift_range), random.choice(shift_range)], [
                              dh-random.choice(shift_range), random.choice(shift_range)], [random.choice(shift_range), dw-random.choice(shift_range)]])
            pts2 = np.float32([[random.choice(shift_range), random.choice(shift_range)], [
                              dh-random.choice(shift_range), random.choice(shift_range)], [random.choice(shift_range), dw-random.choice(shift_range)]])
            M1 = cv2.getAffineTransform(pts1, pts2)
            # affine
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 2, 3, 1)
            seq = [cv2.warpAffine(_[0, ...], M1, (dw, dh))
                   for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                 for _ in seq], 0)
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 3, 1, 2)
            return seq
        
# ******************************************

def Compose(trf_cfg):
    assert is_list(trf_cfg)
    transform = T.Compose([get_transform(cfg) for cfg in trf_cfg])
    return transform


def get_transform(trf_cfg=None):
    if is_dict(trf_cfg):
        transform = getattr(base_transform, trf_cfg['type'])
        valid_trf_arg = get_valid_args(transform, trf_cfg, ['type'])
        return transform(**valid_trf_arg)
    if trf_cfg is None:
        return lambda x: x
    if is_list(trf_cfg):
        transform = [get_transform(cfg) for cfg in trf_cfg]
        return transform
    raise "Error type for -Transform-Cfg-"


class RandomPartBlur():
    def __init__(self, prob=0.5, top_range=(9, 20), bot_range=(29, 40), per_frame=False):
        self.prob = prob
        self.top_range = top_range
        self.bot_range = bot_range
        self.per_frame = per_frame

    def __call__(self, seq):
        '''
        Input:
            seq: a sequence of silhouette frames, [s, h, w]
        Output:
            seq: a sequence of agumented frames, [s, h, w]
        '''
        if not self.per_frame:
            if random.uniform(0, 1) >= self.prob:
                return seq
            else:
                top = random.randint(self.top_range[0], self.top_range[1])
                bot = random.randint(self.bot_range[0], self.bot_range[1])

                seq = seq.transpose(1, 2, 0) # [s, h, w] -> [h, w, s]
                _seq_ = seq.copy()
                _seq_ = _seq_[top:bot, ...]
                _seq_ = cv2.GaussianBlur(_seq_, ksize=(3, 3), sigmaX=0)
                _seq_ = (_seq_ > 0.2).astype(np.float)
                seq[top:bot, ...] = _seq_
                seq = seq.transpose(2, 0, 1) # [h, w, s] -> [s, h, w]

            return seq
        else:
            self.per_frame = False
            frame_num = seq.shape[0]
            ret = [self.__call__(seq[k][np.newaxis, ...]) for k in range(frame_num)]
            self.per_frame = True
            return np.concatenate(ret, 0)