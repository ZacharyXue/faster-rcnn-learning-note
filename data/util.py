import numpy as np
from PIL import Image
import random

def read_image(path, dtype=np.float32, color=True):
    """
    读入图片
    
    要注意的是这里使用Image读入图片的格式是：[H,W,C](height， width，channel)，
    但是在pytorch中使用的时候往往为[C,H,W]，为了后续方面，这里进行了转换。
    至于为什么pytorch中使用[C,H,W]呢？可以参考：
        https://www.zhihu.com/question/310094451/answer/581629970
        简单讲，是为了后续优化。
    """

    f = Image.open(path)

    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('p')
        img = np.asarray(img, dtype=dtype)
    # 无论try语句中是否抛出异常，finally中的语句一定会被执行
    finally:
        if hasattr(f, 'close'):
            f.close()
        
    if img.ndim == 2:
        return img[np.newaxis]
    else:
        # [H,W,C]-->[C,H,W]
        return img.transpose((2,0,1))


def resize_bbox(bbox, in_size, out_size):

    # TODO: 没有搞明白这个函数的意义

    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]

    return bbox


def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    
    H, W = size
    bbox = bbox.copy()
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    
    return bbox


def crop_bbox(bbox, y_slice=None, x_slice=None,
              allow_outside_center=True, return_param=False):
    t, b, _ = y_slice.indices(0)
    l, r, _ = x_slice.indices(0)
    crop_bb = np.array((t, l, b, r))

    if allow_outside_center:
        mask = np.ones(bbox.shape[0], dtype=bool)
    else:
        center = (bbox[:, :2]) / 2.0
        mask = np.logical_and(crop_bb[:2] <= center, center < crop_bb[2:]).all(axis=1)

    bbox = bbox.copy()
    bbox[:, :2] = np.maximum(bbox[:, :2], crop_bb[:2])
    bbox[:, 2:] = np.minimum(bbox[:, 2:], crop_bb[2:])
    bbox[:, :2] -= crop_bb[:2]
    bbox[:, 2:] -= crop_bb[:2]

    mask = np.logical_and(mask, (bbox[:, :2] < bbox[:, 2:]).all(axis=1))
    bbox = bbox[mask]

    if return_param:
        return bbox, {'index': np.flatnonzero(mask)}
    else:
        return bbox
