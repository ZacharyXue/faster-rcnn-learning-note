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
    """
    根据要求的输出尺寸和img的输入尺寸调整bbox的尺寸，
    是Fast RCNN从SPPNet中引入的。
    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]

    return bbox


def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    # TODO: 不太清楚这个函数做什么
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
    # TODO: 不太清楚这部分的作用
    t, b, _ = _slice_to_bounds(y_slice)
    l, r, _ = _slice_to_bounds(x_slice)
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


def _slice_to_bounds(slice_):
    """
    对于所取区域是否超出临界进行判断
    """
    if slice_ is None:
        return 0, np.inf
    
    if slice_.start is None:
        l = 0
    else:
        l = slice_.start

    if slice_.stop is None:
        u = np.inf
    else:
        u = slice_.stop

    return l, u


def translate_bbox(bbox, y_offset=0, x_offset=0):
    """
    这个函数主要是对于输入进行偏移
    """
    out_bbox = bbox.copy()
    out_bbox[:, :2] += (y_offset, x_offset)
    out_bbox[:, 2:] += (y_offset, x_offset)

    return out_bbox


def random_flip(img, y_random=False, x_random=False,
                return_param=False, copy=False):
    """
    对图像进行随机的翻转
    """
    y_flip, x_flip = False, False

    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]
    
    # 这里copy的意义应该是在于传入的img和函数外接收的img虽然值相同
    # 但是两个是相互独立的
    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img