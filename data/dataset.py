from __future__ import absolute_import
from __future__ import division
import torch as t
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt


def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + np.array([122.7717, 115.9465, 102.9801]).reshape(3,1,1)
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    # numpy.clip(): 大于max的使其等于max，小于min的使其等于min
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalize(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    # 不太知道这里normalization的均值和方差是怎么得到的
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    # 这里使用的torch.from_numpy() 和 torch.numpy()还是很有用的
    img = normalize(t.from_numpy(img))
    return img.numpy()


def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.
    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.
    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.
    Returns:
        ~numpy.ndarray: A preprocessed image.
    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    # 这里找的缩放比例是最小的缩放比例
    # 缩放比例的范围是将最大边缩放至max_size
    # 或者将最小边放大至min_size
    scale = min(scale1, scale2)

    img = img / 255
    img = sktsf.resize(img, (C, H * scale, W * scale), 
                       mode='reflect', anti_aliasing=False)
    # 关于填充的模式：
    # https://blog.csdn.net/OuDiShenmiss/article/details/105618200
    #   reflect：以边缘值为轴对称填充
    #   linear_ramp：用边缘递减的方式填充
    #   edge：以边缘值填充
    #   wrap：用原数组后面的值填充前面，前面的值填充后面
    #   minimum/median/mean/maximum：以最小值/中值/均值/最大值填充

    # 对图像进行缩放之后对其进行normalize
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalize
    return normalize(img)


class Transform:
    """
    根据对img的要求调整img的大小，并且对img进行normalization，
    因为img尺寸进行了调整，bbox尺寸也要进行调整，最后随机对img
    和bbox进行反转操作
    """
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    # __call__使得类实例对象可以像调用普通函数那样，以“对象名()”的形式使用
    def __call__(self, in_data):
        
        img, bbox, label = in_data
        _, H, W = img.shape
        # 缩放，normalization
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        # 原始图像调整尺寸之后，bbox也要同比例的调整尺寸
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        # 获得数据类，该类初始化时获得相应目录的txt文件
        # 具体读入在调用get_example()时从txt中路径获取
        self.db = VOCBboxDataset(opt.voc_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        # 读取数据
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        # 对数据进行预处理
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        # len()返回的是__len__()中的返回值
        return len(self.db)


class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)