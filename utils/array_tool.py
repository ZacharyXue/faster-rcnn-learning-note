"""
tools to convert specified type
"""
import torch as t
import numpy as np


def tonumpy(data):
    # 注意isinstance()是用来判断类型的
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, t.Tensor):
        # detach(): 阻断反向传播
        # cpu(): 将变量放在CPU上
        return data.detach().cpu().numpy()


def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = t.from_numpy(data)
    if isinstance(data, t.Tensor):
        tensor = data.detach()
    # 从上面对比来看from_numpy()后得到的是没有grad的tensor

    if cuda:
        tensor = tensor.cuda()
    return tensor
    

def scalar(data):
    # 输入数据只有一个值
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, t.Tensor):
        # item()方法返回张量元素的值，张量中只有一个元素
        # 才能调用该方法，多个元素会报错
        return data.item()