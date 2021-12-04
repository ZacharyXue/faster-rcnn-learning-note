# util 学习笔记

## 前言

开始照抄是几天前了，先选择了导入数据集的函数。但是这两天看知乎，上面提到，开始写一个网络的时候最好先定义 `Model`，然后再去写数据处理相关的函数。下次注意吧，这次已经写了一半。

## Python用法总结

### 图像的导入和处理

这里作者选择 `PIL`（`Pillow`） 包中的 [`Image`](https://www.osgeo.cn/pillow/reference/Image.html) 作为图像导入的方法。

#### `PIL.Image.open(fp, mode='r', formats=None)`

参数：
- **fp**：文件名
- **mode**：模式，这个参数必须是 `r`
- **formats**：尝试加载文件的格式列表或者元组（这里指的就是Python的 `list` 和 `tuple`）

#### `Image.convert(mode=None, matrix=None, dither=None, palette=0, colors=256)`

这个函数时返回图像的转换副本。

一般也就用 **mode**参数，这个表示请求的模式，对于模式 `P`，是使用调色盘进行像素转换。

关于这里调色盘的概念，简单讲是通过预设一个表，然后在数据中就可以只是用单通道记录调色盘中的索引了，减少了存储空间的使用。

### Pytorch图像格式

在使用 `Image` 读入数据后其格式为 [H,W,C](height， width，channel)，但是要注意的是 Pytorch 中使用的是 [C,H,W]，[知乎](https://www.zhihu.com/question/310094451/answer/581629970)上关于这一点的猜测是为了后续 CUDA 对于计算的优化。

### `numpy.copy()`

在程序中对于 Numpy 数组使用 `copy()` 进行深拷贝创建独立的数组。

这是因为 Python 中对象的传入类似于C++中的引用，在函数体中的修改会直接影响函数体外的数组值。因此在特定情况下需要使用深拷贝函数来“复制”一份数组。

### `numpy.maximum()`

`numpy.maximum()` 和 `numpy.max()` 的区别在于前者是对于两个矩阵进行比较，而后者是沿某个轴对于一个矩阵进行比较。

```python
>> np.maximum([-2, -1, 0, 1, 2], 0)
array([0, 0, 0, 1, 2])
```

### `numpy.logical_and()`

该函数是对于两个输入进行逻辑判断，与此类似的还有 `_or` `_not` `_xor`。

```python
>> np.logical_and(True, False)
False
>> np.logical_and([True, False], [False, False])
array([False, False])
```

### `slice` 类

使用 `slice` 类多是为了方面后续对于函数的切片操作，具体使用见下：

```python
>>> temp = slice(0,6)
>>> temp
slice(0, 6, None)
>>> temp.start
0
>>> temp.stop
6
>>> dir(temp)
['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'indices', 'start', 'step', 'stop']
>>>
```

### `random`类

在 `util` 中使用的 `random` 类函数 `choice`,该函数是在输入中随机选择一个元素。使用示例为：

```python
y_flip = random.choice([True, False])
```

另一个使用比较多的是 `random.shuffle()`，这个在自己构建训练集的时候会用于打乱训练集样本。

还有 `random.random()`，用于生成0 到 1.0 的一个随机浮点数，而 `random.uniform(a, b)` 等价于 `a + (b-a) * random()`。

