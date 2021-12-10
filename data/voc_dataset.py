import os
import xml.etree.ElementTree as ET
# XML是中结构化数据形式，在ET中使用ElementTree代表整个XML文档，
# 并视其为一棵树，Element代表这个文档树中的单个节点。

import numpy as np

from .util import read_image


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')


class VOCBboxDataset:
    """Bounding box dataset for PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    Args:
        data_dir (string): Path to the root of the training data. 
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.
    """

    def __init__(self, data_dir, split='trainval',
                use_difficult=False, return_different=False,
                ):
        # 大概流程是读取目标数据集的txt文件，然后获得所有的编号，
        # 之后就可以利用编号和默认的位置信息获得数据

        # 获得txt文件地址
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split)
        )
        # 获得txt文件中所有的编号
        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_different = return_different
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Return the i-th example.

        一个很标准的XML数据提取流程

            returns:
                tuple of an image and bounding boxes
        """
        # 获得指定编号
        id_ = self.ids[i]
        # parse：从XML中获得数据
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml')
        )
        bbox = list()
        label = list()
        difficult = list()

        for obj in anno.findall('object'): 
            # findall：查找当前元素下tag或path能够匹配的直系节点

            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                # .text：元素内容
                continue

            difficult.append(int(obj.find('difficult').text))
            # bndbox_anno是XML中存储bbox位置的节点
            bndbox_anno = obj.find('bndbox')
            # 在bndbox_anno节点下找到bbox需要的各个节点，
            # 然后加入bbox列表中
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')
            ])
            name = obj.find('name').text.lower().strip()
            label.append(self.label_names.index(name))
        
        # 使用stack进行堆叠，默认为 axis = 0
        # 这里使用的原因是上面操作时append新元素到list中
        # 而后续需要numpy数组，不断append对numpy来说效率太低，
        # 所以使用list先append然后利用stack堆叠转换为numpy数组
        # 注意这里stack和concatenate的区别：
        #   >>> arrays = [np.random.randn(3, 4) for _ in range(10)]
        #   >>> np.stack(arrays, axis=0).shape
        #   (10, 3, 4)
        #   >>> np.concatenate(arrays, axis=0).shape
        #   (30, 4)
        # stack堆叠后新建了一个axis而concatenate在原有的axis上进行
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        # PyTorch don't support np.bool
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  

        # 读取img数据
        img_file = os.path.join(self.data_dir, 'JPEGImages', id + '.jpg')
        img = read_image(img_file, color=True)

        return img, bbox, label, difficult

    __getitem__ = get_example


