

## loc2bbox()解释

这个函数是对 bbox 使用偏移和缩放尺寸表示的形式转换为 bbox 的左下侧点和右上侧点的坐标。

下面是RCNN中的转换公式：

$$\hat{g}_y = p_h t_y + p_y$$
$$\hat{g}_x = p_w t_x + p_x$$
$$\hat{g}_h = p_h \cdot exp(t_h)$$
$$\hat{g}_w = p_w \cdot exp(t_w)$$

其中， $t_x,t_y,t_h,t_w$ 为缩放尺寸，$p_x,p_y,p_h,p_w$是中心点的坐标。

实际函数编写中形式不一样，是因为代码中使用的坐标是左下侧点和右上侧点，中间还需要进行转换。
