# A gendar 
## Positional encoding
```
class PositionalEmbedding(nn.Module): # 定义位置嵌入模块

    def __init__(self, d_model, max_len=512): # 初始化函数
        super().__init__() # 调用父类初始化函数

        pe = torch.zeros(max_len, d_model).float() # 创建全0张量，max_len行d_model列
        pe.require_grad = False # 关闭梯度计算

        position = torch.arange(0, max_len).float().unsqueeze(1) # 创建位置索引张量
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp() # 创建除数项张量

        pe[:, 0::2] = torch.sin(position * div_term) # 奇数列使用sin函数
        pe[:, 1::2] = torch.cos(position * div_term) # 偶数列使用cos函数

        pe = pe.unsqueeze(0) # 增加一维，形状变为(1, max_len, d_model)
        self.register_buffer('pe', pe) # 注册位置嵌入缓存

    def forward(self, x): # 前向传播函数
        return self.pe[:, :x.size(1)] # 返回位置嵌入，截取到x的序列长度
```

$$
  PE_(pos, 2i) = sin(pos / 10000^{2i/d_{model}})
$$

pos is the position and i the dimension, 位置编码的每个维度对应一个sinusoid 函数，函数的波长构成一个geometric progression
几何级数从2 $\pi$ , 为什么选择这个函数是因为这个模型很容易通过相对位置，因为任何固定的k offset, $PE_{pos+k}$ 可以通过$PE_{pos}$的
线性组合表示。
文章中也使用了可学习的位置编码positional embeddings,发现这两种方法的效果是差不多的，最终作者选择使用sinusoidal版本，因为这种方法
可以外推（extrapolate）比训练时更长的序列长度。
```python
(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp() # 创建除数项张量

```

```math
  1/10000^{2i/d_{model}} = e^{log^{10000^{-2i/d_{model}}}} = e^{-2i/d_{model}*log^{10000}} = e^{2i*{-log^{10000}/d_{model}}}
```
先不考虑sin/cos变换，模拟偶数列、奇数列的赋值结果
```
pe[:, 0::2] = position * div_term
pe[:, 1::2] = position * div_term
```
可知，position提供了行数据，div_term提供了列数据，同一个position有多个嵌入编码，这些编码再根据奇数偶数来做sin/cos变换，因此，奇数偶数切换跟行轴无关。

ref:
[transformer 模型位置编码](https://zhuanlan.zhihu.com/p/601844632)
