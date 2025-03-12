# 非官方版本的《SIAM: A parameter-free, Spatial Intersection Attention Module》
This is an unofficial implementation of the paper "SIAM: A parameter-free, Spatial Intersection Attention Module".
Paper Link：https://www.sciencedirect.com/science/article/pii/S0031320324002607

SIAM is a parameter-free, plug-to-play attention module that can be easily integrated into existing CNN architectures.

~~~python
from .siam import SIAM2D
import torch

batch = 2
channel = 2
height = 3
width = 3

input_x = torch.randn(batch, channel, height, width)

out_tensor = SIAM2D(input_x)
~~~


