import torch
from torch import nn
import torch.nn.functional as F


# class DSAM_Block(nn.Module):
#     def __init__(self, in_channel):
#         super(DSAM_Block, self).__init__()
#         self.cubic_11 = cubic_attention(in_channel // 2, group=1, kernel=11)
#         self.cubic_7 = cubic_attention(in_channel // 2, group=1, kernel=11)
#         self.pool_att = SpecAtte(in_channel)
#
#     def forward(self, x):
#         out = self.pool_att(x)
#         out = torch.chunk(out, 2, dim=1)
#         out_11 = self.cubic_11(out[0])
#         out_7 = self.cubic_7(out[1])
#         out = torch.cat((out_11, out_7), dim=1)
#         out += x  # 添加残差连接
#
#         return out

class SpecAtte_Block(nn.Module):
    def __init__(self, in_channel):
        super(SpecAtte_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.pool_att = SpecAtte(in_channel)

    def forward(self, x):
        out = self.conv1(x)
        out1 = self.bn1(out)
        # print(f'out1.shape = {out1.shape}')
        out = self.pool_att(out1)
        # print(f'out.shape = {out.shape}')
        out = x + out  # 添加残差连接

        return out


# 实现了一种立方体注意力机制，通过水平和垂直方向的注意力机制增强输入特征
class cubic_attention(nn.Module):
    def __init__(self, dim, group, kernel) -> None:
        super().__init__()

        self.H_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta


# 实现了一种条形池化的空间注意力机制，通过水平或垂直方向的池化和卷积操作来计算注意力权重，然后将其应用到输入特征图上，从而实现增强特征的效果
class spatial_strip_att(nn.Module):
    def __init__(self, dim, kernel=5, group=2, H=True) -> None:
        super().__init__()
        self.k = kernel
        pad = kernel // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel // 2, 1) if H else (1, kernel // 2)

        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group * kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(dim)  # 添加归一化层

    def forward(self, x):
        filter = self.ap(x)
        filter = self.conv(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel).reshape(n, self.group, c // self.group, self.k, h * w)

        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1 // self.k, self.k, p * q).unsqueeze(2)
        filter = self.filter_act(filter)

        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)

        return out


# 结合了全局和局部（不同核大小）池化的注意力机制，通过将各自的注意力结果相加并进行卷积操作来生成最终的特征图
class SpecAtte(nn.Module):
    def __init__(self, k) -> None:
        super().__init__()

        self.global_att = GlobalPoolStripAttention(k)
        self.local_att_7 = LocalPoolStripAttention(k, kernel=7)
        self.local_att_11 = LocalPoolStripAttention(k, kernel=11)
        self.conv = nn.Conv2d(k, k, 1)

    def forward(self, x):
        global_out = self.global_att(x)
        local_7_out = self.local_att_7(x)
        local_11_out = self.local_att_11(x)

        out = global_out + local_7_out + local_11_out

        return self.conv(out)


# 在水平和垂直方向上分别进行全局池化操作来捕获全局注意力
class GlobalPoolStripAttention(nn.Module):
    def __init__(self, k) -> None:
        super().__init__()

        self.channel = k

        self.vert_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.vert_high = nn.Parameter(torch.zeros(k, 1, 1))

        self.hori_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.hori_high = nn.Parameter(torch.zeros(k, 1, 1))

        self.vert_pool = nn.AdaptiveAvgPool2d((1, None))
        self.hori_pool = nn.AdaptiveAvgPool2d((None, 1))

        self.gamma = nn.Parameter(torch.zeros(k,1,1))
        self.beta = nn.Parameter(torch.ones(k,1,1))

    def forward(self, x):

        hori_l = self.hori_pool(x) # 1,3,10,1
        hori_h = x - hori_l

        hori_out = self.hori_low * hori_l + (self.hori_high + 1.) * hori_h
        vert_l = self.vert_pool(hori_out) # 1,3,1,10
        vert_h = hori_out - vert_l

        vert_out = self.vert_low * vert_l + (self.vert_high + 1.) * vert_h

        return x * self.beta + vert_out * self.gamma


# 使用水平和垂直方向上的局部池化操作来进行注意力机制
class LocalPoolStripAttention(nn.Module):
    def __init__(self, k, kernel=7) -> None:
        super().__init__()

        self.channel = k

        self.vert_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.vert_high = nn.Parameter(torch.zeros(k, 1, 1))

        self.hori_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.hori_high = nn.Parameter(torch.zeros(k, 1, 1))

        self.vert_pool = nn.AvgPool2d(kernel_size=(kernel, 1), stride=1)
        self.hori_pool = nn.AvgPool2d(kernel_size=(1, kernel), stride=1)

        pad_size = kernel // 2
        self.pad_vert = nn.ReflectionPad2d((0, 0, pad_size, pad_size))
        self.pad_hori = nn.ReflectionPad2d((pad_size, pad_size, 0, 0))

        self.gamma = nn.Parameter(torch.zeros(k,1,1))
        self.beta = nn.Parameter(torch.ones(k,1,1))

    def forward(self, x):
        hori_l = self.hori_pool(self.pad_hori(x))
        hori_h = x - hori_l

        hori_out = self.hori_low * hori_l + (self.hori_high + 1.) * hori_h

        vert_l = self.vert_pool(self.pad_vert(hori_out))
        vert_h = hori_out - vert_l

        vert_out = self.vert_low * vert_l + (self.vert_high + 1.) * vert_h

        return x * self.beta + vert_out * self.gamma


# if __name__ == "__main__":
#     input_tensor = torch.randn(1, 32, 386, 386)  # batch_size=1, channels=64, height=32, width=32
#     dsam_block = SpecAtte_Block(in_channel=32)
#     output_tensor = dsam_block(input_tensor)
#     print(output_tensor.shape)