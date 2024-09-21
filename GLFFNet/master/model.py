import math

import torch
import torch.nn.functional as F
from torch import nn
from u2net_regular import U2Net


class _EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(_EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)#用于将输入张量中的每个元素转换为一个概率。-1 表示对输入张量沿着最后一维进行 softmax 操作。
        self.agp = nn.AdaptiveAvgPool2d((1, 1))#将对输入张量进行全局平均池化，并将输出张量的大小设置为 (1, 1)。
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))#输入张量进行平均池化，并将输出张量的宽度设置为 1，高度为输入张量的高度。
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))#对输入张量进行平均池化，并将输出张量的高度设置为 1，宽度为输入张量的宽度。
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)#语句将对输入张量进行分组规范化，每组的通道数为 channels // self.groups。
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)#将对输入张量进行 2D 卷积操作，每组的通道数为 channels // self.groups，卷积核的大小为 1，卷积步长为 1，卷积填充为 0。
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)#对输入张量进行2D卷积操作，每组的通道数为channels // self.groups，卷积核的大小为3，卷积步长为1，卷积填充为1。

    def forward(self, x):
        b, c, h, w = x.size()#获取输入张量的形状
        #print('x =', x.shape)
        group_x = x.reshape(b * self.groups, -1, h, w)#b*g, c//g, h, w将输入张量转换为分组格式，以便对其进行分组卷积操作。
        #print('group_x =', group_x.shape)
        x_h = self.pool_h(group_x)#用于从输入张量中提取一条水平方向的通道。例如，在图像分类任务中，x_h 可以用于从图像中提取一条水平方向的特征，以便输入到全连接层进行分类。
        #print('x_h =', x_h.shape)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)#对输入张量 group_x 进行纵向池化操作，并将输出张量的维度进行重新排列，以便与 x_h 的形状相同。
        #print('x_w =', x_w.shape)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))#将水平和垂直方向的特征进行融合，并对融合后的特征进行 1x1 卷积操作。
        #print('hw =', hw.shape)
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        #print('x_h =', x_h.shape)
        #print('x_w =', x_w.shape)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        #print('x1 =', x1.shape)
        x2 = self.conv3x3(group_x)
        #print('x2 =', x2.shape)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        #print('x11 =', x11.shape)
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)#b*g, c//g, hw
        #print('x12 =', x12.shape)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        #print('x21 =', x21.shape)
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)#b*g, c//g, hw
        #print('x22 =', x22.shape)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        #print('weights =', weights.shape)
        return_ema = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        new_channels = return_ema.shape[1] // 2
        return_ema = torch.narrow(return_ema, 1, 0, new_channels)
        #print('return =', return_ema.shape)
        return return_ema



class _AttentionModule(nn.Module):
    def __init__(self):
        super(_AttentionModule, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, dilation=2, padding=2, groups=32, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, dilation=3, padding=3, groups=32, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, dilation=4, padding=4, groups=32, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.down = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32)
        )

    def forward(self, x):
        block1 = F.relu(self.block1(x) + x, True)
        block2 = F.relu(self.block2(block1) + block1, True)
        block3 = torch.sigmoid(self.block3(block2) + self.down(block2))
        return block3


def FocalLoss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss implementation.

    Parameters:
        logits (torch.Tensor): Raw logits from the model.
        targets (torch.Tensor): Ground truth labels.
        alpha (float): Balance parameter for class weights.
        gamma (float): Exponent of the modulating factor (1 - p_t) ^ gamma.

    Returns:
        torch.Tensor: Focal Loss.
    """
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn_downsample = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
            residual = self.bn_downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class MyModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MyModule, self).__init__()
        self.block1 = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            nn.Conv2d(out_channels, out_channels*2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.block1(x)
        return out


class BDRAR(nn.Module):
    def __init__(self):
        super(BDRAR, self).__init__()
        #resnext = ResNeXt101()
        U2net = U2Net()
        self.layer1 = U2net.encoder[0]
        self.layer2 = U2net.encoder[1]
        self.layer3 = U2net.encoder[2]
        self.layer4 = U2net.encoder[3]
        self.layer5 = U2net.encoder[4]
        # self.layer5 = MyModule(128, 256)
        self.layer6 = U2net.encoder[5]
        self.layer6 = MyModule(256, 256)
        self.layer7 = U2net.encoder[6]
        self.layer7 = MyModule(512, 512)

        self.down4 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        #定义一个神经网络模块，其中由卷积层、批归一化层、激活函数层组成。作用是将输入的 2048 个通道的特征图转换为具有 32 个通道的特征图。
        self.down3 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(128, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.refine3_hl = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
         #逐点卷积块：逐点卷积是通过 1x1 的卷积核对每个像素点进行卷积，用于调整通道数。
         #深度可分离卷积块：将输入的 32 个通道分为 32 个组，每组只进行通道内的卷积。深度可分离卷积在减少计算复杂性的同时，保留了逐通道卷积的特性。
         #逐点卷积块
        self.refine2_hl = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.refine1_hl = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )

        self.attention3_hl = _EMA(channels=64)
        #self.attention2_hl = _EMA(channels=64)
        self.attention1_hl = _EMA(channels=64)
        # self.attention3_hl = _AttentionModule()
        self.attention2_hl = _AttentionModule()
        #self.attention1_hl = _AttentionModule()

        self.refine2_lh = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.refine4_lh = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.refine3_lh = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.attention2_lh = _EMA(channels=64)
        #self.attention3_lh = _EMA(channels=64)
        self.attention4_lh = _EMA(channels=64)
        #self.attention2_lh = _AttentionModule()
        self.attention3_lh = _AttentionModule()
        #self.attention4_lh = _AttentionModule()

        self.fuse_attention = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 2, 1)
        )#融合注意力机制

        self.predict = nn.Sequential(
            nn.Conv2d(32, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Dropout(0.1), nn.Conv2d(8, 1, 1)
        )#将张量转化为一个预测值

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = False
        #遍历所有模块，如果是 ReLU 或 Dropout，设置其 inplace 参数为 False。inplace 参数表示是否进行原地操作


    def forward(self, x):
        layer1 = self.layer1(x)
        # print('layer1=', layer1.shape)
        layer2 = self.layer2(layer1)
        # print('layer2=', layer2.shape)
        layer3 = self.layer3(layer2)
        # print('layer3=', layer3.shape)
        layer4 = self.layer4(layer3)
        # print('layer4=', layer4.shape)
        layer5 = self.layer5(layer4)
        # print('layer5=', layer5.shape)
        layer6 = self.layer6(layer5)
        # print('layer6=', layer6.shape)
        layer7 = self.layer7(layer6)
        # print('layer7=', layer7.shape)

        down4 = self.down4(layer7)
        # print('down4=', down4.shape)
        down3 = self.down3(layer6)
        # print('down3=', down3.shape)
        down2 = self.down2(layer5)
        # print('down2=', down2.shape)
        down1 = self.down1(layer4)
        # print('down1=', down1.shape)


        down4 = F.interpolate(down4, size=down3.size()[2:], mode='bilinear', align_corners=True)
        refine3_hl_0 = F.relu(self.refine3_hl(torch.cat((down4, down3), 1)) + down4, True)
        #特征融合：torch.cat((down4, down3), 1)，将两个特征图在通道维度上连接起来。
        #特征融合后的卷积和残差连接：self.refine3_hl()，进行卷积操作；torch.relu(... + down4, True)，将上一步得到的卷积结果与原始的 down4 特征图相加，并经过 ReLU 激活函数。这里的相加操作构成了残差连接
        # print('refine3_hl_0 =', refine3_hl_0.shape)
        # print('down4 =', down4.shape)
        # print('down3 =', down3.shape)
        refine3_hl_0 = (1 + self.attention3_hl(torch.cat((down4, down3), 1))) * refine3_hl_0
        #torch.cat((down4, down3), 1)，将两个特征图拼接起来；self.attention3_hl(...)，调用注意力机制；(... * refine3_hl_0)，将原始特征图与注意力调整后的特征图相乘，实现特征的加权融合。
        # print('refine3_hl_0=', refine3_hl_0.shape)
        refine3_hl_1 = F.relu(self.refine3_hl(torch.cat((refine3_hl_0, down3), 1)) + refine3_hl_0, True)
        refine3_hl_1 = (1 + self.attention3_hl(torch.cat((refine3_hl_0, down3), 1))) * refine3_hl_1
        # print('refine3_hl_1=', refine3_hl_1.shape)

        refine3_hl_1 = F.interpolate(refine3_hl_1, size=down2.size()[2:], mode='bilinear', align_corners=True)
        refine2_hl_0 = F.relu(self.refine2_hl(torch.cat((refine3_hl_1, down2), 1)) + refine3_hl_1, True)
        refine2_hl_0 = (1 + self.attention2_hl(torch.cat((refine3_hl_1, down2), 1))) * refine2_hl_0
        refine2_hl_1 = F.relu(self.refine2_hl(torch.cat((refine2_hl_0, down2), 1)) + refine2_hl_0, True)
        refine2_hl_1 = (1 + self.attention2_hl(torch.cat((refine2_hl_0, down2), 1))) * refine2_hl_1
        # print('refine2_hl_1=', refine2_hl_1.shape)

        refine2_hl_1 = F.interpolate(refine2_hl_1, size=down1.size()[2:], mode='bilinear', align_corners=True)
        refine1_hl_0 = F.relu(self.refine1_hl(torch.cat((refine2_hl_1, down1), 1)) + refine2_hl_1, True)
        refine1_hl_0 = (1 + self.attention1_hl(torch.cat((refine2_hl_1, down1), 1))) * refine1_hl_0
        refine1_hl_1 = F.relu(self.refine1_hl(torch.cat((refine1_hl_0, down1), 1)) + refine1_hl_0, True)
        refine1_hl_1 = (1 + self.attention1_hl(torch.cat((refine1_hl_0, down1), 1))) * refine1_hl_1
        # print('refine1_hl_1=', refine1_hl_1.shape)

        down2 = F.interpolate(down2, size=down1.size()[2:], mode='bilinear', align_corners=True)
        refine2_lh_0 = F.relu(self.refine2_lh(torch.cat((down1, down2), 1)) + down1, True)
        refine2_lh_0 = (1 + self.attention2_lh(torch.cat((down1, down2), 1))) * refine2_lh_0
        refine2_lh_1 = F.relu(self.refine2_lh(torch.cat((refine2_lh_0, down2), 1)) + refine2_lh_0, True)
        refine2_lh_1 = (1 + self.attention2_lh(torch.cat((refine2_lh_0, down2), 1))) * refine2_lh_1
        # print('refine2_lh_1=', refine2_lh_1.shape)

        down3 = F.interpolate(down3, size=down1.size()[2:], mode='bilinear', align_corners=True)
        refine3_lh_0 = F.relu(self.refine3_lh(torch.cat((refine2_lh_1, down3), 1)) + refine2_lh_1, True)
        refine3_lh_0 = (1 + self.attention3_lh(torch.cat((refine2_lh_1, down3), 1))) * refine3_lh_0
        refine3_lh_1 = F.relu(self.refine3_lh(torch.cat((refine3_lh_0, down3), 1)) + refine3_lh_0, True)
        refine3_lh_1 = (1 + self.attention3_lh(torch.cat((refine3_lh_0, down3), 1))) * refine3_lh_1
        # print('refine3_lh_1=', refine3_lh_1.shape)

        down4 = F.interpolate(down4, size=down1.size()[2:], mode='bilinear', align_corners=True)
        refine4_lh_0 = F.relu(self.refine4_lh(torch.cat((refine3_lh_1, down4), 1)) + refine3_lh_1, True)
        refine4_lh_0 = (1 + self.attention4_lh(torch.cat((refine3_lh_1, down4), 1))) * refine4_lh_0
        refine4_lh_1 = F.relu(self.refine4_lh(torch.cat((refine4_lh_0, down4), 1)) + refine4_lh_0, True)
        refine4_lh_1 = (1 + self.attention4_lh(torch.cat((refine4_lh_0, down4), 1))) * refine4_lh_1
        # print('refine4_lh_1=', refine4_lh_1.shape)

        refine3_hl_1 = F.interpolate(refine3_hl_1, size=down1.size()[2:], mode='bilinear', align_corners=True)
        # print('refine3_hl_1=', refine3_hl_1.shape)
        predict4_hl = self.predict(down4)
        # print('predict4_hl=', predict4_hl.shape)
        predict3_hl = self.predict(refine3_hl_1)
        # print('predict3_hl=', predict3_hl.shape)
        predict2_hl = self.predict(refine2_hl_1)
        # print('predict2_hl=', predict2_hl.shape)
        predict1_hl = self.predict(refine1_hl_1)
        # print('predict1_hl=', predict1_hl.shape)

        predict1_lh = self.predict(down1)
        # print('predict1_lh=', predict1_lh.shape)
        predict2_lh = self.predict(refine2_lh_1)
        # print('predict2_lh=', predict2_lh.shape)
        predict3_lh = self.predict(refine3_lh_1)
        # print('predict3_lh=', predict3_lh.shape)
        predict4_lh = self.predict(refine4_lh_1)
        # print('predict4_lh=', predict4_lh.shape)

        fuse_attention = torch.sigmoid(self.fuse_attention(torch.cat((refine1_hl_1, refine4_lh_1), 1)))
        # print('fuse_attention=', fuse_attention.shape)
        fuse_predict = torch.sum(fuse_attention * torch.cat((predict1_hl, predict4_lh), 1), 1, True)
        # print('fuse_predict=', fuse_predict.shape)

        predict4_hl = F.interpolate(predict4_hl, size=x.size()[2:], mode='bilinear',align_corners=True)#执行插值操作，将输入张量的尺寸从当前大小调整为与另一个张量具有相同的大小
        # print('predict4_hl=', predict4_hl.shape)
        predict3_hl = F.interpolate(predict3_hl, size=x.size()[2:], mode='bilinear',align_corners=True)
        # print('predict3_hl=', predict3_hl.shape)
        predict2_hl = F.interpolate(predict2_hl, size=x.size()[2:], mode='bilinear',align_corners=True)
        # print('predict2_hl=', predict2_hl.shape)
        predict1_hl = F.interpolate(predict1_hl, size=x.size()[2:], mode='bilinear',align_corners=True)
        # print('predict1_hl=', predict1_hl.shape)
        predict1_lh = F.interpolate(predict1_lh, size=x.size()[2:], mode='bilinear',align_corners=True)
        # print('predict1_lh=', predict1_lh.shape)
        predict2_lh = F.interpolate(predict2_lh, size=x.size()[2:], mode='bilinear',align_corners=True)
        # print('predict2_lh=', predict2_lh.shape)
        predict3_lh = F.interpolate(predict3_lh, size=x.size()[2:], mode='bilinear',align_corners=True)
        # print('predict3_lh=', predict3_lh.shape)
        predict4_lh = F.interpolate(predict4_lh, size=x.size()[2:], mode='bilinear',align_corners=True)
        # print('predict4_lh=', predict4_lh.shape)
        fuse_predict = F.interpolate(fuse_predict, size=x.size()[2:], mode='bilinear',align_corners=True)
        # print('fuse_predict=', fuse_predict.shape)

        if self.training:
            return fuse_predict, predict1_hl, predict2_hl, predict3_hl, predict4_hl, predict1_lh, predict2_lh, predict3_lh, predict4_lh
        return torch.sigmoid(fuse_predict)

# if __name__ == '__main__':
#     net = BDRAR()
#     #print(net)
#     input_tensor = torch.randn(5, 3, 416, 416)
#     output_tensor = net.forward(input_tensor)
    # print(output_tensor)