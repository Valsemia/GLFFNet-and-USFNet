import torch
import torch.nn as nn
import torch.nn.functional as F


class PConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(PConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

        # Initialize mask convolution weights to all 1s
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, x, mask):
        if mask.size(1) != x.size(1):
            mask = mask.expand(-1, x.size(1), -1, -1)
        # 调整 mask 的形状以匹配 x
        if mask.size() != x.size():
            mask = torch.nn.functional.interpolate(mask, size=x.size()[2:], mode='nearest')
        masked_x = x * mask
        output = self.conv(masked_x)
        mask_output = self.mask_conv(mask)
        output = output / (mask_output + 1e-8)
        return output


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(dilation)
        self.pconv1 = PConv(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(True)
        self.pad2 = nn.ReflectionPad2d(1)
        self.pconv2 = PConv(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x, mask):
        out = self.pad1(x)
        out = self.pconv1(out, mask)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.pconv2(out, mask)
        out = self.bn2(out)
        if out.size() != x.size():
            out = torch.nn.functional.interpolate(out, size=x.size()[2:], mode='nearest')
        out += x
        return out


# define the Encoder unit 构建 U-Net 网络中的编码器部分的单个块,从输入图像中提取特征，并逐渐减少特征图的空间大小
class UnetSkipConnectionEBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetSkipConnectionEBlock, self).__init__()

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)

        downrelu = nn.LeakyReLU(0.2, True)

        downnorm = norm_layer(inner_nc, affine=True)

        if outermost:
            down = [downconv]
        elif innermost:
            down = [downrelu, downconv]
        else:
            down = [downrelu, downconv, downnorm]
            if use_dropout:
                down.append(nn.Dropout(0.5))

        self.model = nn.Sequential(*down)

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, res_num=4, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Encoder, self).__init__()

        # construct unet structure
        self.Encoder_1 = UnetSkipConnectionEBlock(input_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, outermost=True)
        # self.PConv = PConv(in_channels=64, out_channels=128)
        self.Encoder_2 = UnetSkipConnectionEBlock(ngf, ngf * 2, norm_layer=norm_layer, use_dropout=use_dropout)
        self.Encoder_3 = UnetSkipConnectionEBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout)
        self.Encoder_4 = UnetSkipConnectionEBlock(ngf * 4, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout)
        # self.cbam2 = CBAM(in_channels=512, reduction=16, kernel_size=7)
        self.Encoder_5 = UnetSkipConnectionEBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout)
        self.Encoder_6 = UnetSkipConnectionEBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout, innermost=True)
        # self.cbam3 = CBAM(in_channels=512, reduction=16, kernel_size=7)

        # blocks = [ResnetBlock(ngf * 8, 2) for _ in range(res_num)]
        # self.middle = nn.Sequential(*blocks)

    def forward(self, x, mask):
        y_1 = self.Encoder_1(x)
        # y_1 = self.cbam1(y_1)
        # print(y_1.shape)
        y_2 = self.Encoder_2(y_1)
        # print(y_2.shape)
        y_3 = self.Encoder_3(y_2)
        # print(y_3.shape)
        y_4 = self.Encoder_4(y_3)
        # y_4 = self.cbam2(y_4)
        # print(y_4.shape)
        y_5 = self.Encoder_5(y_4)
        # print(y_5.shape)
        y_6 = self.Encoder_6(y_5)
        # y_6 = self.cbam3(y_6)
        # print(y_6.shape)
        y_7 = y_6
        # for block in self.middle:
        #     y_7 = block(y_7, mask)
            # print(y_7.shape)
            # print(f'mask.shape = {mask.shape}')

        return y_1, y_2, y_3, y_4, y_5, y_7


# if __name__ == '__main__':
#     net = Encoder(3, 3)
#     #print(net)
#     input_tensor = torch.randn(5, 3, 400, 400)
#     mask_tensor = torch.randn(5, 1, 400, 400)
#     output_tensor = net.forward(input_tensor, mask_tensor)
#     print(output_tensor)