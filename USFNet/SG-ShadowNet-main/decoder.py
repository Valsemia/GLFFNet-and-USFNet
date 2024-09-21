import torch.nn as nn
import torch
import torch.nn.functional as F
from encoder import PConv
from DSAM import SpecAtte_Block


class UnetSkipConnectionDBlock(nn.Module):
    def __init__(self, inner_nc, outer_nc, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetSkipConnectionDBlock, self).__init__()
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)
        upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                    kernel_size=4, stride=2,
                                    padding=1)

        if outermost:
            up = [uprelu, upconv, nn.Tanh()]
        else:
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                up.append(nn.Dropout(0.5))

        self.model = nn.Sequential(*up)
        # self.pconv = PConv(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1)

    def forward(self, x, mask=None):
        # if mask is not None:
        #     x, mask = self.pconv(x, mask)
        return self.model(x)


class BSB(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(BSB, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)

        # 添加SpecAtte_Block
        self.SB = SpecAtte_Block(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(f'out.shape = {out.shape}')

        out = self.SB(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # print(f'out.shape = {out.shape}')

        out += residual
        out = self.relu(out)

        return out


class Decoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, pretrained_path=r'checkpoint_paris.pth'):
        super(Decoder, self).__init__()

        # construct unet structure
        self.Decoder_1 = UnetSkipConnectionDBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout,
                                                  innermost=True)
        self.Decoder_2 = UnetSkipConnectionDBlock(ngf * 16, ngf * 8, norm_layer=norm_layer, use_dropout=use_dropout)
        self.Decoder_3 = UnetSkipConnectionDBlock(ngf * 16, ngf * 4, norm_layer=norm_layer, use_dropout=use_dropout)

        self.Decoder_4 = UnetSkipConnectionDBlock(ngf * 8, ngf * 2, norm_layer=norm_layer, use_dropout=use_dropout)
        self.Decoder_5 = UnetSkipConnectionDBlock(ngf * 4, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
        self.Decoder_6 = UnetSkipConnectionDBlock(ngf * 2, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                                  outermost=True)
        self.bsb1 = BSB(512, 512)
        self.bsb2 = BSB(128, 128)

    def forward(self, input_1, input_2, input_3, input_4, input_5, input_6):
        y_1 = self.Decoder_1(input_6)
        # print(f'y_1[0].shape={y_1[0].shape}', f'input_6.shape={input_6.shape}')
        # print(f'y_1.shape={y_1.shape}')
        y_1 = self.bsb1(y_1)

        input5 = F.interpolate(input_5, size=y_1.shape[2:], mode='bilinear', align_corners=True)
        y_2 = self.Decoder_2(torch.cat([y_1, input5], 1))
        # print(f'y_2.shape={y_2.shape}', f'input5.shape={input5.shape}')

        input4 = F.interpolate(input_4, size=y_2.shape[2:], mode='bilinear', align_corners=True)
        # print(f'y_2.shape={y_2.shape}', f'input5.shape={input5.shape}')
        y_3 = self.Decoder_3(torch.cat([y_2, input4], 1))
        # print(f'y_3.shape={y_3.shape}', f'input4.shape={input4.shape}')

        input3 = F.interpolate(input_3, size=y_3.shape[2:], mode='bilinear', align_corners=True)
        y_4 = self.Decoder_4(torch.cat([y_3, input3], 1))
        # print(f'y_4.shape={y_4.shape}', f'input3.shape={input3.shape}')
        y_4 = self.bsb2(y_4)

        input2 = F.interpolate(input_2, size=y_4.shape[2:], mode='bilinear', align_corners=True)
        y_5 = self.Decoder_5(torch.cat([y_4, input2], 1))
        # print(f'y_5.shape={y_5.shape}', f'input2.shape={input2.shape}')

        input1 = F.interpolate(input_1, size=y_5.shape[2:], mode='bilinear', align_corners=True)
        y_6 = self.Decoder_6(torch.cat([y_5, input1], 1))
        # print(f'y_6.shape={y_6.shape}', f'input1.shape={input1.shape}')

        out = F.interpolate(y_6, size=(400, 400), mode='bilinear', align_corners=True)

        return out

# if __name__ == '__main__':
#     net = Decoder(3, 3)
#     #print(net)
#     input_tensor_1 = torch.randn(5, 64, 200, 200)  # 第一个输入张量
#     input_tensor_2 = torch.randn(5, 128, 100, 100)  # 第二个输入张量
#     input_tensor_3 = torch.randn(5, 256, 50, 50)  # 第三个输入张量
#     input_tensor_4 = torch.randn(5, 512, 25, 25)  # 第四个输入张量
#     input_tensor_5 = torch.randn(5, 512, 12, 12)  # 第五个输入张量
#     input_tensor_6 = torch.randn(5, 512, 6, 6)  # 第六个输入张量
#     output_tensor = net.forward(input_tensor_1, input_tensor_2, input_tensor_3, input_tensor_4, input_tensor_5,
#                                 input_tensor_6)
#     print(output_tensor)

    # net2 = BSB(32, 3)
    # x = torch.randn(5, 32, 384, 384)
    # mask = torch.randn(5, 32, 384, 384)
    # output = net2.forward(x, mask)
    # print(output)