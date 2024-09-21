import torch.nn as nn
import torch.nn.functional as F
from utils.utils import weights_init_normal
import torch


class ResidualBlock(nn.Module):
    '''Residual block with residual connections
    ---Conv-IN-ReLU-Conv-IN-x-+-
     |________________________|
    '''
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class ConGenerator_S2F(nn.Module):
    '''Coarse Deshadow Network
    '''
    def __init__(self,init_weights=False):
        super(ConGenerator_S2F, self).__init__()
        # Initial convolution block
        self.conv1_0 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(4, 32, 7))
        self.conv1_1 = nn.Sequential(ResidualBlock(32))
        self.conv1_2 = nn.Sequential(ResidualBlock(32))
        self.pool1 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1, padding_mode='reflect'))
        self.conv2_1 = nn.Sequential(ResidualBlock(64))
        self.conv2_2 = nn.Sequential(ResidualBlock(64))
        self.pool2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1, padding_mode='reflect'))
        self.conv3_1 = nn.Sequential(ResidualBlock(128))
        self.conv3_2 = nn.Sequential(ResidualBlock(128))
        self.conv3_3 = nn.Sequential(ResidualBlock(128))
        self.conv3_4 = nn.Sequential(ResidualBlock(128))
        self.conv3_5 = nn.Sequential(ResidualBlock(128))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1))
        self.conv4_1 = nn.Sequential(ResidualBlock(64))
        self.conv4_2 = nn.Sequential(ResidualBlock(64))
        self.up5 = nn.Sequential(nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1))
        self.conv5_1 = nn.Sequential(ResidualBlock(32))
        self.conv5_2 = nn.Sequential(ResidualBlock(32))
        self.conv5_3 = nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(32, 3, 7))
        
        if init_weights:
            self.apply(weights_init_normal)
    
    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = ConGenerator_S2F(init_weights=True)
        return model


    def forward(self,xin,mask):
        x = self.conv1_0(torch.cat((xin,mask*2.0-1.0), dim=1))
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.conv3_5(x)
        x = self.up4(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.up5(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        xout = x + xin
        return xout.tanh()


# 计算非阴影区域的区域风格
class Condition(nn.Module):
    ''' Compute the region style of non-shadow regions'''

    def __init__(self, in_nc=3, nf=128):
        super(Condition, self).__init__()
        stride = 1
        pad = 0
        self.conv1 = nn.Conv2d(in_nc, nf//4, 1, stride, pad, bias=True) # 创建一个卷积层，输入通道数为 in_nc，输出通道数为 nf//4，卷积核大小为 1x1
        self.conv2 = nn.Conv2d(nf//4, nf//2, 1, stride, pad, bias=True)
        self.conv3 = nn.Conv2d(nf//2, nf, 1, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        out = self.act(self.conv3(out))

        mask = F.interpolate(mask.detach(), size=out.size()[2:], mode='nearest')  ## use the dilated mask to get the condition 使用插值操作将掩码 mask 调整到与输出特征图相同的大小
        zero = torch.zeros_like(mask)
        one = torch.ones_like(mask) # 创建了两个与 mask 相同大小的张量，分别填充为零和一
        mask = torch.where(mask >= 1.0, one, zero) # 将掩码中大于等于 1 的值设为 1，其余值设为 0，以确保掩码只包含 0 和 1
        cond = out*(1.0-mask) # 将特征图 out 中非阴影区域的特征提取出来，其余部分置零
        cond = torch.mean(cond, dim=[2, 3], keepdim=False) # 对提取出的非阴影区域的特征进行均值池化，得到一个全局的特征表示 cond
        
        return cond


# 用于计算前景-背景区域内的区域归一化
class RN(nn.Module):
    '''Compute the region normalization within the foreground-background region
    '''
    def __init__(self, dims_in, eps=1e-5):
        super(RN, self).__init__()
        self.eps = eps  #  是一个小的常数，用于防止除以零的情况

    def forward(self, x, mask):
        mean_back, std_back = self.get_foreground_mean_std(x * (1-mask), 1 - mask)  # the background features 调用 get_foreground_mean_std 方法计算背景区域的均值和标准差
        normalized = (x - mean_back) / std_back
        normalized_background = normalized * (1 - mask)  # 根据计算的背景均值和标准差对输入张量进行归一化

        mean_fore, std_fore = self.get_foreground_mean_std(x * mask, mask) # the background features 调用 get_foreground_mean_std 方法计算前景区域的均值和标准差
        normalized = (x - mean_fore) / std_fore 
        normalized_foreground = normalized * mask  # 根据计算的前景均值和标准差对输入张量进行归一化

        return normalized_foreground + normalized_background

    # 定义了一个辅助函数，用于计算前景或背景区域的均值和标准差
    def get_foreground_mean_std(self, region, mask):
        sum = torch.sum(region, dim=[2, 3])     # (B, C)沿着指定维度对输入张量进行求和，得到每个通道的总和
        num = torch.sum(mask, dim=[2, 3])       # (B, C)计算掩码中非零元素的数量，用于后续的均值计算
        mu = sum / (num + self.eps) # 计算均值
        mean = mu[:, :, None, None] # 将均值扩展为与输入张量相同的形状
        var = torch.sum((region + (1 - mask)*mean - mean) ** 2, dim=[2, 3]) / (num + self.eps) # 计算方差
        var = var[:, :, None, None]
        return mean, torch.sqrt(var+self.eps)


# 计算前景-背景区域内的空间区域反归一化
class SINLayer(nn.Module):
    '''Compute the spatial region denormalization within the foreground-background region
    '''
    def __init__(self, dims_in=256):
         super(SINLayer, self).__init__() 
         self.gamma_conv0 = nn.Conv2d(dims_in+1, dims_in//2, 1)
         self.gamma_conv1 = nn.Conv2d(dims_in//2, dims_in, 1) # 创建了两个卷积层，用于学习输入特征的缩放系数（gamma）
         self.gamma_conv2 = nn.Conv2d(dims_in, dims_in, 1) # 创建了另一个卷积层，用于学习输入特征的偏移量（beta）
         self.bate_conv0 = nn.Conv2d(dims_in+1, dims_in//2, 1)
         self.bate_conv1 = nn.Conv2d(dims_in//2, dims_in, 1)
         self.bate_conv2 = nn.Conv2d(dims_in, dims_in, 1) # 创建了用于学习输入特征的偏移量（beta）的三个卷积层，与 gamma 的计算类似

    def forward(self, x, cond_f, mask):
        m_cond_f = torch.cat((mask * cond_f, mask*2.0-1.0), dim=1) # 将掩码 mask 与区域风格 cond_f 相乘，并将结果与 (mask*2.0-1.0)（将掩码映射到 -1 到 1 之间）在通道维度上拼接起来，以便输入到后续的卷积层中
        gamma = self.gamma_conv2(self.gamma_conv1(F.leaky_relu(self.gamma_conv0(m_cond_f), 0.2, inplace=True))) # 通过一系列的卷积和激活函数操作，计算输入特征的缩放系数 gamma
        beta = self.bate_conv2(self.bate_conv1(F.leaky_relu(self.bate_conv0(m_cond_f), 0.2, inplace=True))) # 通过一系列的卷积和激活函数操作，计算输入特征的偏移量 beta

        return x * (gamma) + beta # 将输入特征 x 通过缩放系数 gamma 进行缩放，并加上偏移量 beta，从而实现空间区域反归一化操作


# 具有空间区域感知原型归一化的残差块（Residual Block）模块
class ResidualBlock_SIN(nn.Module):
    '''Residual block with spatially region-aware prototypical normalization
    ---Conv-SRPNorm-ReLU-Conv-SRPNorm-x-+-
     |__________________________________|
    '''

    def __init__(self, in_features=256, cond_dim=128):
        super(ResidualBlock_SIN, self).__init__()
        self.conv0 = nn.Sequential(nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, in_features, 3))
        self.local_scale0 = nn.Sequential(
            nn.Linear(cond_dim, in_features//16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 16, in_features, bias=False)
            )
        self.local_shift0 = nn.Sequential(
            nn.Linear(cond_dim, in_features//16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 16, in_features, bias=False)
            )

        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, in_features, 3))
        self.local_scale1 = nn.Sequential(
            nn.Linear(cond_dim, in_features//16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 16, in_features, bias=False)
            )
        self.local_shift1 = nn.Sequential(
            nn.Linear(cond_dim, in_features//16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 16, in_features, bias=False)
            )
        self.in_features = in_features
        self.act = nn.ReLU(inplace=True)
        self.RN0 = RN(in_features)
        self.RN1 = RN(in_features)
        self.SIN0 = SINLayer(in_features)
        self.SIN1 = SINLayer(in_features)

    def forward(self, x):
        identity = x[0]
        cond = x[1]
        mask = x[2]
        mask = F.interpolate(mask.detach(), size=identity.size()[2:], mode='nearest')

        local_scale_0 = self.local_scale0(cond)
        local_scale_1 = self.local_scale1(cond)
        local_shift_0 = self.local_shift0(cond)
        local_shift_1 = self.local_shift1(cond)
        # - Conv -SRPNorm - Relu
        out = self.conv0(identity)        
        out = self.RN0(out, mask) # with no extra params
        cond_f0 = out * (local_scale_0.view(-1, self.in_features, 1, 1)) + local_shift_0.view(-1, self.in_features, 1, 1)
        out = self.SIN0(out, cond_f0, mask)
        out = self.act(out)
        # - Conv -SRPNorm 
        out = self.conv1(out)
        out = self.RN1(out, mask)
        cond_f1 = out * (local_scale_1.view(-1, self.in_features, 1, 1)) + local_shift_1.view(-1, self.in_features, 1, 1)
        out = self.SIN1(out, cond_f1, mask)
        #  shortcut
        out += identity

        return out, cond


# 实现基于风格引导的去阴影网络
class ConRefineNet(nn.Module):
    ''' Style-guided Re-deshadow Network
    '''
    def __init__(self,init_weights=False):
        super(ConRefineNet, self).__init__()
        # Region style
        self.cond_net = Condition()
        # Initial convolution blocks
        self.conv1_b=nn.Sequential(nn.ReflectionPad2d(3),
                    nn.Conv2d(4, 32, 7),
                    nn.ReLU(inplace=True))
        self.downconv2_b=nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                    nn.ReLU(inplace=True))
        self.downconv3_b=nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                    nn.ReLU(inplace=True))
        self.conv4_b=nn.Sequential(ResidualBlock_SIN(128))
        self.conv5_b=nn.Sequential(ResidualBlock_SIN(128))
        self.conv6_b=nn.Sequential(ResidualBlock_SIN(128))
        self.conv7_b=nn.Sequential(ResidualBlock_SIN(128))
        self.conv8_b=nn.Sequential(ResidualBlock_SIN(128))
        self.conv9_b=nn.Sequential(ResidualBlock_SIN(128))
        self.conv10_b=nn.Sequential(ResidualBlock_SIN(128))
        self.conv11_b=nn.Sequential(ResidualBlock_SIN(128))
        self.conv12_b=nn.Sequential(ResidualBlock_SIN(128))
        self.upconv13_b=nn.Sequential(nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1),
                        nn.ReLU(inplace=True))
        self.upconv14_b=nn.Sequential(nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1),
                        nn.ReLU(inplace=True))
        self.conv15_b=nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(32, 3, 7))
        if init_weights:
            self.apply(weights_init_normal)
    
    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = ConRefineNet(init_weights=True)
        return model

    def forward(self,xin,mask):
        cond = self.cond_net(xin, mask)
        x=xin
        x=self.conv1_b(torch.cat((x, mask), dim=1))
        x=self.downconv2_b(x)
        x=self.downconv3_b(x)
        x, cond=self.conv4_b([x, cond, mask])
        x, cond=self.conv5_b([x, cond, mask])
        x, cond=self.conv6_b([x, cond, mask])
        x, cond=self.conv7_b([x, cond, mask])
        x, cond=self.conv8_b([x, cond, mask])
        x, cond=self.conv9_b([x, cond, mask])
        x, cond=self.conv10_b([x, cond, mask])
        x, cond=self.conv11_b([x, cond, mask])
        x, cond=self.conv12_b([x, cond, mask])
        x=self.upconv13_b(x)
        x=self.upconv14_b(x)
        x=self.conv15_b(x)
        xout=x+xin
        return xout.tanh()
