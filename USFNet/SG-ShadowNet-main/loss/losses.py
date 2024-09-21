import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import device


# 计算图像的空间一致性损失
class L_spa(torch.nn.Module):
    """spatial consistency loss

    See `Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement
    <https://arxiv.org/abs/2001.06826v2>`_ for details.

    """
    def __init__(self):
        super(L_spa, self).__init__()
        # 初始化卷积核（kernel）这些卷积核用于计算图像在不同方向上的差异
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = torch.nn.Parameter(data=kernel_left, requires_grad=False) # 将左侧方向的卷积核设置为模型的参数，但不进行梯度计算
        self.weight_right = torch.nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = torch.nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = torch.nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = torch.nn.AvgPool2d(4) # 定义了一个平均池化层，用于对输入的图像进行平均池化操作

    # 定义了前向传播方法，计算空间一致性损失
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)

        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)

        return E


# 感知损失
def preceptual_loss(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        loss_value += torch.mean(torch.abs(A_feat - B_feat))
    return loss_value


class SobelConv(nn.Module):
    def __init__(self, channels, cuda=True):
        super(SobelConv, self).__init__()
        x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        self.convx = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False, groups=channels)
        self.convy = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False, groups=channels)

        weights_x = torch.from_numpy(x_filter).float().unsqueeze(0).unsqueeze(0)
        weights_x = weights_x.repeat(channels, 1, 1, 1)
        weights_y = torch.from_numpy(y_filter).float().unsqueeze(0).unsqueeze(0)
        weights_y = weights_y.repeat(channels, 1, 1, 1)

        if cuda:
            weights_x = weights_x.cuda()
            weights_y = weights_y.cuda()

        self.convx.weight = nn.Parameter(weights_x)
        self.convy.weight = nn.Parameter(weights_y)

    def forward(self, x):
        g1_x = self.convx(x)
        g1_y = self.convy(x)
        # Compute gradient magnitude
        g_1 = torch.sqrt(torch.pow(g1_x, 2) + torch.pow(g1_y, 2) + 1e-8)  # Add a small constant to avoid division by zero
        return g_1

def edge_loss_with_mask(out, target, mask, cuda=True):
    sobel_conv = SobelConv(out.size(1), cuda=cuda)
    if cuda:
        sobel_conv = sobel_conv.cuda()

    # Apply mask to the output and target
    out_masked = out * mask
    target_masked = target * mask

    # Compute edge maps
    g1 = sobel_conv(out_masked)
    g2 = sobel_conv(target_masked)

    # Compute the loss
    loss = torch.mean((g1 - g2).pow(2))
    return loss


# Example usage:
# out: the generated image tensor of shape (batch_size, channels, height, width)
# target: the ground truth image tensor of shape (batch_size, channels, height, width)
# mask: the shadow mask tensor of shape (batch_size, 1, height, width)
# Assuming `out`, `target`, and `mask` are already defined tensors
