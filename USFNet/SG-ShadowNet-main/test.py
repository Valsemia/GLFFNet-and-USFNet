import argparse
import os

import torch
import numpy as np
from skimage import io, color
from skimage.transform import resize
from torch.utils.data import DataLoader

from utils.utils import labimage2tensor, tensor2img
from networks import MyModel
from data.datasets import TestImageDataset
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--net_dir', type=str, default=r'ckpt_pre_Pc_ReP_BSB/net_100.pth', help='checkpoint file')
parser.add_argument('--savepath', type=str, default='results/srd/', help='save path')
parser.add_argument('--dataset', type=str, default='srd', help='save path')
parser.add_argument('--input_nc', type=int, default=3, help='input channel')
parser.add_argument('--output_nc', type=int, default=3, help='output channel')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
opt = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if opt.dataset == 'istd':
    ## ISTD dataset
    opt.dataroot_A = r'Dataset/ISTD_Dataset/test/test_A'
    opt.im_suf_A = '.png'
    opt.dataroot_B = r'Dataset/ISTD_Dataset/test/(BDRAR) sbu_prediction_ISTD'
    opt.im_suf_B = '.png'
elif opt.dataset == 'srd':
    ## SRD dataset
    opt.dataroot_A = 'Dataset/srd/img'
    opt.im_suf_A = '.jpg'
    opt.dataroot_B = 'Dataset/srd/mask'
    opt.im_suf_B = '.jpg'
else:
    print("Please check the name of dataset...")
    exit(0)

if torch.cuda.is_available():
    opt.cuda = True
    device = torch.device('cuda:0')
print(opt)


def remove_module_prefix(state_dict):
    """
    Remove the 'module.' prefix from the keys in state_dict.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


# save dir
if not os.path.exists(opt.savepath):
    os.makedirs(opt.savepath)

###### Definition of variables ######
# Networks
net = MyModel(opt.input_nc, opt.output_nc, opt.ngf, opt.norm, opt.use_dropout)
# net = init_net(net, 'normal', 0.02, [0])  # Assuming you are using GPU 0

checkpoint = torch.load(opt.net_dir, map_location=device)
checkpoint = remove_module_prefix(checkpoint)
# 检查是否需要使用 DataParallel
if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net)

net.load_state_dict(checkpoint, strict=False)
net.to(device)
net.eval()

# test_dataloader = DataLoader(TestImageDataset(opt.dataroot), batch_size=1, shuffle=False, num_workers=0)
# 创建一个包含所有测试图像文件名（不包括文件扩展名）的列表 gt_list
gt_list = [os.path.splitext(f)[0] for f in os.listdir(opt.dataroot_A) if f.endswith(opt.im_suf_A)]

###### evaluation ######
for idx, img_name in enumerate(gt_list):
    # Set model input
    with torch.no_grad():
        labimage = color.rgb2lab(io.imread(os.path.join(opt.dataroot_A, img_name + opt.im_suf_A))) # 这一行加载测试图像，并将其转换为 LAB 色彩空间,
        h = labimage.shape[0] - labimage.shape[0] % 4  # 这两行用于计算 LAB 图像的高度和宽度，确保它们能够被 4 整除
        w = labimage.shape[1] - labimage.shape[1] % 4
        labimage = labimage2tensor(labimage, h, w).to(device)  # 这一行将 LAB 图像转换为 PyTorch 张量，并将其移到指定的计算设备（device）上

        mask = io.imread(os.path.join(opt.dataroot_B, img_name + opt.im_suf_B))  # 这一行加载掩码图像
        mask = resize(mask,(h,w))  # 这一行将掩码图像调整为与 LAB 图像相同的尺寸
        mask = torch.from_numpy(mask).float()  # 这一行将掩码图像转换为 PyTorch 张量，并将其类型转换为浮点数
        mask = mask.view(h,w,1)  # 这一行重新塑造掩码张量的形状，添加一个维度以便与 LAB 图像相乘
        mask = mask.transpose(0, 1).transpose(0, 2).contiguous()  # 这一行重新排列掩码张量的维度顺序，以便与 LAB 图像相乘
        mask = mask.unsqueeze(0).to(device)  # 这一行将掩码张量添加一个批处理维度，并将其移到指定的计算设备上
        mask = torch.where(mask > 0.5, torch.ones_like(mask), torch.zeros_like(mask))  # 这一行将掩码张量中大于 0.5 的像素值设置为 1，小于等于 0.5 的像素值设置为 0
        inv_mask = 1.0 - mask  # 这一行计算掩码的补码，即 1 减去掩码

        input = labimage  # 这一行将 LAB 图像与掩码的补码相乘，以获取掩码内的图像部分
        s_without_mask = input * inv_mask
        shadow_only = input * mask
        fake_temp = net(input, mask)  # 这一行将输入传递给神经网络模型 net 进行前向传播，生成输出图像
        # s_output = fake_temp * mask
        # n_output = input * inv_mask
        output = fake_temp


        outputimage = tensor2img(output, h, w)  # 这一行将模型生成的输出张量转换为图像格式

        save_path = os.path.join(opt.savepath, img_name + opt.im_suf_A)  # 这一行构建输出图像的保存路径
        io.imsave(save_path, outputimage)  # 这一行将输出图像保存到指定的路径

        print('Generated images %04d of %04d' % (idx + 1, len(gt_list)))  # 这一行打印出当前生成图像的进度，格式化输出当前索引和总图像数量