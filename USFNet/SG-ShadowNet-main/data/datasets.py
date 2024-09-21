import glob
import random
import os

from torch.utils.data import Dataset
from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean
import random
import numpy as np
import torch


class ImageDataset(Dataset):
    """Data processing with mask and the dilated mask

    See `from shadow generation to shadow removal
    <https://arxiv.org/abs/2103.12997>`_ for details.

    """
    def __init__(self, root, unaligned=False, mode='train'):
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/train_A' % mode) + '/*.*'))
        # print(self.files_A)
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/train_C' % mode) + '/*.*'))
        # print(self.files_B)
        self.files_mask = sorted(glob.glob(os.path.join(root, '%s/train_B' % mode) + '/*.*'))
        # print(self.files_mask)
        # self.files_mask50 = sorted(glob.glob(os.path.join(root, '%s/train_mask50' % mode) + '/*.*'))

    def __getitem__(self, index):
        # 随机生成变量 i、j 和 k，用于图像的裁剪和翻转
        i = random.randint(0, 48)
        j = random.randint(0, 48)
        k=random.randint(0,100)

        item_A=color.rgb2lab(io.imread(self.files_A[index % len(self.files_A)])) # 将 RGB 图像转换为 LAB 色彩空间
        item_A=resize(item_A,(448,448,3))
        item_A=item_A[i:i+400,j:j+400,:] # 裁剪图像为 400x400 大小
        if k>50:
            item_A=np.fliplr(item_A) # 如果 k > 50，则对图像进行左右翻转
        item_A[:,:,0]=np.asarray(item_A[:,:,0])/50.0-1.0
        item_A[:,:,1:]=2.0*(np.asarray(item_A[:,:,1:])+128.0)/255.0-1.0 # 将图像的像素值标准化到 [-1, 1] 的范围内
        item_A=torch.from_numpy(item_A.copy()).float()
        item_A=item_A.view(400,400,3)
        item_A=item_A.transpose(0, 1).transpose(0, 2).contiguous()
        
        item_B=color.rgb2lab(io.imread(self.files_B[index % len(self.files_B)]))
        item_B=resize(item_B,(448,448,3))
        item_B=item_B[i:i+400,j:j+400,:]
        if k>50:
            item_B=np.fliplr(item_B)
        item_B[:,:,0]=np.asarray(item_B[:,:,0])/50.0-1.0
        item_B[:,:,1:]=2.0*(np.asarray(item_B[:,:,1:])+128.0)/255.0-1.0
        item_B=torch.from_numpy(item_B.copy()).float()
        item_B=item_B.view(400,400,3)
        item_B=item_B.transpose(0, 1).transpose(0, 2).contiguous()

        item_mask=io.imread(self.files_mask[index % len(self.files_mask)]) # 将遮罩图像的像素值大于 0 的部分设置为 1
        item_mask=resize(item_mask,(448,448,1))
        item_mask=item_mask[i:i+400,j:j+400,:]
        item_mask[item_mask>0] = 1.0
        if k>50:
            item_mask=np.fliplr(item_mask) # 如果 k > 50，则对图像进行左右翻转
        item_mask=np.asarray(item_mask)
        item_mask=torch.from_numpy(item_mask.copy()).float()
        item_mask=item_mask.view(400,400,1)
        item_mask=item_mask.transpose(0, 1).transpose(0, 2).contiguous()
        
        # item_mask50=io.imread(self.files_mask50[index % len(self.files_mask50)])
        # item_mask50=resize(item_mask50,(448,448,1))
        # item_mask50=item_mask50[i:i+400,j:j+400,:]
        # item_mask50[item_mask50>0] = 1.0
        # if k>50:
        #     item_mask50=np.fliplr(item_mask50)
        # item_mask50=np.asarray(item_mask50)
        # item_mask50=torch.from_numpy(item_mask50.copy()).float()
        # item_mask50=item_mask50.view(400,400,1)
        # item_mask50=item_mask50.transpose(0, 1).transpose(0, 2).contiguous()

        return item_A, item_B, item_mask

    def __len__(self):
        return max(len(self.files_B), len(self.files_mask))
        
class TestImageDataset(Dataset):
    def __init__(self, root, mode='test'):
        
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/test_A' % mode) + '/*.*'))                # full shadow image
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/test_C_fixed' % mode) + '/*.*'))
        self.files_mask = sorted(glob.glob(os.path.join(root, '%s/test_mask_6' % mode) + '/*.*'))
        
    def __getitem__(self, index):
        item_A_rgb = io.imread(self.files_A[index % len(self.files_A)])
        item_A = color.rgb2lab(item_A_rgb)
        item_A = resize(item_A,(480,640,3))
        item_A[:,:,0] = np.asarray(item_A[:,:,0]) / 50.0 - 1.0
        item_A[:,:,1:] = 2.0 * (np.asarray(item_A[:,:,1:]) + 128.0) / 255.0 - 1.0
        item_A=torch.from_numpy(item_A.copy()).float()
        item_A=item_A.view((480, 640, 3))
        item_A=item_A.transpose(0, 1).transpose(0, 2).contiguous()
        
        item_B_rgb = io.imread(self.files_B[index % len(self.files_B)])
        item_B = color.rgb2lab(item_B_rgb)
        item_B = resize(item_B,(480,640,3))
        item_B[:,:,0] = np.asarray(item_B[:,:,0]) / 50.0 - 1.0
        item_B[:,:,1:] = 2.0 * (np.asarray(item_B[:,:,1:]) + 128.0) / 255.0 - 1.0
        item_B=torch.from_numpy(item_B.copy()).float()
        item_B=item_B.view((480, 640, 3))
        item_B=item_B.transpose(0, 1).transpose(0, 2).contiguous()

        item_mask = io.imread(self.files_mask[index % len(self.files_mask)])
        item_mask = resize(item_mask,(480, 640, 1))
        item_mask[item_mask > 0] = 1.0
        item_mask = np.asarray(item_mask)
        item_mask = torch.from_numpy(item_mask.copy()).float()
        item_mask = item_mask.view(480, 640, 1)
        item_mask = item_mask.transpose(0, 1).transpose(0, 2).contiguous()
        
        return item_A, item_B, item_mask
        
    def __len__(self):
        return max(len(self.files_B),len(self.files_mask))
