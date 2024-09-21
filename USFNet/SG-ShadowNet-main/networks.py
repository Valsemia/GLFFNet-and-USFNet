import torch
from torch import nn

from encoder import Encoder
from decoder import Decoder
import torch.nn.functional as F
from other_utils import get_norm_layer


class MyModel(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm='batch', use_dropout=False, no_dropout='store_true', init_type='xavier',
                 init_gain=0.02, gpu_ids=[], pretrained_path=r'checkpoint_paris.pth'):
        super(MyModel, self).__init__()
        norm_layer = get_norm_layer(norm_type=norm)

        self.device = torch.device('cuda' if torch.cuda.is_available() and len(gpu_ids) > 0 else 'cpu')

        # 实例化 Encoder 和 Decoder，并传入额外的参数
        self.encoder = Encoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

        self.decoder = Decoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

        self.refine = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=False),
            nn.Conv2d(32, 3, 1, bias=False), nn.BatchNorm2d(3)
        )

    def forward(self, x, mask):
        # 编码输入张量
        encoded_tensors = self.encoder(x, mask)
        # print(f'mask.shape = {mask.shape}')
        # 选择前六个张量作为 Decoder 的输入
        decoder_input = encoded_tensors[:6]
        # 解码输入张量
        reconstructed_image = self.decoder(*decoder_input)

        reconstructed_image = self.refine(reconstructed_image)
        # print(f'reconstructed_image.shape = {reconstructed_image.shape}')

        reconstructed_image_resized = F.interpolate(reconstructed_image, size=(x.size(2), x.size(3)), mode='bilinear',
                                                    align_corners=True)

        final_output = reconstructed_image_resized + x

        # print(f'final_output.shape = {final_output.shape}')

        return final_output


# if __name__ == '__main__':
#     net = MyModel(3, 3, 64)
#     #print(net)
#     input_tensor = torch.randn(5, 3, 400, 400)
#     mask_tensor = torch.randn(5, 1, 400, 400)
#     output_tensor = net.forward(input_tensor, mask_tensor)
#     print(output_tensor)