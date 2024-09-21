import torch
import torch.nn as nn
from config import u2net_path
from u2net import U2NET
import torch.nn.init as init

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class U2Net(nn.Module):
    def __init__(self, u2net_path=r'u2net.pth'):  # Assuming you are doing binary classification
        super(U2Net, self).__init__()

        # Load pre-trained U2NET
        u2net = U2NET()
        u2net.load_state_dict(torch.load(u2net_path))
        u2net = list(u2net.children())

        # Extract layers from U2NET
        self.encoder = nn.Sequential(*u2net[:7])

        # self.apply(weights_init)

    def forward(self, x):
        # Forward pass through the encoder
        x1 = self.encoder[0](x)
        # print('x1=', x1.shape)
        x2 = self.encoder[1](x1)
        # print('x2=', x2.shape)
        x3 = self.encoder[2](x2)
        # print('x3=', x3.shape)
        x4 = self.encoder[3](x3)
        # print('x4=', x4.shape)
        x5 = self.encoder[4](x4)
        # print('x5=', x5.shape)
        x6 = self.encoder[5](x5)
        # print('x6=', x6.shape)
        x7 = self.encoder[6](x6)
        # print('x7=', x7.shape)

        return x7


# if __name__ == '__main__':
#     model = U2Net()  # For binary classification
#     #print(net)
#     input_tensor = torch.randn(5, 3, 416, 416)
#     output_tensor = model.forward(input_tensor)
#
#     print(output_tensor.shape)


# class u2net(nn.Module):
#     def __init__(self, net=U2NET):
#         super(u2net, self).__init__()
#         net.load_state_dict(torch.load(u2net_path))
#
#         net = list(net.children())
#         self.layer0 = nn.Sequential(*net[:7])
#
#     def forward(self, x):
#         layer0 = self.layer0(x)
#         return layer0