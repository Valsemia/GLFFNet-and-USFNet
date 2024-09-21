import torch
from RFR import RFRNet, RFRModule, Bottleneck

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TransferLearningModel(RFRNet):
    def __init__(self, pretrained_path):
        super(TransferLearningModel, self).__init__()
        self.pretrained_path = pretrained_path

        # 加载预训练权重
        self.load_pretrained_weights()

        self.RFRModule = RFRModule()
        self.bottleneck = Bottleneck(32, 8)

    def load_pretrained_weights(self):
        """
        加载预训练的权重到 netP
        """
        try:
            state_dict = torch.load(self.pretrained_path, map_location=device)
            # 使用 strict=False 来加载预训练模型，并忽略缺少的键和多余的键
            self.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded pretrained weights from {self.pretrained_path}")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")