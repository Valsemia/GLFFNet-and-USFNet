import numpy as np
import os

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import sbu_testing_root
from misc import check_mkdir, crf_refine
from model import BDRAR

torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = 'BDRAR+EMA+U2-net-inp/Last'
args = {
    'snapshot': '840',
    'size': 416
}

img_transform = transforms.Compose([
    transforms.Resize(args['size']),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_test = {'sbu': sbu_testing_root}

to_pil = transforms.ToPILImage()


def main():
    net = BDRAR().cuda()

    if len(args['snapshot']) > 0:
        print('load snapshot \'%s\' for testing' % args['snapshot'])
        #net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        # net.load_state_dict(
        #     torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'), map_location='cpu'))
        net.load_state_dict(
            torch.load(os.path.join(ckpt_path, exp_name, '96_net_last' + '.pth'), map_location='cpu'))

    net.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            img_list = [img_name for img_name in os.listdir(os.path.join(root, 'test_A')) if
                        img_name.endswith('.png')]
            for idx, img_name in enumerate(img_list):
                print('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                # check_mkdir(
                #     os.path.join(ckpt_path, exp_name, '(%s) %s_prediction_%s' % (exp_name, name, args['snapshot'])))
                check_mkdir(
                    os.path.join(ckpt_path, exp_name, 'ISTD'))
                img = Image.open(os.path.join(root, 'test_A', img_name))
                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda()
                res = net(img_var)
                prediction = np.array(transforms.Resize((h, w))(to_pil(res.data.squeeze(0).cpu())))
                prediction = crf_refine(np.array(img.convert('RGB')), prediction)

                Image.fromarray(prediction).save(
                    # os.path.join(ckpt_path, exp_name, '(%s) %s_prediction_%s' % (
                    #     exp_name, name, args['snapshot']), img_name)
                    os.path.join(ckpt_path, exp_name, 'ISTD', img_name)
                )


if __name__ == '__main__':
    main()
