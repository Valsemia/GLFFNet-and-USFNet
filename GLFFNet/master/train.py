import argparse
import datetime
import os

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import joint_transforms
from config import sbu_training_root
from dataset import ImageFolder
from misc import AvgMeter, check_mkdir, validate_model
from model import BDRAR

plt.switch_backend('Agg')
cudnn.benchmark = True

torch.cuda.set_device(0)

ckpt_path = './ckpt+ISTD'
exp_name = 'BDRAR+EMA+U2-net-inp'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--train_batch_size', type=int, default=5, help='size of the batches')
parser.add_argument('--sbu_training_root', type=str, default=r'Datasets/SBU/SBUTrain4KRecoveredSmall', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=5e-3, help='initial learning rate')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='optimization algorithm')
parser.add_argument('--iter_num', type=int, default=266, help='iteration')
parser.add_argument('--last_iter', type=int, default=0, help='last iteration')
parser.add_argument('--decay_epoch', type=int, default=100,
					help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=416, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--snapshot_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--snapshot', type=str, default='', help='number of epochs of training')
parser.add_argument('--resume', action='store_true', help='resume')
parser.add_argument('--iter_loss', type=int, default=500, help='average loss for n iterations')
args = parser.parse_args()
args_default = vars(args)

# batch size of 8 with resolution of 416*416 is exactly OK for the GTX 1080Ti GPU

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args.size, args.size))
])
val_joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((args.size, args.size))
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

train_set = ImageFolder(sbu_training_root, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args.train_batch_size, num_workers=8, shuffle=True)
# val_set = ImageFolder(sbu_validation_root, joint_transform, img_transform, target_transform)
# val_loader = DataLoader(val_set, batch_size=args.val_batch_size, num_workers=8, shuffle=False)


bce_logit = nn.BCEWithLogitsLoss().cuda()
#log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')
timestamp = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')
log_path = os.path.join(ckpt_path, exp_name, f'{timestamp}.txt')

Cuda = True
def main():
    net = BDRAR().train()
    net_train= net.train()
    net_train=net_train.cuda()

    loss = []
    val_loss = []

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args.lr},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args.lr, 'weight_decay': args.weight_decay}
    ], momentum=args.momentum)

    if len(args.snapshot) > 0:
        print('training resumes from \'%s\'' % args.snapshot)
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args.snapshot + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args.snapshot + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args.lr
        optimizer.param_groups[1]['lr'] = args.lr

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')

    for epoch in range(args.epoch, args.n_epochs):
        losses = fit_one_epoch(net_train, net, optimizer, epoch, args.iter_num, train_loader, args.n_epochs, Cuda, bce_logit,
                      args.train_batch_size)
        loss.append(losses)

        # val_losses = validate_model(net, bce_logit, val_loader)
        loss.append(losses)
        # val_loss.append(val_losses)

        plt.figure(figsize=(10, 5))  # 设置图片信息 例如：plt.figure(num = 2,figsize=(640,480))
        plt.plot(loss, 'b', label='loss')  # epoch_losses 传入模型训练中的 loss[]列表,在训练过程中，先创建loss列表，将每一个epoch的loss 加进这个列表
        # plt.plot(val_loss, 'r', label='val_loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')  # 个性化图例（颜色、形状等）
        plt.savefig(os.path.join(ckpt_path, exp_name, 'loss.png'))  # 保存图片 路径：/imgPath/
        plt.close('all')


def fit_one_epoch(net_train, net, optimizer, epoch, iter_num, train_loader, n_epochs, cuda, bce_logit, train_batch_size):
    total_loss=0
    #curr_iter = args.last_iter
    net_train.train()
    epoch_losses = []  # 用于记录每个 epoch 的总损失
    best_loss = float('inf')  # 初始化最佳损失为无穷大
    # while True:
    train_loss_record, loss_fuse_record, loss1_h2l_record = AvgMeter(), AvgMeter(), AvgMeter()
    loss2_h2l_record, loss3_h2l_record, loss4_h2l_record = AvgMeter(), AvgMeter(), AvgMeter()
    loss1_l2h_record, loss2_l2h_record, loss3_l2h_record = AvgMeter(), AvgMeter(), AvgMeter()
    loss4_l2h_record = AvgMeter()

    # for epoch in range(args.epoch, args.n_epochs):
    for i, data in enumerate(train_loader, start=1):
        optimizer.param_groups[0]['lr'] = 2 * args.lr * (1 - float(i) / args.iter_num
                                                            ) ** args.lr_decay
        optimizer.param_groups[1]['lr'] = args.lr * (1 - float(i) / args.iter_num
                                                        ) ** args.lr_decay

        inputs, labels = data
        # batch_size = inputs.size(0)
        with torch.no_grad():
            if cuda:
                inputs = Variable(inputs).cuda()
                labels = Variable(labels).cuda()

        optimizer.zero_grad()

        fuse_predict, predict1_h2l, predict2_h2l, predict3_h2l, predict4_h2l, \
        predict1_l2h, predict2_l2h, predict3_l2h, predict4_l2h = net(inputs)

        loss_fuse = bce_logit(fuse_predict, labels)
        loss1_h2l = bce_logit(predict1_h2l, labels)
        loss2_h2l = bce_logit(predict2_h2l, labels)
        loss3_h2l = bce_logit(predict3_h2l, labels)
        loss4_h2l = bce_logit(predict4_h2l, labels)
        loss1_l2h = bce_logit(predict1_l2h, labels)
        loss2_l2h = bce_logit(predict2_l2h, labels)
        loss3_l2h = bce_logit(predict3_l2h, labels)
        loss4_l2h = bce_logit(predict4_l2h, labels)

        loss = (loss_fuse + loss1_h2l + loss2_h2l + loss3_h2l + loss4_h2l + loss1_l2h + \
                loss2_l2h + loss3_l2h + loss4_l2h) / 9


        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        train_loss_record.update(loss.data, train_batch_size)
        loss_fuse_record.update(loss_fuse.data, train_batch_size)
        loss1_h2l_record.update(loss1_h2l.data, train_batch_size)
        loss2_h2l_record.update(loss2_h2l.data, train_batch_size)
        loss3_h2l_record.update(loss3_h2l.data, train_batch_size)
        loss4_h2l_record.update(loss4_h2l.data, train_batch_size)
        loss1_l2h_record.update(loss1_l2h.data, train_batch_size)
        loss2_l2h_record.update(loss2_l2h.data, train_batch_size)
        loss3_l2h_record.update(loss3_l2h.data, train_batch_size)
        loss4_l2h_record.update(loss4_l2h.data, train_batch_size)

        # if i % 1 ==0 or i == iter_num:
        # # curr_iter += 1
        #     print('Epoch [{:03d}/{:03d}], iter [{:4d}/{:04d}],'
        #           '[loss:{:.5f}], loss_fuse:{:.5f}],  loss1_h2l:{:.5f}], loss2_h2l:{:.5f}], loss3_h2l :{:.5f}], loss4_h2l:{:.5f}], loss1_l2h :{:.5f}], loss2_l2h:{:.5f}], loss3_l2h :{:.5f}], loss4_l2h:{:.5f}]'.
        #           format(epoch + 1, 100, i, iter_num, train_loss_record.show(), loss_fuse_record.show(),
        #                  loss1_h2l_record.show(), loss2_h2l_record.show(), loss3_h2l_record.show(), loss4_h2l_record.show(),
        #                  loss1_l2h_record.show(), loss2_l2h_record.show(), loss3_l2h_record.show(),
        #                  loss4_l2h_record.show()))
        #     torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % iter_num))

        # 每隔一定频率打印损失信息
        if i % 1 == 0:
            print('Epoch [{:03d}/{:03d}], iter [{:4d}/{:04d}],'
                      '[loss:{:.5f}], loss_fuse:{:.5f}],  loss1_h2l:{:.5f}], loss2_h2l:{:.5f}], loss3_h2l :{:.5f}], loss4_h2l:{:.5f}], loss1_l2h :{:.5f}], loss2_l2h:{:.5f}], loss3_l2h :{:.5f}], loss4_l2h:{:.5f}]'.
                      format(epoch + 1, 100, i, iter_num, train_loss_record.show(), loss_fuse_record.show(),
                             loss1_h2l_record.show(), loss2_h2l_record.show(), loss3_h2l_record.show(), loss4_h2l_record.show(),
                             loss1_l2h_record.show(), loss2_l2h_record.show(), loss3_l2h_record.show(),
                             loss4_l2h_record.show()))

        # 记录每个 epoch 的总损失
        epoch_losses.append(total_loss)

        with open(log_path, 'a') as f:
            f.write('[Epoch {}], [Total Loss {:.4f}]\n'.format(epoch + 1, total_loss))

        # 保存模型
        # 保存最后一个权重
        if (epoch + 1) % 1 == 0:
            torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, 'Last', f'{epoch + 1}_net_last.pth'))

        # 保存最好的权重
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, 'Best', 'best_net.pth'))

    print('Epoch:'+str(epoch+1)+'/'+str(n_epochs))
    print('Total Loss: %.4f' % (total_loss / iter_num))

    return total_loss / iter_num




if __name__ == '__main__':
    main()
