import cv2
import torch
from torchvision import transforms
from tqdm import tqdm
import misc
from config import true_dir, pred_dir
from model import BDRAR
from pylab import *  # 导入savetxt模块


def batched_split(coord, bsize):
    n = coord.shape[1]
    num_batch = math.ceil(n / bsize)
    batch_list = []
    for i in range(num_batch):
        start = i * bsize
        end = min(n, (i + 1) * bsize)
        batch_list.append(coord[:, start: end, :])
    return batch_list

# numpy转化为tensor
def toTensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 修改颜色通道
    img = cv2.resize(img, (224, 224))  # 缩放图片尺寸
    img = torch.from_numpy(img.transpose((2, 0, 1)))  # 修改数据形状
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_path = './ckpt+ISTD'
exp_name = 'BDRAR+EMA+U2-net-inp'
args = {
    'snapshot': '840',
    'size': 416
}
net = BDRAR().cuda()

def save_evaluation_results(model_name, eval_type, metric1, metric2, metric3, metric4):
    """
    将评估结果保存到txt文件中。

    Args:
        model_name: 模型名称。
        eval_type: 评估类型。
        metric1: 第一个评估指标的值。
        metric2: 第二个评估指标的值。
        metric3: 第三个评估指标的值。
        metric4: 第四个评估指标的值。
    """

    # 创建保存评估结果的文件
    eval_results_file = os.path.join(r'ckpt+ISTD/eval_results', model_name + '.txt')

    # 打开文件并写入评估结果
    if os.path.exists(os.path.join(eval_results_file)):
        # 文件存在，打开文件追加内容
        with open(eval_results_file, 'a') as f:

            f.write('评估类型：{}\n'.format(eval_type))
            f.write('metric1: {:.4f}\n'.format(metric1))
            f.write('metric2: {:.4f}\n'.format(metric2))
            f.write('metric3: {:.4f}\n'.format(metric3))
            f.write('metric4: {:.4f}\n'.format(metric4))
    else:
        # 文件不存在，创建文件并写入内容
        with open(eval_results_file, 'w') as f:
            f.write('模型名称：{}\n'.format(model_name))
            f.write('评估类型：{}\n'.format(eval_type))
            f.write('metric1: {:.4f}\n'.format(metric1))
            f.write('metric2: {:.4f}\n'.format(metric2))
            f.write('metric3: {:.4f}\n'.format(metric3))
            f.write('metric4: {:.4f}\n'.format(metric4))

def eval_psnr(loader, pred_list, true_list, eval_type=None, eval_bsize=None, verbose=False):
    if eval_type == 'f1':
        metric_fn = misc.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = misc.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = misc.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = misc.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
    elif eval_type == 'rmse':
        metric_fn = misc.calc_rmse
        metric1, metric2, metric3, metric4 = 'rmse', 'none', 'none', 'none'

    val_metric1 = misc.Averager()
    val_metric2 = misc.Averager()
    val_metric3 = misc.Averager()
    val_metric4 = misc.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    print(pbar)

    pred = pred_list
    inp = true_list

    result1, result2, result3, result4 = metric_fn(pred, inp)#使用评估指标计算模型的性能
    val_metric1.add(result1.item(), inp.shape[0])#将评估指标的结果添加到累加器中
    val_metric2.add(result2.item(), inp.shape[0])
    val_metric3.add(result3.item(), inp.shape[0])
    val_metric4.add(result4.item(), inp.shape[0])

    if verbose:
        pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))#更新进度条的描述
        pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
        pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
        pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))

    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item()


if __name__ == '__main__':
    import os

    # 获取预测图像文件列表
    pred_image_files = os.listdir(pred_dir)

    # 获取真实图像文件列表
    true_image_files = os.listdir(true_dir)

    # 创建图片列表
    pred_list = []
    true_list = []

    # 将图片添加到列表
    for pred_image_file in pred_image_files:
        pred_list.append(os.path.join('pred', pred_image_file))

    for true_image_file in true_image_files:
        true_list.append(os.path.join('true', true_image_file))

    data = {true_image_files[i]: pred_image_files[i] for i in range(len(true_image_files))}

    # 加载模型
    model = BDRAR().cuda()

    pred = []
    true = []

    for pred_file in os.listdir(pred_dir):
        pred_path = os.path.join(pred_dir, pred_file)
        pred_img = cv2.imread(pred_path)
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
        pred_img = toTensor(pred_img)
        pred.append(pred_img)

    for true_file in os.listdir(true_dir):
        true_path = os.path.join(true_dir, true_file)
        true_img = cv2.imread(true_path)
        true_img = cv2.cvtColor(true_img, cv2.COLOR_BGR2RGB)
        true_img = toTensor(true_img)
        true.append(true_img)

    pred = torch.stack(pred)
    true = torch.stack(true)

    pred = torch.squeeze(pred)
    true = torch.squeeze(true)

    if pred.ndim != 4:
        raise ValueError("预期 pred 为 4 维张量，但得到 pred.ndim = {}。".format(pred.ndim))
    pred = pred.permute(0, 2, 3, 1)  # Reduce dimensions

    if true.ndim != 4:
        raise ValueError("预期 pred 为 4 维张量，但得到 pred.ndim = {}。".format(pred.ndim))
    true = true.permute(0, 2, 3, 1)  # Reduce dimensions

    # Check dimensions
    print("pred.ndim:", pred.ndim)
    print("true.ndim:", true.ndim)

    # 评估模型
    metric1, metric2, metric3, metric4 = eval_psnr(
         data, pred, true, eval_type='ber')

    # 打印评估结果
    print('metric1: {:.4f}'.format(metric1))
    print('metric2: {:.4f}'.format(metric2))
    print('metric3: {:.4f}'.format(metric3))
    print('metric4: {:.4f}'.format(metric4))

    save_evaluation_results(exp_name, 'ber', metric1, metric2, metric3, metric4)






