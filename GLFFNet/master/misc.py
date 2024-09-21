import copy

import cv2
import numpy as np
import os

from torch import device
from torch.utils.data import Dataset

import model

import pydensecrf.densecrf as dcrf
import torch
from sklearn.metrics import recall_score, precision_score, precision_recall_curve, roc_auc_score

import metric

models = {}

class AvgMeter(object):
    def __init__(self, num=40):
        self.reset()
        self.num=num
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses=[]

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)
    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num,0):]))

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def crf_refine(img, annos):
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')


def make(model_spec, args=None, load_sd=False):#make() 函数的主要功能是根据 model_spec 来构造模型。它可以接受 args 和 load_sd 两个参数来控制模型的构造过程。
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)
    if load_sd:
        model.load_state_dict(model_spec['sd'])
    return model


def get_binary_classification_metrics(pred, gt, threshold=None):
    if threshold is not None:
        gt = (gt > threshold)
        pred = (pred > threshold)
    TP = np.logical_and(gt, pred).sum()
    TN = np.logical_and(np.logical_not(gt), np.logical_not(pred)).sum()
    FN = np.logical_and(gt, np.logical_not(pred)).sum()
    FP = np.logical_and(np.logical_not(gt), pred).sum()
    BER = cal_ber(TN, TP, FN, FP)
    ACC = cal_acc(TN, TP, FN, FP)
    return TP, TN, FP, FN, BER, ACC


def cal_ber(tn, tp, fn, fp):#计算BER
    return 0.5*(fp/(tn+fp) + fn/(fn+tp))


def cal_acc(tn, tp, fn, fp):#计算ACC
    return (tp + tn) / (tp + tn + fp + fn)


def calc_cod(y_pred, y_true):
    batchsize = y_true.shape[0]

    metric_FM = metric.Fmeasure()#F1分数指标：F1 分数是精度和召回率的加权平均值。它是一个衡量模型预测性能的常用指标。
    metric_WFM = metric.WeightedFmeasure()#加权F1分数指标：加权 F1 分数是根据每个类的样本数量来计算的 F1 分数。它可以避免模型对数量较多的类的偏差。
    metric_SM = metric.Smeasure()#S-measure 指标：S-measure 是一个综合考虑了精度、召回率和 F1 分数的指标。它更适合用于评估多类分类模型。
    metric_EM = metric.Emeasure()#E-measure 指标：E-measure 是一个综合考虑了精度、召回率和 F1 分数的指标。它更适合用于评估不平衡数据集上的模型。
    metric_MAE = metric.MAE()#MAE指标：MAE 是平均绝对误差。它是一个衡量模型预测误差的常用指标。
    with torch.no_grad():
        assert y_pred.shape == y_true.shape

        #此循环用于遍历所有样本，并将预测值和真实值转换为 NumPy 数组
        for i in range(batchsize):
            true, pred = \
                y_true[i, 0].cpu().data.numpy() * 255, y_pred[i, 0].cpu().data.numpy() * 255

            #用于更新评估指标。每个指标都有自己的 step() 函数，用于处理预测值和真实值
            metric_FM.step(pred=pred, gt=true)
            metric_WFM.step(pred=pred, gt=true)
            metric_SM.step(pred=pred, gt=true)
            metric_EM.step(pred=pred, gt=true)
            metric_MAE.step(pred=pred, gt=true)

        fm = metric_FM.get_results()["fm"]
        wfm = metric_WFM.get_results()["wfm"]
        sm = metric_SM.get_results()["sm"]
        em = metric_EM.get_results()["em"]["curve"].mean()
        mae = metric_MAE.get_results()["mae"]

    return sm, em, wfm, mae


def calc_rmse(y_pred, y_true):
    batchsize = y_true.shape[0]
    with torch.no_grad():
        assert y_pred.shape == y_true.shape
        rmse = 0
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        for i in range(batchsize):
            true = y_true[i].flatten()
            pred = y_pred[i].flatten()

            rmse += np.sqrt(np.mean((true - pred)**2))

    return rmse / batchsize, np.array(0), np.array(0), np.array(0)



def calc_f1(y_pred,y_true):
    batchsize = y_true.shape[0]
    with torch.no_grad():
        assert y_pred.shape == y_true.shape
        f1, auc = 0, 0
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        for i in range(batchsize):
            true = y_true[i].flatten()
            true = true.astype(np.int)
            pred = y_pred[i].flatten()

            precision, recall, thresholds = precision_recall_curve(true, pred)#precision_recall_curve() 函数用于计算精确率-召回率曲线。精确率-召回率曲线是衡量模型预测性能的一个常用指标

            # auc
            auc += roc_auc_score(true, pred)#roc_auc_score() 函数用于计算受试者操作特征曲线（ROC 曲线）下的面积（AUC）。AUC 是衡量模型预测性能的另一个常用指标
            # auc += roc_auc_score(np.array(true>0).astype(np.int), pred)
            f1 += max([(2 * p * r) / (p + r+1e-10) for p, r in zip(precision, recall)])

    return f1/batchsize, auc/batchsize, np.array(0), np.array(0)


def calc_fmeasure(y_pred,y_true):
    batchsize = y_true.shape[0]

    mae, preds, gts = [], [], []
    with torch.no_grad():
        for i in range(batchsize):
            gt_float, pred_float = \
                y_true[i, 0].cpu().data.numpy(), y_pred[i, 0].cpu().data.numpy()

            # # MAE
            mae.append(np.sum(cv2.absdiff(gt_float.astype(float), pred_float.astype(float))) / (
                        pred_float.shape[1] * pred_float.shape[0]))
            # mae.append(np.mean(np.abs(pred_float - gt_float)))
            #
            pred = np.uint8(pred_float * 255)
            gt = np.uint8(gt_float * 255)

            pred_float_ = np.where(pred > min(1.5 * np.mean(pred), 255), np.ones_like(pred_float),
                                   np.zeros_like(pred_float))
            gt_float_ = np.where(gt > min(1.5 * np.mean(gt), 255), np.ones_like(pred_float),
                                 np.zeros_like(pred_float))

            preds.extend(pred_float_.ravel())
            gts.extend(gt_float_.ravel())

        RECALL = recall_score(gts, preds)
        PERC = precision_score(gts, preds)

        fmeasure = (1 + 0.3) * PERC * RECALL / (0.3 * PERC + RECALL)#F1分数
        MAE = np.mean(mae)

    return fmeasure, MAE, np.array(0), np.array(0)


def calc_ber(y_pred, y_true):
    batchsize = y_true.shape[0]
    y_pred, y_true = y_pred.permute(0, 2, 3, 1).squeeze(-1), y_true.permute(0, 2, 3, 1).squeeze(-1)
    with torch.no_grad():
        assert y_pred.shape == y_true.shape
        pos_err, neg_err, ber = 0, 0, 0
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        for i in range(batchsize):
            true = y_true[i].flatten()
            pred = y_pred[i].flatten()

            TP, TN, FP, FN, BER, ACC = get_binary_classification_metrics(pred * 255,
                                                                         true * 255, 125)
            pos_err += (1 - TP / (TP + FN)) * 100#正例错误率（Positive Error Rate）
            neg_err += (1 - TN / (TN + FP)) * 100#负例错误率（Negative Error Rate）

    return pos_err / batchsize, neg_err / batchsize, (pos_err + neg_err) / 2 / batchsize, np.array(0)


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v

def validate_model(net, criterion, val_loader, cuda=True):
    net.eval()  # Set the model to evaluation mode
    total_val_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(val_loader, start=1):
            inputs, labels = data
            if cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            fuse_predict, predict1_h2l, predict2_h2l, predict3_h2l, predict4_h2l, \
            predict1_l2h, predict2_l2h, predict3_l2h, predict4_l2h = net(inputs)

            # Compute validation loss
            loss_fuse = criterion(fuse_predict, labels)
            loss1_h2l = criterion(predict1_h2l, labels)
            loss2_h2l = criterion(predict2_h2l, labels)
            loss3_h2l = criterion(predict3_h2l, labels)
            loss4_h2l = criterion(predict4_h2l, labels)
            loss1_l2h = criterion(predict1_l2h, labels)
            loss2_l2h = criterion(predict2_l2h, labels)
            loss3_l2h = criterion(predict3_l2h, labels)
            loss4_l2h = criterion(predict4_l2h, labels)

            # Calculate average validation loss
            avg_loss = (loss_fuse + loss1_h2l + loss2_h2l + loss3_h2l + loss4_h2l + loss1_l2h +
                        loss2_l2h + loss3_l2h + loss4_l2h) / 9

            total_val_loss += avg_loss.item()

    return total_val_loss / len(val_loader)  # Return average validation loss

