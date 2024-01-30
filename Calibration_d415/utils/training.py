import os
import sys
import math
import string
import random
import shutil

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F

import os
import sys
sys.path.append(os.getcwd())
import utils.imgs as img_utils

RESULTS_PATH = '.results/'
WEIGHTS_PATH = '.weights/'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_weights(model, epoch, loss, err):      #保存每次训练的权重（训练的顺序、损失、误差）
    weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, err)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save({
            'startEpoch': epoch,
            'loss':loss,
            'error': err,
            'state_dict': model.state_dict()
        }, weights_fpath)                       #在文件中保存训练的序号、损失、误差和权重等字典
    shutil.copyfile(weights_fpath, WEIGHTS_PATH+'latest.th')    #保存最新的权重,shutil.copyfile(src,dst)

def load_weights(model, fpath):                     #加载权重
    print("loading weights '{}'".format(fpath))     #文件名称
    weights = torch.load(fpath)                     #加载文件中的内容
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch-1, weights['loss'], weights['error']))
    return startEpoch

def get_predictions(output_batch):
    bs,c,h,w = output_batch.size()              #batch_size(每批训练的数量)、n_channel(通道数量)、height(图像高度)、width(图像宽度)
    tensor = output_batch.detach()                  #NN输出的内容
    values, indices = tensor.cpu().max(1)       #输出中channel中最大的值和对应的标记
    indices = indices.view(bs,h,w)
    return indices

def error(preds, targets):
    assert preds.size() == targets.size()
    bs,h,w = preds.size()
    n_pixels = bs*h*w
    incorrect = preds.ne(targets).cpu().sum()   #预测和真实tensor中不相同元素的累计
    err = incorrect.item()/n_pixels
    return round(err,5)                         #保留5位小数

def train(model, trn_loader, optimizer, criterion, epoch):
    model.train()
    trn_loss = 0
    trn_error = 0
    for inputs,targets in trn_loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        output = model(inputs)

        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()
        
        pred = get_predictions(output)
        trn_error += error(pred, targets.detach().cpu())

    trn_loss /= len(trn_loader)
    trn_error /= len(trn_loader)
    return trn_loss, trn_error

def test(model, test_loader, criterion, epoch=1):
    model.eval()
    test_loss = 0
    test_error = 0
    for data, target in test_loader:
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        
        with torch.no_grad():
            output = model(data)
        test_loss += criterion(output, target).item()
        pred = get_predictions(output)
        test_error += error(pred, target.detach().cpu())
    test_loss /= len(test_loader)
    test_error /= len(test_loader)
    return test_loss, test_error

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()

def predict(model, input_loader, n_batches=1):
    input_loader.batch_size = 1
    predictions = []
    model.eval()
    for inputs, target in input_loader:
        data = inputs.to(DEVICE)
        label = target.to(DEVICE)

        with torch.no_grad():
            output = model(data)
        pred = get_predictions(output)
        predictions.append([inputs,target,pred])
    return predictions

def view_sample_predictions(model, loader, n):
    inputs, targets = next(iter(loader))
    data = inputs.to(DEVICE)
    label = targets.to(DEVICE)
    
    with torch.no_grad():
        output = model(data)
    pred = get_predictions(output)
    batch_size = inputs.size(0)
    for i in range(min(n, batch_size)):
        img_utils.view_image(inputs[i])
        img_utils.view_annotated(targets[i])
        img_utils.view_annotated(pred[i])

#################
######added######
#################
def get_iou(label,prediction):
    iou_list = []
    for i in {1,2,3,4,5,6,7,8,9}:
        # 找到指定类别的像素
        operate_label = np.copy(label.numpy())
        operate_prediction = np.copy(prediction.numpy())
        #判定是否有该类像素
        if i not in operate_label:
            continue
        else:
            operate_label[operate_label != i] = 0
            operate_label[operate_label == i] = 1

            operate_prediction[operate_prediction != i] = 0
            operate_prediction[operate_prediction == i] = 1
            
            iu = operate_label + operate_prediction    #元素2表示交;1,2表示并
            # print(iu)
            #得到交的个数和并的个数，计算比重
            num_1 = str(iu.tolist()).count("1")
            num_intersection = str(iu.tolist()).count("2")
            num_union = num_1 + num_intersection
            iou = round(num_intersection/num_union,2)   #计算iou,保留三位小数
            iou_list.append(iou)

    mean_iou = np.mean(iou_list)

    return mean_iou 
