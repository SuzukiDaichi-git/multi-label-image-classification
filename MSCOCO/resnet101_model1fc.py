# -*- coding: utf-8 -*-
"""
=========================
Qian Wang, Ning Jia, Toby P. Breckon, A Baseline for Multi-Label Image Classification Using Ensemble Deep CNN, ICIP 2019
**Author**: Qian Wang
qian.wang173@hotmail.com
==========================

"""
# License: BSD

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import sys
import copy
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
from evaluation_metrics import prf_cal, cemap_cal
import scipy
import scipy.io
from myresnet_fc import resnet101
import pdb
import csv

# experiment conditions
trial = 1
missing = 20
overlabeling = 20
img_size = 224
bsize=16
lr = 0.1
step_size = 20
num_epoch = 40
result_dir = 'results2'

randseed = trial
np.random.seed(randseed)
torch.manual_seed(randseed)

class CocoDataset(Dataset):
    def __init__(self, label_file, image_dir, transform=None):
        self.labels = pd.read_csv(label_file,header=None)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        img_name = os.path.join(self.image_dir,
                                self.labels.iloc[idx,0])
        image = io.imread(img_name)
        label = self.labels.iloc[idx,1:].values
        label = label.astype('double')
        if len(image.shape)==2:
            image = np.expand_dims(image,2)
            image = np.concatenate((image,image,image),axis=2)
        if self.transform:
            image = self.transform(image)
        return image, label

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        #transforms.RandomResizedCrop((448)),
        transforms.Resize((img_size,img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# =============================================================================
# image_datasets = {x: VocDataset('~/Qian/labels/'+'classification_'+x+'.csv',
#                                  '~/Qian/VOCdevkit/VOC2007/JPEGImages/',
#                                  data_transforms[x])
#                   for x in ['train', 'val']}
# =============================================================================
image_datasets = {x: CocoDataset('./annotation/{0}_{1}_{2}Annotation.csv'.format(missing, overlabeling, x),
                                 os.path.join('//srv/datasets/MSCOCO/images', 'train2014'),
                                 data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {'train': 0, 'val': 0}

dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=bsize,shuffle=True, num_workers=10)
dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=bsize,shuffle=True, num_workers=10)
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

image_datasets_test = CocoDataset('./testAnnotation.csv', '//srv/datasets/MSCOCO/images/val2014',data_transforms['test'])
dataloaders_test = torch.utils.data.DataLoader(image_datasets_test,batch_size=bsize, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
######################################################################
# Mixup data for data augmentation
# ^^^^^^^^^^^^^^^^^^^^^^
# https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
def mixup_data(x,y,use_cuda=True):
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    mixed_x = 0.5*x + 0.5*x[index,:]
    mixed_y = (y + y[index,:])>0
    mixed_y = mixed_y.float()
    return torch.cat((x,mixed_x),0), torch.cat((y,mixed_y),0)

def mixup_data2(x,y,use_cuda=True):
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    mixed_x = 0.5*x + 0.5*x[index,:]
    mixed_y = y
    return torch.cat((x,mixed_x),0), torch.cat((y,y),0)

def mixup_data3(x,y,use_cuda=True):
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    mixed_x = 0.5*x + 0.5*x[index,:]
    mixed_y = (y + y[index,:])>0
    mixed_y = mixed_y.float()
    return mixed_x, mixed_y

def train_model(model, criterion, optimizer, scheduler, num_epochs=40):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_cmap = 0
    
    for epoch in range(num_epochs):
        scheduler.step()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        print('Current learning rate: ' + '%.5f'%cur_lr)
        if cur_lr < 0.00001:
            break      
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for index, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # mixed up
                inputs = inputs.float()
                labels = labels.float()
                #if epoch % 2 == 0:
                #    inputs,labels = mixup_data3(inputs,labels,use_cuda=True)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # loss = newHinge_rank_loss(outputs,labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    if phase == 'val':
                        if index == 0:
                            outputs_val = outputs
                            labels_val = labels
                        else:
                            outputs_val = torch.cat((outputs_val,outputs),0)
                            labels_val = torch.cat((labels_val,labels),0)
                running_loss += loss.item() * inputs.size(0)
            if phase=='train':
                train_loss = running_loss / dataset_sizes['train']
            else:
                val_loss = running_loss / dataset_sizes['val']
                val_cmap,val_emap = cemap_cal(outputs_val.to(torch.device("cpu")).numpy(),labels_val.to(torch.device("cpu")).numpy())
                val_p,val_r,val_f = prf_cal(outputs_val.to(torch.device("cpu")).numpy(),labels_val.to(torch.device("cpu")).numpy(),3)
                
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and cmap > best_cmap:
                best_cmap = cmap
                print(cmap,emap)
                print(p,r,f)
                best_model_wts = copy.deepcopy(model.state_dict())
                
                
        test_loss,cmap,emap,p,r,f1 = test_model(model,optimizer,criterion,trial)

        # save model
        exp_cond = "{0}_{1}_img{2}_batch{3}_lr{4}_step{5}_epoch{6}".format(missing, overlabeling, img_size, bsize, lr, step_size, num_epochs)
        print("experiment conditions = ", exp_cond)
        save_path = os.path.join(result_dir, exp_cond+'_indicator.csv')
        with open(save_path, 'a') as f:
            w = csv.writer(f)
            if epoch == 0:
                w.writerow(['epoch', 'train_loss', 'val_loss', 'val_cmap', 'val_emap', 'val_precision', 'val_recall', 'val_f1', 'test_loss', 'cmap', 'emap', 'precision', 'recall', 'f1'])
            w.writerow([epoch+1, train_loss, val_loss, val_cmap, val_emap, val_p, val_r, val_f, test_loss, cmap, emap, p, r, f1])
        ckpt_path = os.path.join(result_dir, exp_cond+'_checkpoint.pth.tar')

        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': exp_lr_scheduler.state_dict(),
                    'loss': epoch_loss},
                   ckpt_path)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_cmap


def test_model(model,optimizer,criterion, trial):
    since = time.time()
    model.eval()
    # Iterate over data.
    for index, (inputs, labels) in enumerate(dataloaders_test):
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.float()
        labels = labels.float()
        # zero the parameter gradients
        optimizer.zero_grad()
        running_loss = 0.0

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # backward + optimize only if in training phase
            if index == 0:
                outputs_test = outputs
                labels_test = labels
            else:
                outputs_test = torch.cat((outputs_test, outputs), 0)
                labels_test = torch.cat((labels_test, labels), 0)
            running_loss += loss.item() * inputs.size(0)

    cmap, emap = cemap_cal(outputs_test.to(torch.device("cpu")).numpy(), labels_test.to(torch.device("cpu")).numpy())
    print('Test:')
    print(cmap,emap)
    p, r, f = prf_cal(outputs_test.to(torch.device("cpu")).numpy(), labels_test.to(torch.device("cpu")).numpy(), 3)
    
    epoch_loss = running_loss / len(dataloaders_test)
            

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
        
    # scipy.io.savemat('./results/resnet101_model1fc_results_'+str(trial)+'imgSize'+str(img_size)+'.mat', mdict={'cmap': cmap, 'emap': emap, 'p': p,'r': r, 'f': f, 'scores': outputs_test.to(torch.device("cpu")).numpy()})
    # load best model weights
    # return cmap,emap,p,r,f,outputs_test.to(torch.device("cpu")).numpy()
    return epoch_loss,cmap,emap,p,r,f


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#

num_classes=80

# select model
model_ft = resnet101(pretrained=True,num_classes=num_classes)
if result_dir == 'results_vision':
    print("model from torchvision")
    model_ft = models.resnet101(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

model_ft = model_ft.to(device)

criterion = nn.MultiLabelSoftMarginLoss()
# Observe that all parameters are being optimized
if result_dir != 'results_vision':
    optimizer_ft = optim.SGD([{'params':model_ft.features.parameters()},{'params': model_ft.fc1.parameters(),'lr':lr}],lr=lr*0.1, momentum=0.9, weight_decay=1e-4)
else:
    optimizer_ft = optim.SGD([{'params':model_ft.conv1.parameters()},{'params':model_ft.bn1.parameters()},{'params':model_ft.layer1.parameters()},{'params':model_ft.layer2.parameters()},{'params':model_ft.layer3.parameters()},{'params':model_ft.layer4.parameters()},{'params': model_ft.fc.parameters(),'lr':lr}],lr=lr*0.1, momentum=0.9, weight_decay=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, gamma=0.1,step_size=step_size)

#####################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^


model_ft, best_cmap = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=num_epoch)

torch.save(model_ft.state_dict(),'./results/{0}_{1}_img{2}_batch{3}_lr{4}_step{5}_epoch{6}_best{7}_checkpoint.pth.tar'.format(missing, overlabeling, img_size, bsize, lr, step_size, num_epochs, best_cmap))
# model_ft.load_state_dict(torch.load('./resnet101_model_test.pt'))
######################################################################
#

