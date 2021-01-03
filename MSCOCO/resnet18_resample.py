from __future__ import print_function, division

import torch
from torch.utils.data import Sampler
import torch.nn as nn
import torch.nn.functional as F
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
import random

# experiment conditions
trial = 1
num_classes=80
missing = 0
overlabeling = 0
img_size = 224
bsize= 80
lr = 0.1
step_size = 100
num_epoch = 100
result_dir = 'results_tmp'
resume = None # 'results_resnet18/0_0_img224_batch16_lr0.1_step20_nepoch40_epoch15_checkpoint.pth.tar'
num_pred = 10
crop_range = 0.5
nsample = int(8e4)
alpha=1.0
beta=1.0

# todo: 後で, cropを入れないverもやる.
exp_cond = "{0}_{1}_img{2}_batch{3}_lr{4}_step{5}_nepoch{6}_alpha{7}_beta{8}_crop{9}_nresample{10}".format(missing, overlabeling, img_size, bsize, lr, step_size, num_epoch, alpha, beta, crop_range, nsample)

randseed = trial
np.random.seed(randseed)
torch.manual_seed(randseed)

class CocoDataset(Dataset):
    def __init__(self, label_file, image_dir, transform=None):
        self.labels = pd.read_csv(label_file,header=None)
        self.soft_labels = self.labels.iloc[:,1:].values
        self.soft_labels = self.soft_labels.astype('double')
        self.image_dir = image_dir
        self.transform = transform
        self.prediction = np.zeros((len(self.labels), num_pred, 80), dtype=np.float32)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        # in val/test phase, image1!=image2
        img_name = os.path.join(self.image_dir,
                                self.labels.iloc[idx,0])
        image = io.imread(img_name)
        label = self.soft_labels[idx]
        if len(image.shape)==2:
            image = np.expand_dims(image,2)
            image = np.concatenate((image,image,image),axis=2)
        if self.transform:
            image = self.transform(image)
        return image, label, idx
    
    def get_index_dic(self):
        """ build a dict with class as key and img_ids as values
        :return: dict()
        """

        index_dic = [[] for i in range(num_classes)]

        for i in range(self.labels.shape[0]):
            label = self.labels.iloc[i,1:].values
            for idx in np.where(np.asarray(label) == 1)[0]:
                index_dic[idx].append(i)

        return index_dic

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop((img_size), scale=(crop_range, 1.0)),
        # transforms.Resize((img_size,img_size)),
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


class RandomCycleIter:

    def __init__(self, data_list, test_mode=False):
        self.data_list = list(data_list)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1

        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)

        return self.data_list[self.i]


def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):
    i = 0
    j = 0
    while i < n:

        #         yield next(data_iter_list[next(cls_iter)])

        if j >= num_samples_cls:
            j = 0

        if j == 0:
            cls = next(cls_iter)
            temp_tuple = next(zip(*[data_iter_list[cls]] * num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]

        i += 1
        j += 1


class ClassAwareSampler(Sampler):

    def __init__(self, data_source):
        random.seed(0)
        torch.manual_seed(0)

        self.epoch = 0

        self.class_iter = RandomCycleIter(range(num_classes))
        self.cls_data_list = data_source.get_index_dic()

        self.num_classes = len(self.cls_data_list)
        self.data_iter_list = [RandomCycleIter(x) for x in self.cls_data_list] # repeated
        self.num_samples = nsample
        self.num_samples_cls = bsize // 80
        self.data_source = data_source
        print(self.num_samples)
        print('>>> Class Aware Sampler Built! Class number: {}'.format(num_classes))

    def __iter__(self):
        
        return class_aware_sample_generator(self.class_iter, self.data_iter_list, self.num_samples, self.num_samples_cls)

    def __len__(self):
        return self.num_samples


image_datasets = {x: CocoDataset('./annotation/{0}_{1}_{2}Annotation.csv'.format(missing, overlabeling, x),
                                 os.path.join('//srv/datasets/MSCOCO/images', 'train2014'),
                                 data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {'train': 0, 'val': 0}
sampler_train = ClassAwareSampler(image_datasets['train'])
sampler_val = ClassAwareSampler(image_datasets['val'])

dataloaders['train'] = DataLoader(image_datasets['train'], batch_size=bsize, sampler=sampler_train, num_workers=10)
dataloaders['val'] = DataLoader(image_datasets['val'], batch_size=bsize, sampler=sampler_val, num_workers=10)
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

image_datasets_test = CocoDataset('./testAnnotation.csv', '//srv/datasets/MSCOCO/images/val2014',data_transforms['test'])
sampler_test = ClassAwareSampler(image_datasets_test)
dataloaders_test = DataLoader(image_datasets_test, batch_size=bsize, sampler=sampler_test, num_workers=10)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def train_model(model, criterion, optimizer, scheduler, num_epochs=40, resume=False):
    
    start_epoch = 0
    
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else print('no optimizer found')
            start_epoch = checkpoint['epoch']
            dataloaders['train'].dataset.prediction = checkpoint['prediction']
            # dataloaders['train'].dataset.soft_labels = dataloaders['train'].dataset.prediction.mean(axis=1)   # label update
            # exp_lr_scheduler.load_state_dict(checkpoint['scheduler'])
            loss = checkpoint['loss']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']+1 if 'epoch' in checkpoint else 'unknown'))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            
            
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_cmap = 0
    
    for epoch in range(start_epoch, num_epochs):
        scheduler.step()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        print('Current learning rate: ' + '%.5f'%cur_lr)
        results = np.zeros((len(dataloaders['train'].dataset), num_classes), dtype=np.float32)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for i, (inputs, labels, indexes) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()
        
                optimizer.zero_grad()
            
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        probs = torch.sigmoid(outputs)
                        probs = probs.cpu().detach().numpy()
                        results[indexes] = probs
                        loss.backward()
                        optimizer.step()
                    if phase == 'val':
                        if i == 0:
                            outputs_val = outputs
                            labels_val = labels
                        else:
                            outputs_val = torch.cat((outputs_val,outputs),0)
                            labels_val = torch.cat((labels_val,labels),0)
                running_loss += loss.item() * inputs.size(0)
                
            if phase=='train':
                train_loss = running_loss / dataset_sizes['train']
                print('{} Loss: {:.4f}'.format(phase, train_loss))
            else:
                val_loss = running_loss / dataset_sizes['val']
                val_cmap,val_emap = cemap_cal(outputs_val.to(torch.device("cpu")).numpy(),labels_val.to(torch.device("cpu")).numpy())
                val_p,val_r,val_f = prf_cal(outputs_val.to(torch.device("cpu")).numpy(),labels_val.to(torch.device("cpu")).numpy(),3)
                print('{} Loss: {:.4f}'.format(phase, val_loss))
            

            # deep copy the model
            if phase == 'val' and val_cmap > best_cmap:
                best_cmap = val_cmap
                print(val_cmap,val_emap)
                print(val_p,val_r,val_f)
                torch.save(model.state_dict(),os.path.join(result_dir, exp_cond+'_best_checkpoint.pth.tar'.format()))
                
                
        test_loss,cmap,emap,p,r,f1 = test_model(model,optimizer,criterion,trial)

        # save model
        print("experiment conditions = ", exp_cond)
        save_path = os.path.join(result_dir, exp_cond+'_indicator.csv')
        with open(save_path, 'a') as f:
            w = csv.writer(f)
            if epoch == 0:
                w.writerow(['epoch', 'train_loss', 'val_loss', 'val_cmap', 'val_emap', 'val_precision', 'val_recall', 'val_f1', 'test_loss', 'cmap', 'emap', 'precision', 'recall', 'f1'])
            w.writerow([epoch+1, train_loss, val_loss, val_cmap, val_emap, val_p, val_r, val_f, test_loss, cmap, emap, p, r, f1])
        ckpt_path = os.path.join(result_dir, exp_cond+'_checkpoint.pth.tar')
        ckpt_path_epoch = os.path.join(result_dir, exp_cond+'_epoch{}_checkpoint.pth.tar'.format(epoch+1))
        torch.save({'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': exp_lr_scheduler.state_dict(),
                    'loss': train_loss,
                    'prediction': dataloaders['train'].dataset.prediction},
                   ckpt_path)
        if epoch+1 in [8,10,12,15,20]:
            torch.save({'epoch': epoch+1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': exp_lr_scheduler.state_dict(),
                        'loss': train_loss,
                        'prediction': dataloaders['train'].dataset.prediction,
                        'soft_labels': dataloaders['train'].dataset.soft_labels},
                        ckpt_path_epoch)

        print()
        
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# def judge_and_update(preds, labels, epoch):
    
#     labels_update = labels.clone()
#     probs = torch.sigmoid(preds)
#     losses = -labels * torch.log(probs) - (1-labels) * torch.log(1-probs)
#     losses_argsort_idx = losses.argsort(dim=0)
#     preds_pos_mean = (preds * labels).sum(dim=0) / labels.sum(dim=0)
#     preds_neg_mean = (preds * (1-labels)).sum(dim=0) / (1-labels).sum(dim=0)
#     for cls in range(80):
#         for idx in losses_argsort_idx[-round(labels.size(0)*rate_schedule[epoch].item()):, cls]:
#             if labels[idx][cls] == 1 and preds[idx][cls]<preds_neg_mean[cls]:
#                 labels_update[idx][cls] = 0
#             elif labels[idx][cls] == 0 and preds[idx][cls]>preds_pos_mean[cls]:
#                 labels_update[idx][cls] = 1
                
#     return probs, labels_update


def test_model(model,optimizer,criterion, trial):
    since = time.time()
    model.eval()
    running_loss = 0.0
    # Iterate over data.
    for i, (inputs, labels, indexes) in enumerate(dataloaders_test):
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.float()
        labels = labels.float()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # backward + optimize only if in training phase
            if i == 0:
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



class my_criterion():
    def __init__(self, alpha=0.0, beta=0.0):
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.MultiLabelSoftMarginLoss()
        annotation = pd.read_csv(os.path.join('annotation','{0}_{1}_trainAnnotation.csv'.format(missing, overlabeling)), header=None).iloc[:,1:].values
        co_labels = np.zeros((80,80))
        for i in range(annotation.shape[0]):
            cls = np.where(annotation[i]==1)[0]
            for j in cls:
                for k in cls:
                    co_labels[j][k] += 1.
        for i in range(80):
            co_labels[i][i] /= 2.
        self.anno_dist = np.zeros(80)
        for i in range(80):
            for j in range(80):
                self.anno_dist[i] += co_labels[i][j] / co_labels[j][j]
        self.anno_dist = torch.from_numpy(self.anno_dist.astype(np.float32)).to(device)
    
    # バッチ内の平均
    def __call__(self, preds, targets):
        bce = self.bce(preds, targets)
        probs = torch.sigmoid(preds)
        prob_entropy = -torch.mean(torch.mean(probs * torch.log(probs), 1), 0)
        prob_avg = torch.mean(probs, dim=0)
        prob_dist = -torch.mean(self.anno_dist * torch.log(prob_avg))
        
        return bce + self.alpha*prob_entropy + self.beta*prob_dist




# select model
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)

model_ft = model_ft.to(device)

# criterion = nn.MultiLabelSoftMarginLoss()
criterion = my_criterion(alpha=alpha, beta=beta)

optimizer_ft = optim.SGD([{'params':model_ft.conv1.parameters()},{'params':model_ft.bn1.parameters()},{'params':model_ft.layer1.parameters()},{'params':model_ft.layer2.parameters()},{'params':model_ft.layer3.parameters()},{'params':model_ft.layer4.parameters()},{'params': model_ft.fc.parameters(),'lr':lr}],lr=lr*0.1, momentum=0.9, weight_decay=1e-4)

exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=step_size, eta_min=1e-4)



#####################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=num_epoch, resume=resume)

# model_ft.load_state_dict(torch.load('./resnet101_model_test.pt'))
######################################################################
#

