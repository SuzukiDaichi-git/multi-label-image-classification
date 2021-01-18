from __future__ import print_function, division

import torch
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

# experiment conditions
trial = 1
missing = 40
overlabeling = 40
img_size = 224
bsize= 128
lr = 0.1
step_size = 40
num_epoch = 40
result_dir = 'results_clean'
resume = None
clean_begin = 5
update_begin = 40
num_pred = 10
num_classes=20
# pos_rate = 0.4
# neg_rate = 0.985

exp_cond = "{0}_{1}_img{2}_batch{3}_lr{4}_step{5}_nepoch{6}_cbegin{7}_ubegin{8}_cosann_clean".format(missing, overlabeling, img_size, bsize, lr, step_size, num_epoch, clean_begin, update_begin)

randseed = trial
np.random.seed(randseed)
torch.manual_seed(randseed)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(1, 20)
        self.l2 = nn.Linear(20, 40)
        self.l3 = nn.Linear(40, 40)
        self.l4 = nn.Linear(40, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x


class VocDataset(Dataset):
    def __init__(self, label_file, image_dir, transform=None):
        self.labels = pd.read_csv(label_file,header=None)
        self.soft_labels = self.labels.iloc[1:,1:].values.astype('double')
        self.img_names = self.labels.iloc[1:,0].values
        self.labels = self.labels.iloc[1:,1:].values.astype('double')
        self.cbegin = clean_begin
        self.ubegin = update_begin
        self.image_dir = image_dir
        self.transform = transform
        self.prediction = np.zeros((len(self.labels), num_pred, num_classes), dtype=np.float32)
        self.thresh_pos = np.zeros(num_classes)
        self.thresh_neg = np.zeros(num_classes)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path)
        label = self.soft_labels[idx].astype('double')
        if not image.mode=='RGB':
            image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, idx
    
    def label_update(self, results, epoch):
        
        idx = epoch % num_pred
        self.prediction[:, idx] = results
        
        results = self.prediction.mean(axis=1)
        if epoch+1 < num_pred:
            results = self.prediction[:,:epoch+1, :].mean(axis=1)
        
        num_datas = self.labels.shape[0]
        hist_bins = 50
        hist_range = (0,1)
        
        probs_P = [[] for i in range(num_datas)]
        probs_N = [[] for i in range(num_datas)]

        for i in range(num_datas):
            for c in range(num_classes):
                if self.labels[i][c] == 1:
                    probs_P[c].append(results[i][c])
                else:
                    probs_N[c].append(results[i][c])

        for c in range(num_classes):
            x = np.linspace(0, 1, hist_bins)
            pos_label, _ = np.histogram(np.array(probs_P[c]), density=True, bins=hist_bins, range=hist_range)
            y = np.zeros(hist_bins)
            y[0] = pos_label[0]
            y[1] = pos_label[1]
            for i in range(1, hist_bins-1):
                y[i] = (pos_label[i-1] + pos_label[i] + pos_label[i+1]) / 3
            y[hist_bins-2] = pos_label[hist_bins-2]
            y[hist_bins-1] = pos_label[hist_bins-1]

            for i in range(2, hist_bins-2):
                diff1 = y[i-1] - y[i-2]
                diff2 = y[i] - y[i-1]
                diff3 = y[i+1] - y[i]
                diff4 = y[i+2] - y[i+1]
                if diff2>diff1 and diff3>diff1 and diff4>diff1 and (diff1<0 or diff2<0 or diff3<0 or diff4<0):
                    pred = i
                    break
            self.thresh_pos[c] = pred/hist_bins
            
            neg_label, _ = np.histogram(np.array(probs_N[c]), bins=hist_bins, range=hist_range)
            for i in range(2, hist_bins):
                if neg_label[i]>neg_label[i-1] and neg_label[i-1]<neg_label[i-2]:
                    pred = i
                    break
            self.thresh_neg[c] = pred/50

        if epoch+1 >= self.ubegin:
            for c in range(num_classes):
                idx_neg = np.where(self.labels[:,c]==0)
                update_idx_neg = np.where((self.labels[:,c]==0) & (results[:,c]>self.thresh_neg[c]))
                self.soft_labels[idx_neg, c] = 0
                self.soft_labels[update_idx_neg, c] = 1

        save_thresh_pos_path = os.path.join(result_dir, exp_cond+'_thresh_pos.csv')
        save_thresh_neg_path = os.path.join(result_dir, exp_cond+'_thresh_neg.csv')
        with open(save_thresh_pos_path, 'a') as f:
            w = csv.writer(f)
            w.writerow([epoch+1]+list(self.thresh_pos))
        with open(save_thresh_neg_path, 'a') as f:
            w = csv.writer(f)
            w.writerow([epoch+1]+list(self.thresh_neg))
            
            
            
        

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=(224,224), scale=(0.5, 1.0)),
        # transforms.Resize((img_size,img_size)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        #transforms.ToPILImage(),
        #transforms.RandomResizedCrop((224)),
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {'train': 0, 'val': 0}
image_datasets['train'] = VocDataset('./annotation/{0}_{1}_trainAnnotation.csv'.format(missing, overlabeling), './VOCdevkit/VOC2012/JPEGImages/', data_transforms['train'])
image_datasets['val'] = VocDataset('./annotation/valAnnotation.csv', './VOCdevkit/VOC2012/JPEGImages/', data_transforms['val'])

dataloaders = {'train': 0, 'val': 0}
dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=bsize,shuffle=True, num_workers=10)
dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=bsize,shuffle=False, num_workers=10)
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

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
            dataloaders['train'].dataset.label_update(epoch=start_epoch)
            exp_lr_scheduler.load_state_dict(checkpoint['scheduler'])
            scheduler.step()
            loss = checkpoint['loss']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']+1 if 'epoch' in checkpoint else 'unknown'))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            
            
    since = time.time()
    
#     results = np.random.rand(dataloaders['train'].dataset.soft_labels.shape[0], num_classes)
#     dataloaders['train'].dataset.label_update(results, 0)
    
    bce = nn.BCELoss(reduction='mean')
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_cmap = 0
    
    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        print('Current learning rate: ' + '%.5f'%cur_lr)
        results = np.zeros((len(dataloaders['train'].dataset), num_classes), dtype=np.float32)
        
        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            mean_probs = dataloaders['train'].dataset.prediction.mean(axis=1)
            if epoch+1 < num_pred:
                mean_probs = dataloaders['train'].dataset.prediction[:,:epoch+1, :].mean(axis=1)
            mean_probs = torch.from_numpy(mean_probs).cuda()
            
            # Iterate over data.
            for i, (inputs, labels, indexes) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs = inputs.float()
                labels = labels.float()
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if epoch+1 >= clean_begin:
                            thresh_pos = torch.tensor(dataloaders['train'].dataset.thresh_pos).cuda()
                            thresh_neg = torch.tensor(dataloaders['train'].dataset.thresh_neg).cuda()
                            ind_update = torch.where(((labels==1) & (mean_probs[indexes]>thresh_pos)) | ((labels==0) & (mean_probs[indexes]<=thresh_neg)))
                            if len(ind_update)==1:
                                ind_update = (torch.zeros(ind_update[0].shape[0], dtype=int), ind_update)
                            loss_update = bce(torch.sigmoid(outputs[ind_update]), labels[ind_update])
                        else:
                            loss_update = loss
                        loss_update.backward()
                        optimizer.step()
                        
                        probs = torch.sigmoid(outputs)
                        probs = probs.cpu().detach().numpy()
                        results[indexes] = probs
                        
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
                w.writerow(['epoch', 'train_loss', 'test_loss', 'cmap', 'emap', 'precision', 'recall', 'f1'])
            w.writerow([epoch+1, train_loss, test_loss, cmap, emap, p, r, f1])
        ckpt_path = os.path.join(result_dir, exp_cond+'_checkpoint.pth.tar')
        ckpt_path_epoch = os.path.join(result_dir, exp_cond+'_epoch{}_checkpoint.pth.tar'.format(epoch+1))
        torch.save({'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': exp_lr_scheduler.state_dict(),
                    'loss': train_loss,
                    'prediction': dataloaders['train'].dataset.prediction},
                   ckpt_path)
        if epoch+1 in [5,7,9,11,13,15,20,25,30,35]:
            torch.save({'epoch': epoch+1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': exp_lr_scheduler.state_dict(),
                        'loss': train_loss,
                        'prediction': dataloaders['train'].dataset.prediction},
                        ckpt_path_epoch)
        
        # label updateと, scheduler stepは, saveには入らない
        dataloaders['train'].dataset.label_update(results, epoch)
        scheduler.step()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test_model(model,optimizer,criterion, trial):
    since = time.time()
    model.eval()
    running_loss = 0.0
    # Iterate over data.
    for i, (inputs, labels, indexes) in enumerate(dataloaders['val']):
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
    
    epoch_loss = running_loss / dataset_sizes['val']

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return epoch_loss,cmap,emap,p,r,f


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#

# select model
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)

model_ft = model_ft.to(device)

criterion = nn.MultiLabelSoftMarginLoss()
optimizer_ft = optim.SGD([{'params':model_ft.conv1.parameters()},{'params':model_ft.bn1.parameters()},{'params':model_ft.layer1.parameters()},{'params':model_ft.layer2.parameters()},{'params':model_ft.layer3.parameters()},{'params':model_ft.layer4.parameters()},{'params': model_ft.fc.parameters(),'lr':lr}],lr=lr*0.1, momentum=0.9, weight_decay=1e-4)

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, gamma=0.1,step_size=step_size)
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=step_size, eta_min=1e-4)




#####################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=num_epoch, resume=resume)

# model_ft.load_state_dict(torch.load('./resnet101_model_test.pt'))
######################################################################
#
