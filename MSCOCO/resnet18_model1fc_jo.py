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
missing = 20
overlabeling = 20
img_size = 224
bsize=16
lr = 0.1
step_size = 25
num_epoch = 50
result_dir = 'results_resnet18'
resume = 'results_resnet18/20_20_img224_batch16_lr0.1_step20_nepoch40_epoch15_checkpoint.pth.tar'
update_begin = 17
num_pred = 10
alpha = 0
beta = 0
r_pos = 0
r_neg = 0
# exp_cond = "{0}_{1}_img{2}_batch{3}_lr{4}_step{5}_nepoch{6}".format(missing, overlabeling, img_size, bsize, lr, step_size, num_epoch)
# exp_cond = "{0}_{1}_img{2}_batch{3}_lr{4}_step{5}_nepoch{6}_alpha{7}_beta{8}_jo".format(missing, overlabeling, img_size, bsize, lr, step_size, num_epoch, alpha, beta)
exp_cond = "{0}_{1}_img{2}_batch{3}_lr{4}_step{5}_nepoch{6}_jo".format(missing, overlabeling, img_size, bsize, lr, step_size, num_epoch)

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
    
    def label_update(self, results, epoch):
        idx = (epoch - 1) % num_pred
        self.prediction[:, idx] = results

        if epoch+1 >= update_begin:
            self.soft_labels = self.prediction.mean(axis=1)
    
    def __getitem__(self,idx):
        img_name = os.path.join(self.image_dir,
                                self.labels.iloc[idx,0])
        image = io.imread(img_name)
#         label = self.labels.iloc[idx,1:].values
#         label = label.astype('double')
        label = self.soft_labels[idx]
        if len(image.shape)==2:
            image = np.expand_dims(image,2)
            image = np.concatenate((image,image,image),axis=2)
        if self.transform:
            image = self.transform(image)
        return image, label, idx

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
        if epoch+1 in [5,8,10,12,15,20,25,30]:
            torch.save({'epoch': epoch+1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': exp_lr_scheduler.state_dict(),
                        'loss': train_loss,
                        'prediction': dataloaders['train'].dataset.prediction,
                        'soft_labels': dataloaders['train'].dataset.soft_labels},
                        ckpt_path_epoch)

        print()
        
        dataloaders['train'].dataset.label_update(results, epoch)

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
        self.anno_dist = torch.from_numpy((annotation.sum(axis=0) / annotation.shape[0]).astype(np.float32)).cuda()
    
    # バッチ内の平均
    def __call__(self, preds, targets):
        bce = self.bce(preds, targets)
        probs = torch.sigmoid(preds)
        prob_entropy = -torch.mean(torch.mean((probs * torch.log(probs)), 1), 0)
        prob_avg = torch.mean(probs, dim=0)
        prob_dist = -torch.mean(self.anno_dist * torch.log(prob_avg))

        
        return bce + self.alpha*prob_entropy + self.beta*prob_dist
    
    
class my_criterion2():
    def __init__(self, alpha=0.0, beta=0.0, r_pos=0.0, r_neg=0.0):
        self.alpha = alpha
        self.beta = beta
        self.r_pos = r_pos
        self.r_neg = r_neg
        self.bce = nn.MultiLabelSoftMarginLoss()
        annotation = pd.read_csv(os.path.join('annotation','{0}_{1}_trainAnnotation.csv'.format(missing, overlabeling)), header=None).iloc[:,1:].values
        self.anno_dist = torch.from_numpy((annotation.sum(axis=0) / annotation.shape[0]).astype(np.float32)).cuda()
    
    # バッチ内の平均
    def __call__(self, preds, targets):
        bce = self.bce(preds, targets)
        probs = torch.sigmoid(preds)
        # 以下のようなやり方は上手くいかない
#         probs_pos = probs * targets
#         probs_neg = probs * (1-targets)
#         prob_entropy = torch.sum(probs_pos**r_pos * (1-probs_pos) + probs_neg**r_neg * (1-probs_neg)) / (preds.shape[0]*preds.shape[1])
        prob_entropy = torch.sum(self.r_pos * probs**self.r_pos * torch.log(probs) * targets + self.r_neg * probs**self.r_neg * torch.log(probs) * (1-targets)) / (preds.shape[0]*preds.shape[1])
        prob_avg = torch.mean(probs, dim=0)
        prob_dist = -torch.mean(self.anno_dist * torch.log(prob_avg))
        
        return bce + self.alpha*prob_entropy + self.beta*prob_dist


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#

num_classes=80

# select model
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)

model_ft = model_ft.to(device)

# criterion = nn.MultiLabelSoftMarginLoss()
criterion = my_criterion(alpha=alpha, beta=beta)
if r_pos != 0 and r_neg != 0:
    print("use my_criterion2")
    criterion = my_criterion2(alpha=alpha, beta=beta, r_pos=r_pos, r_neg=r_neg)
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD([{'params':model_ft.conv1.parameters()},{'params':model_ft.bn1.parameters()},{'params':model_ft.layer1.parameters()},{'params':model_ft.layer2.parameters()},{'params':model_ft.layer3.parameters()},{'params':model_ft.layer4.parameters()},{'params': model_ft.fc.parameters(),'lr':lr}],lr=lr*0.1, momentum=0.9, weight_decay=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, gamma=0.1,step_size=step_size)



#####################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=num_epoch, resume=resume)

# model_ft.load_state_dict(torch.load('./resnet101_model_test.pt'))
######################################################################
#

