import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from myresnet_fc import resnet101
import argparse
import os
import torch
import pickle
import numpy as np
from evaluation_metrics import ap_per_class, ap_pos_neg, ap_per_data
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torch.nn as nn
from skimage import io
import shutil
import csv
import pickle



class VocDataset(Dataset):
    def __init__(self, label_file, image_dir, transform=None):
        self.labels = pd.read_csv(label_file,header=0)
        self.img_names = self.labels.iloc[:,0].values
        self.soft_labels = self.labels.iloc[:,1:].values.astype('double')
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        img_name = os.path.join(self.image_dir, self.img_names[idx])
        image = io.imread(img_name)
        label = self.soft_labels[idx,:]
        if len(image.shape)==2:
            image = np.expand_dims(image,2)
            image = np.concatenate((image,image,image),axis=2)
        if self.transform:
            image = self.transform(image)
        return image, label, idx

    
img_size = 224
bsize = 16


data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size,img_size)),
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



voc_class = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']
    

def main(args):
    
    image_datasets = {'train': 0, 'val': 0}
    image_datasets['train'] = VocDataset('./annotation/{0}_{1}_trainAnnotation.csv'.format(args.missing, args.overlabeling), './VOCdevkit/VOC2012/JPEGImages/', data_transforms['train'])
    image_datasets['val'] = VocDataset('./annotation/valAnnotation.csv', './VOCdevkit/VOC2012/JPEGImages/', data_transforms['val'])

    dataloaders = {'train': 0, 'val': 0}
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=bsize,shuffle=True, num_workers=10)
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=bsize,shuffle=True, num_workers=10)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


    num_class = 20

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_class)

    model = model.cuda()
    
    # change!
    base_name = '40_40_img224_batch128_lr0.1_step40_nepoch40_cbegin5_ubegin40_cosann_clean_epoch25'
    
    checkpoint = torch.load(os.path.join(args.ckpt_dir, base_name+"_checkpoint.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    # model.load_state_dict(checkpoint, strict=False)
    prediction = checkpoint['prediction']
    epoch = checkpoint['epoch']-1
    
#     thresh_neg = pd.read_csv(os.path.join(args.ckpt_dir, "40_40_img224_batch128_lr0.1_step40_nepoch40_cbegin5_ubegin40_cosann_clean_thresh_neg.csv"), header=None).iloc[epoch,1:].values
#     thresh_pos = pd.read_csv(os.path.join(args.ckpt_dir, "40_40_img224_batch128_lr0.1_step40_nepoch40_cbegin5_ubegin40_cosann_clean_thresh_pos.csv"), header=None).iloc[epoch,1:].values
    thresh_neg = pd.read_csv(os.path.join(args.ckpt_dir, "40_40_img224_batch128_lr0.1_step40_nepoch40_cbegin5_ubegin40_cosann_clean_thresh_neg.csv"), header=None).iloc[epoch,1:].values
    thresh_pos = pd.read_csv(os.path.join(args.ckpt_dir, "40_40_img224_batch128_lr0.1_step40_nepoch40_cbegin5_ubegin40_cosann_clean_thresh_pos.csv"), header=None).iloc[epoch,1:].values
    
    hist_bins = 50
    hist_range=(0,1)
    
    # train/val
    for phase in ['train']:
        
        clean_label_path = os.path.join('annotation', "0_0_{}Annotation.csv".format(phase))
        noisy_label_path = os.path.join('annotation', "{0}_{1}_{2}Annotation.csv".format(args.missing, args.overlabeling, phase))
        clean_label = pd.read_csv(clean_label_path,header=0)
        noisy_label = pd.read_csv(noisy_label_path,header=0)
        clean_label = clean_label.iloc[:,1:].values
        noisy_label = noisy_label.iloc[:,1:].values
        
        print('{} dataset'.format(phase))
        if phase == 'train':
            model.train()
        else:
            model.eval()
        
        probs_TP = [[] for i in range(num_class)]
        probs_TN = [[] for i in range(num_class)]
        probs_FP = [[] for i in range(num_class)]
        probs_FN = [[] for i in range(num_class)]

        # compute AP and probs
        with torch.no_grad():

            for b, (inputs, labels, indexes) in enumerate(dataloaders[phase]):
                inputs = inputs.cuda()
                labels = labels.cuda()
                inputs = inputs.float()
                labels = labels.float()

                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                
                if b == 0:
                    outputs_test = outputs
                    indexes_test = indexes
                else:
                    outputs_test = torch.cat((outputs_test, outputs), 0)
                    indexes_test = torch.cat((indexes_test, indexes), 0)
                    
                prediction[indexes, epoch%10, :] = probs.detach().cpu().numpy()
                probs = prediction[indexes,:epoch+1,:].mean(axis=1)
                    
                for i, idx in enumerate(indexes):
                    for j in range(num_class):
                        if clean_label[idx][j] == noisy_label[idx][j]:
                            if noisy_label[idx][j] == 1:
                                probs_TP[j].append(probs[i][j].item())
                            else:
                                probs_TN[j].append(probs[i][j].item())
                        elif clean_label[idx][j] != noisy_label[idx][j]:
                            if noisy_label[idx][j] == 1:
                                probs_FP[j].append(probs[i][j].item())
                            else:
                                probs_FN[j].append(probs[i][j].item())
                           
                        
        AP = ap_per_class(outputs_test.to(torch.device("cpu")).numpy(), clean_label[indexes_test.to(torch.device("cpu")).numpy()])
        AP_pos, AP_neg = ap_pos_neg(outputs_test.to(torch.device("cpu")).numpy(), clean_label[indexes_test.to(torch.device("cpu")).numpy()], noisy_label[indexes_test.to(torch.device("cpu")).numpy()])
        print(AP.mean())
        
        with open(os.path.join(args.result_dir, base_name+'_{}_AP.binaryfile'.format(phase)), 'bw') as f:
            pickle.dump(AP, f)
        with open(os.path.join(args.result_dir, base_name+'_{}_AP_pos.binaryfile'.format(phase)), 'bw') as f:
            pickle.dump(AP_pos, f)
        with open(os.path.join(args.result_dir, base_name+'_{}_AP_neg.binaryfile'.format(phase)), 'bw') as f:
            pickle.dump(AP_neg, f)
                        
        fig = plt.figure(figsize=(16,num_class*6))
        ax = [0] * num_class * 2
        
        for i in range(num_class):
            ax[2*i] = fig.add_subplot(num_class,2,2*i+1)
            ax[2*i].hist([probs_TP[i],probs_FP[i]], label=['true positive','false positive'], range=hist_range, bins=hist_bins, rwidth=0.8)
            ax[2*i].set_title('probs of TP and FP(class: {0})(AP: {1})'.format(voc_class[i], round(AP_pos[i], 3)), fontsize=18)
            ax[2*i].tick_params(labelsize=18)
            y_min, y_max = ax[2*i].get_ylim()
            ax[2*i].set_ylim(0,y_max)
            ax[2*i].vlines(thresh_pos[i], 0, 10000, linestyle='dashed')
            ax[2*i].vlines(thresh_pos[i], 0, 10000, linestyle='dashed')
            ax[2*i].legend(fontsize=18)

            ax[2*i+1] = fig.add_subplot(num_class,2,2*i+2)
            ax[2*i+1].hist([probs_FN[i],probs_TN[i]], label=['false negative','true negative'], range=hist_range, bins=hist_bins, rwidth=0.8)
            ax[2*i+1].set_title('probs of FN and TN(class: {0})(AP: {1})'.format(voc_class[i], round(AP_neg[i], 3)), fontsize=18)
            ax[2*i+1].tick_params(labelsize=18)
            ax[2*i+1].vlines(thresh_neg[i], 0, 10000, linestyle='dashed')
            ax[2*i+1].vlines(thresh_neg[i], 0, 10000, linestyle='dashed')
            ax[2*i+1].set_ylim(0,len(probs_FN[i]))
            ax[2*i+1].legend(fontsize=18)
        
        
#         for i in range(num_class):
#             pos_label = np.histogram(np.array(probs_TP[i]+probs_FP[i]), bins=hist_bins, range=hist_range)
#             neg_label = np.histogram(np.array(probs_TN[i]+probs_FN[i]), bins=hist_bins, range=hist_range)            
            
#             ax[2*i] = fig.add_subplot(num_class,2,2*i+1)
#             ax[2*i].hist(probs_TP[i]+probs_FP[i], label=['positive'], range=hist_range, bins=hist_bins, rwidth=0.8)
#             ax[2*i].set_title('probs of P(class: {0})(AP: {1})'.format(voc_class[i], round(AP_pos[i], 3)), fontsize=18)
#             ax[2*i].tick_params(labelsize=18)
#             y_min, y_max = ax[2*i].get_ylim()
#             ax[2*i].set_ylim(0,y_max)
#             ax[2*i].vlines(thresh_pos[i], 0, 10000, linestyle='dashed')
#             ax[2*i].legend(fontsize=18)

#             ax[2*i+1] = fig.add_subplot(num_class,2,2*i+2)
#             ax[2*i+1].hist(probs_FN[i]+probs_TN[i], label=['negative'], range=hist_range, bins=hist_bins, rwidth=0.8)
#             ax[2*i+1].set_title('probs of N(class: {0})(AP: {1})'.format(voc_class[i], round(AP_neg[i], 3)), fontsize=18)
#             ax[2*i+1].tick_params(labelsize=18)
#             ax[2*i+1].vlines(thresh_neg[i], 0, 10000, linestyle='dashed')
#             ax[2*i+1].set_ylim(0,len(probs_FN[i]))
#             ax[2*i+1].legend(fontsize=18)

        fig.savefig(os.path.join(args.result_dir, base_name+'_{}_probs_concat.jpg'.format(phase)))

        
        
    # test 
#     print("test dataset")
    
#     model.eval()
#     clean_label_path = "./annotation/valAnnotation.csv"
#     clean_label = pd.read_csv(clean_label_path,header=0)
#     clean_label = clean_label.iloc[:,1:].values

#     # 各クラスについて算出する
#     probs_P = [[] for i in range(num_class)]
#     probs_N = [[] for i in range(num_class)]

#     with torch.no_grad():
#         for b, (inputs, labels, indexes) in enumerate(dataloaders['val']):

#             inputs = inputs.cuda()
#             labels = labels.cuda()
#             inputs = inputs.float()
#             labels = labels.float()

#             outputs = model(inputs)
#             probs = torch.sigmoid(outputs)

#             if b == 0:
#                 outputs_test = outputs
#                 labels_test = labels
#             else:
#                 outputs_test = torch.cat((outputs_test, outputs), 0)
#                 labels_test = torch.cat((labels_test, labels), 0)

#             for i, idx in enumerate(indexes):
#                 for j in range(num_class):
#                     if clean_label[idx][j] == 1:
#                         probs_P[j].append(probs[i][j].item())
#                     elif clean_label[idx][j] == 0:
#                         probs_N[j].append(probs[i][j].item())


#     AP = ap_per_class(outputs_test.to(torch.device("cpu")).numpy(), labels_test.to(torch.device("cpu")).numpy())
#     with open(os.path.join(args.result_dir, base_name+'_test_AP.binaryfile'), 'bw') as f:
#         pickle.dump(AP, f)

#     fig = plt.figure(figsize=(16,num_class*6))
#     ax = [0] * num_class * 2
#     for i in range(num_class):
#         ax[2*i] = fig.add_subplot(num_class,2,2*i+1)
#         ax[2*i].hist(probs_P[i], label='positive', range=hist_range, bins=hist_bins, rwidth=0.8)
#         ax[2*i].set_title('probs of P samples(class: {0})(AP: {1})'.format(voc_class[i], round(AP[i], 3)), fontsize=18)
#         ax[2*i].tick_params(labelsize=18)
#         ax[2*i].legend(fontsize=18)
#         ax[2*i] = fig.add_subplot(num_class,2,2*i+2)
#         ax[2*i].hist(probs_N[i], label='negative', range=hist_range, bins=hist_bins, rwidth=0.8)
#         ax[2*i].set_title('probs of N samples', fontsize=18)
#         ax[2*i].tick_params(labelsize=18)
#         ax[2*i].legend(fontsize=18)

#     fig.savefig(os.path.join(args.result_dir, base_name+'_test_probs.jpg'))
    
    
    # save row data
#     print('sample images')
#     train_sample_indexes = np.random.choice([i for i in range(dataset_sizes['train'])], size=100, replace=False)
#     test_sample_indexes = np.random.choice([i for i in range(dataset_sizes['val'])], size=50, replace=False)
#     train_sample_ann = image_datasets['train'].labels.iloc[train_sample_indexes, :].values
#     test_sample_ann = image_datasets['val'].labels.iloc[test_sample_indexes, :].values
#     os.makedirs(os.path.join(args.sample_dir, base_name, 'train_img'), exist_ok=True)
#     os.makedirs(os.path.join(args.sample_dir, base_name, 'test_img'), exist_ok=True)
#     with open(os.path.join(args.sample_dir, base_name, 'train_sample_ann.csv'), 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(['file_name']+[voc_class[i] for i in range(num_class)])
#         writer.writerows(train_sample_ann)
#     with open(os.path.join(args.sample_dir, base_name, 'test_sample_ann.csv'), 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(['file_name']+[voc_class[i] for i in range(num_class)])
#         writer.writerows(test_sample_ann)
#     for i, idx in enumerate(train_sample_indexes):
#         img_path = os.path.join(image_datasets['train'].image_dir, image_datasets['train'].img_names[idx])
#         new_img_path = os.path.join(args.sample_dir, base_name, 'train_img', image_datasets['train'].img_names[idx])
#         shutil.copy(img_path, new_img_path)
#         model.eval()
#         image, _, _ = image_datasets['train'][idx]
#         image = image.cuda().float().unsqueeze(0)
#         with torch.no_grad():
#             pred = model(image)
#             prob = torch.sigmoid(pred)
#             prob = prob.squeeze()
#         with open(os.path.join(args.sample_dir, base_name, 'train_sample_pred.csv'), 'a') as f:
#             writer = csv.writer(f)
#             if i == 0:
#                 writer.writerow(['file_name']+[voc_class[j] for j in range(num_class)])
#             writer.writerow([image_datasets['train'].img_names[idx]]+[round(prob[j].cpu().detach().item(), 3) for j in range(num_class)])
#     for i, idx in enumerate(test_sample_indexes):
#         img_path = os.path.join(image_datasets['val'].image_dir, image_datasets['val'].img_names[idx])
#         new_img_path = os.path.join(args.sample_dir, base_name, 'test_img', image_datasets['val'].img_names[idx])
#         shutil.copy(img_path, new_img_path)
#         model.eval()
#         image, _, _ = image_datasets['val'][idx]
#         image = image.cuda().float().unsqueeze(0)
#         with torch.no_grad():
#             pred = model(image)
#             prob = torch.sigmoid(pred)
#             prob = prob.squeeze()
#         with open(os.path.join(args.sample_dir, base_name, 'test_sample_pred.csv'), 'a') as f:
#             writer = csv.writer(f)
#             if i == 0:
#                 writer.writerow(['file_name']+[voc_class[j] for j in range(num_class)])
#             writer.writerow([image_datasets['val'].img_names[idx]]+[round(prob[j].cpu().detach().item(), 3) for j in range(num_class)])
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--missing", default=40, type=int)
    parser.add_argument("--overlabeling", default=40, type=int)
    parser.add_argument("--ckpt_dir", type=str, default="results_clean")
    parser.add_argument("--result_dir", type=str, default="visualize")
    parser.add_argument("--sample_dir", type=str, default="sample")

    args = parser.parse_args()
    
    torch.manual_seed(727)
    np.random.seed(456)
        
    main(args)