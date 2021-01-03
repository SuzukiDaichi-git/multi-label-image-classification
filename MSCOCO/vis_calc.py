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
from skimage import io
import shutil
import csv
import pickle


class CocoDataset(Dataset):
    def __init__(self, label_file, image_dir, transform=None):
        self.labels = pd.read_csv(label_file,header=None)
        self.soft_labels = self.labels.iloc[:,1:].values
        self.soft_labels = self.soft_labels.astype('double')
        self.image_dir = image_dir
        self.transform = transform
        self.prediction = np.zeros((len(self.labels), 10, 80), dtype=np.float32)

    def __len__(self):
        return len(self.labels)
    
    def label_update(self, results, epoch):
        idx = (epoch - 1) % 10
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

    
img_size = 224
bsize = 16


data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        #transforms.RandomResizedCrop((448)),
        transforms.Resize((img_size,img_size)),
        # transforms.RandomHorizontalFlip(),
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



with open('./mscoco_class.json') as f:
    mscoco_class = json.load(f)
    

def main(args):
    
    image_datasets = {x: CocoDataset('./annotation/{0}_{1}_{2}Annotation.csv'.format(args.missing, args.overlabeling, x),
                                     os.path.join('//srv/datasets/MSCOCO/images', 'train2014'),
                                     data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {'train': 0, 'val': 0}

    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=bsize,shuffle=True, num_workers=10)
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=bsize,shuffle=True, num_workers=10)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    image_datasets_test = CocoDataset('./testAnnotation.csv', '//srv/datasets/MSCOCO/images/val2014',data_transforms['test'])
    dataloaders_test = torch.utils.data.DataLoader(image_datasets_test,batch_size=bsize, shuffle=False, num_workers=10)

    num_class = 80

    model = resnet101(pretrained=True,num_classes=num_class)
    model = model.cuda()
    
    # change!
    base_name = '{0}_{1}_img224_batch16_lr0.1_step20_nepoch40_epoch10'.format(args.missing, args.overlabeling)
    
    checkpoint = torch.load(os.path.join(args.ckpt_dir, base_name+"_checkpoint.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    with open('instance_argsort.binaryfile', 'rb') as f:
         instance_argsort = pickle.load(f)
    
    
    # train/val
#     for phase in ['train']:
        
#         clean_label_path = os.path.join('annotation', "0_0_{}Annotation.csv".format(phase))
#         noisy_label_path = os.path.join('annotation', "{0}_{1}_{2}Annotation.csv".format(args.missing, args.overlabeling, phase))
#         clean_label = pd.read_csv(clean_label_path,header=None)
#         noisy_label = pd.read_csv(noisy_label_path,header=None)
#         clean_label = clean_label.iloc[:,1:].values
#         noisy_label = noisy_label.iloc[:,1:].values
        
#         print('{} dataset'.format(phase))
#         if phase == 'train':
#             model.train()
#         else:
#             model.eval()
        
#         probs_TP = [[] for i in range(num_class)]
#         probs_TN = [[] for i in range(num_class)]
#         probs_FP = [[] for i in range(num_class)]
#         probs_FN = [[] for i in range(num_class)]

#         # compute AP and probs
#         with torch.no_grad():

#             for b, (inputs, labels, indexes) in enumerate(dataloaders[phase]):
#                 inputs = inputs.cuda()
#                 labels = labels.cuda()
#                 inputs = inputs.float()
#                 labels = labels.float()

#                 outputs = model(inputs)
#                 probs = torch.sigmoid(outputs)
#                 if b == 0:
#                     outputs_test = outputs
#                     indexes_test = indexes
#                 else:
#                     outputs_test = torch.cat((outputs_test, outputs), 0)
#                     indexes_test = torch.cat((indexes_test, indexes), 0)
                    
#                 for i, idx in enumerate(indexes):
#                     for j in range(num_class):
#                         if clean_label[idx][j] == noisy_label[idx][j]:
#                             if noisy_label[idx][j] == 1:
#                                 probs_TP[j].append(probs[i][j].item())
#                             else:
#                                 probs_TN[j].append(probs[i][j].item())
#                         elif clean_label[idx][j] != noisy_label[idx][j]:
#                             if noisy_label[idx][j] == 1:
#                                 probs_FP[j].append(probs[i][j].item())
#                             else:
#                                 probs_FN[j].append(probs[i][j].item())
                           
                        
#         AP = ap_per_class(outputs_test.to(torch.device("cpu")).numpy(), clean_label[indexes_test.to(torch.device("cpu")).numpy()])
#         AP_pos, AP_neg = ap_pos_neg(outputs_test.to(torch.device("cpu")).numpy(), clean_label[indexes_test.to(torch.device("cpu")).numpy()], noisy_label[indexes_test.to(torch.device("cpu")).numpy()])
#         print(AP.mean())
        
#         with open(os.path.join('visualize', base_name+'_{}_AP.binaryfile'.format(phase)), 'bw') as f:
#             pickle.dump(AP, f)
#         with open(os.path.join('visualize', base_name+'_{}_AP_pos.binaryfile'.format(phase)), 'bw') as f:
#             pickle.dump(AP_pos, f)
#         with open(os.path.join('visualize', base_name+'_{}_AP_neg.binaryfile'.format(phase)), 'bw') as f:
#             pickle.dump(AP_neg, f)
            
            
#         fig = plt.figure(figsize=(24,480))
#         ax = [0] * num_class * 3
#         for i in range(num_class):
#             ax[3*i] = fig.add_subplot(num_class,3,3*i+1)
#             ax[3*i].hist([probs_TP[i],probs_FP[i]], label=['true positive','false positive'], range=(0,1), bins=50, rwidth=0.8)
#             ax[3*i].set_title('probs of TP and FP samples(class: {0})(AP: {1})({2})'.format(mscoco_class[str(i)], AP_pos[i], instance_argsort[i]))
#             ax[3*i].set_xlabel('prob')
#             ax[3*i].set_ylabel('pdf')
#             ax[3*i].legend()

#             ax[3*i+1] = fig.add_subplot(num_class,3,3*i+2)
#             ax[3*i+1].hist(probs_FN[i], label='false negative', range=(0,1), bins=50, rwidth=0.8)
#             ax[3*i+1].set_title('probs of FN samples(class: {0})(AP: {1})({2})'.format(mscoco_class[str(i)], AP_neg[i], instance_argsort[i]))
#             ax[3*i+1].set_xlabel('prob')
#             ax[3*i+1].set_ylabel('pdf')
#             ax[3*i+1].legend()

#             ax[3*i+2] = fig.add_subplot(num_class,3,3*i+3)
#             ax[3*i+2].hist(probs_TN[i], label='true negative', range=(0,1), bins=50, rwidth=0.8)
#             ax[3*i+2].set_title('probs of TN samples')
#             ax[3*i+2].set_xlabel('prob')
#             ax[3*i+2].set_ylabel('pdf')
#             ax[3*i+2].legend()

#         fig.savefig(os.path.join('visualize', base_name+'_{}_probs.jpg'.format(phase)))

        
        
#     # test 
#     print("test dataset")
    
#     model.eval()
#     clean_label_path = "./testAnnotation.csv"
#     clean_label = pd.read_csv(clean_label_path,header=None)
#     clean_label = clean_label.iloc[:,1:].values

#     # 各クラスについて算出する
#     probs_P = [[] for i in range(num_class)]
#     probs_N = [[] for i in range(num_class)]

#     with torch.no_grad():
#         for b, (inputs, labels, indexes) in enumerate(dataloaders_test):

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
#     with open(os.path.join('visualize', base_name+'_test_AP.binaryfile'), 'bw') as f:
#         pickle.dump(AP, f)

#     fig = plt.figure(figsize=(16,480))
#     ax = [0] * num_class * 2
#     for i in range(80):
#         ax[2*i] = fig.add_subplot(num_class,2,2*i+1)
#         ax[2*i].hist(probs_P[i], label='positive', range=(0,1), bins=50, rwidth=0.8)
#         ax[2*i].set_title('probs of P samples(class: {0})(AP: {1})({2})'.format(mscoco_class[str(i)], AP[i], instance_argsort[i]))
#         ax[2*i].set_xlabel('prob')
#         ax[2*i].set_ylabel('pdf')
#         ax[2*i].legend()
#         ax[2*i] = fig.add_subplot(num_class,2,2*i+2)
#         ax[2*i].hist(probs_N[i], label='negative', range=(0,1), bins=50, rwidth=0.8)
#         ax[2*i].set_title('probs of N samples')
#         ax[2*i].set_xlabel('prob')
#         ax[2*i].set_ylabel('pdf')
#         ax[2*i].legend()

#     fig.savefig(os.path.join('visualize', base_name+'_test_probs.jpg'))
    
    
    # save row data
    print('sample images')
    train_sample_indexes = np.random.choice([i for i in range(dataset_sizes['train'])], size=100, replace=False)
    test_sample_indexes = np.random.choice([i for i in range(len(image_datasets_test))], size=50, replace=False)
    train_sample_ann = image_datasets['train'].labels.iloc[train_sample_indexes].values
    test_sample_ann = image_datasets_test.labels.iloc[test_sample_indexes].values
    os.makedirs(os.path.join('row_data', base_name, 'train_img'), exist_ok=True)
    os.makedirs(os.path.join('row_data', base_name, 'test_img'), exist_ok=True)
    with open(os.path.join('row_data', base_name, 'train_sample_ann.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name']+[mscoco_class[str(i)] for i in range(80)])
        writer.writerows(train_sample_ann)
    with open(os.path.join('row_data', base_name, 'test_sample_ann.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name']+[mscoco_class[str(i)] for i in range(80)])
        writer.writerows(test_sample_ann)
    for i, idx in enumerate(train_sample_indexes):
        img_path = os.path.join(image_datasets['train'].image_dir, image_datasets['train'].labels.iloc[idx,0])
        new_img_path = os.path.join('row_data', base_name, 'train_img', image_datasets['train'].labels.iloc[idx,0])
        shutil.copy(img_path, new_img_path)
        model.train()
        image, _, _ = image_datasets['train'][idx]
        image = image.cuda().float().unsqueeze(0)
        with torch.no_grad():
            pred = model(image)
            prob = torch.sigmoid(pred)
            prob = prob.squeeze()
        with open(os.path.join('row_data', base_name, 'train_sample_pred.csv'), 'a') as f:
            writer = csv.writer(f)
            if i == 0:
                writer.writerow(['file_name']+[mscoco_class[str(i)] for i in range(80)])
            writer.writerow([image_datasets['train'].labels.iloc[idx,0]]+[round(prob[i].cpu().detach().item(), 2) for i in range(80)])
    for idx in test_sample_indexes:
        img_path = os.path.join(image_datasets_test.image_dir, image_datasets_test.labels.iloc[idx,0])
        new_img_path = os.path.join('row_data', base_name, 'test_img', image_datasets_test.labels.iloc[idx,0])
        shutil.copy(img_path, new_img_path)
        model.eval()
        image, _, _ = image_datasets_test[idx]
        image = image.cuda().float().unsqueeze(0)
        with torch.no_grad():
            pred = model(image)
            prob = torch.sigmoid(pred)
            prob = prob.squeeze()
        with open(os.path.join('row_data', base_name, 'test_sample_pred.csv'), 'a') as f:
            writer = csv.writer(f)
            if i == 0:
                writer.writerow(['file_name']+[mscoco_class[str(i)] for i in range(80)])
            writer.writerow([image_datasets_test.labels.iloc[idx,0]]+[round(prob[i].cpu().detach().item(), 2) for i in range(80)])
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--missing", default=0, type=int)
    parser.add_argument("--overlabeling", default=0, type=int)
    parser.add_argument("--ckpt_dir", default='results2', type=str)

    args = parser.parse_args()
    
    torch.manual_seed(727)
    np.random.seed(456)
        
    main(args)