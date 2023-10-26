#coding=utf-8
import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pickle as pickle
import copy
from tqdm import tqdm
import torch, torchvision
from torchvision import transforms
from torchvision.models import ResNet50_Weights
import glob
import shutil
import time,math
import torch
from torchvision import models
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

#####################################################
setname = 'CUB_H' #'CUB_H', 'Butterfly_H'
CUDA_VISIBLE_DEVICES = [2,3]
batch_size = 32
base_lr = 0.006
input_size = 448 #224,448

#####################################################
pretrained = True
transfer_mode = 'random'
transfer_mode_test = 'center'

model_name = 'resnet-50-coarse-fine-size-'
model_name += str(input_size) + '-hier'
model_path = './experiments/'+setname

##########################info about the training############################
end_epoch = 90
num_workers = 4
seed = 65
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
    
#set the initial learning rate
init_lr = base_lr*batch_size/64.0
lrf = min(1e-4,init_lr*0.01)
weight_decay = 1e-4

warmup_epoch = 20
lr_step_gamma = [30,60]
###########set the visible GPU cards#################
def set_gpu_envs(gpu_list):
    gpu_envs_str = ''
    if len(gpu_list) == 1:
        gpu_envs_str = str(gpu_list[0])
    else:
        for idx in range(len(gpu_list)-1):
            gpu_envs_str += (str(gpu_list[idx])+',')
        gpu_envs_str += str(gpu_list[-1])
        
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_envs_str
    return gpu_envs_str
###############################################

def spatial_random(imgdata, imgwith=224):
    transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(imgwith),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                   std=(0.229, 0.224, 0.225)),
        ])
    return transform(imgdata)

def spatial_center(imgdata, imgwith=224):
    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(int(imgwith*8/7)),# Let smaller edge match
            torchvision.transforms.CenterCrop(imgwith),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                   std=(0.229, 0.224, 0.225)),
        ])
    return transform(imgdata)

def process_image(imgdata, resize_width, transfer_mode): 

    if transfer_mode == 'random':
        imgdata = spatial_random(imgdata, resize_width)
        
    if transfer_mode == 'center':
        imgdata = spatial_center(imgdata, resize_width)

    return imgdata
           
class CUB_H(Dataset):
    def __init__(self, resize_width, transfer_mode, is_train=True, show_info=False, load_to_memory=False):
        self.resize_width = resize_width
        self.is_train = is_train
        self.transfer_mode = transfer_mode
        self.load_to_memory = load_to_memory
        self.use_cache = False
        self.root_dir = model_path+'/../../data/CUB_200_2011/CUB_200_2011/images'
        filelist_info = model_path+'/../../data/CUB_200_2011/CUB_200_2011'
        train_test_info = open(filelist_info+'/CUB_200_2011_train_test_multi_level_info.pkl', 'rb')
        filelist_train = pickle.load(train_test_info)
        label_train = pickle.load(train_test_info)
        filelist_test = pickle.load(train_test_info)
        label_test = pickle.load(train_test_info)
        train_test_info.close()

        assert len(label_train) == len(filelist_train)
        assert len(label_test) == len(filelist_test)
        if show_info:
            print('Number of samples: {:d}, train:{}, test:{}, mode:{}'.format(len(filelist_train)+len(filelist_test),
                                              len(filelist_train),len(filelist_test),self.transfer_mode))
        self.filelist_train = filelist_train
        self.label_train = label_train
        self.filelist_test = filelist_test
        self.label_test = label_test
        super(CUB_H, self).__init__()
        
    def __getitem__(self, index):
        if self.is_train:
            imgfile = self.filelist_train[index]
            imglabel = self.label_train[index]
        else:
            imgfile = self.filelist_test[index]
            imglabel = self.label_test[index]
        imgdata_pil = Image.open(os.path.join(self.root_dir,imgfile)).convert('RGB')
        imgdata = process_image(imgdata_pil, self.resize_width, self.transfer_mode)
        imgdata = imgdata.float()
        imglabel = torch.tensor(imglabel).long()
        return imgdata,imglabel

    def set_use_cache(self):
        self.use_cache = True

    def __len__(self):
        if self.is_train:
            filelen = len(self.filelist_train)
        else:
            filelen = len(self.filelist_test)
            
        return filelen
          
class Butterfly_H(Dataset):
    def __init__(self, resize_width, transfer_mode, is_train=True, show_info=False, load_to_memory=False):
        self.resize_width = resize_width
        self.is_train = is_train
        self.transfer_mode = transfer_mode
        self.load_to_memory = load_to_memory
        self.use_cache = False
        self.root_dir = model_path+'/../../data/Butterfly200/Butterfly200/images'
        filelist_info = model_path+'/../../data/Butterfly200/Butterfly200'
        train_test_info = open(filelist_info+'/Butterfly_train_test_multi_level_info.pkl', 'rb')
        filelist_train = pickle.load(train_test_info)
        label_train = pickle.load(train_test_info)
        filelist_test = pickle.load(train_test_info)
        label_test = pickle.load(train_test_info)
        train_test_info.close()

        assert len(label_train) == len(filelist_train)
        assert len(label_test) == len(filelist_test)
        if show_info:
            print('Number of samples: {:d}, train:{}, test:{}, mode:{}'.format(len(filelist_train)+len(filelist_test),
                                              len(filelist_train),len(filelist_test),self.transfer_mode))
        self.filelist_train = filelist_train
        self.label_train = label_train
        self.filelist_test = filelist_test
        self.label_test = label_test
        super(Butterfly_H, self).__init__()
        
    def __getitem__(self, index):
        if self.is_train:
            imgfile = self.filelist_train[index]
            imglabel = self.label_train[index]
        else:
            imgfile = self.filelist_test[index]
            imglabel = self.label_test[index]
        imgdata_pil = Image.open(os.path.join(self.root_dir,imgfile)).convert('RGB')
        imgdata = process_image(imgdata_pil, self.resize_width, self.transfer_mode)
        imgdata = imgdata.float()
        imglabel = torch.tensor(imglabel).long()
        return imgdata,imglabel

    def set_use_cache(self):
        self.use_cache = True

    def __len__(self):
        if self.is_train:
            filelen = len(self.filelist_train)
        else:
            filelen = len(self.filelist_test)
            
        return filelen

class ResNet(nn.Module):

    def __init__(self, features_base, feature_branch_list, fc_indim=512*4, grid_rate=1, num_classes=[13,37,122,200]):
        super(ResNet, self).__init__()
        
        self.features_base = features_base
        self.features_branch_1 = feature_branch_list[0]
        self.features_branch_2 = feature_branch_list[1]
        self.features_branch_3 = feature_branch_list[2]
        self.features_branch_4 = feature_branch_list[3]
        
        self.fc_ori_1 = nn.Linear(fc_indim, num_classes[0])
        self.fc_ori_2 = nn.Linear(fc_indim, num_classes[1])
        self.fc_ori_3 = nn.Linear(fc_indim, num_classes[2])
        self.fc_ori_4 = nn.Linear(fc_indim, num_classes[3])
        
        self.averpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b,c,w,h = x.size()
        x = self.features_base(x)
        
        x_out_1 = self.features_branch_1(x.detach())
        x_out_1 = self.averpool(x_out_1).view(b, -1)
        x_out_1 = self.fc_ori_1(x_out_1)
        
        x_out_2 = self.features_branch_2(x.detach())
        x_out_2 = self.averpool(x_out_2).view(b, -1)
        x_out_2 = self.fc_ori_2(x_out_2)
        
        x_out_3 = self.features_branch_3(x.detach())
        x_out_3 = self.averpool(x_out_3).view(b, -1)
        x_out_3 = self.fc_ori_3(x_out_3)
        
        x_out_4 = self.features_branch_4(x)
        x_out_4 = self.averpool(x_out_4).view(b, -1)
        x_out_4 = self.fc_ori_4(x_out_4)

        return [x_out_1,x_out_2,x_out_3,x_out_4]
        
#########################################################################################
#########################################################################################
    
def resnet50(pretrained=True, grid_rate=1, num_classes=[], **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    features_tmp_0 = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    features_base = torch.nn.Sequential(*list(features_tmp_0.children())[:-3])#conv1-3
    
    features_tmp_1 = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    features_branch_1 = torch.nn.Sequential(*list(features_tmp_1.children())[-3:-2])#conv1-3
    
    features_tmp_2 = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    features_branch_2 = torch.nn.Sequential(*list(features_tmp_2.children())[-3:-2])#conv1-3
    
    features_tmp_3 = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    features_branch_3 = torch.nn.Sequential(*list(features_tmp_3.children())[-3:-2])#conv1-3
    
    features_tmp_4 = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
    features_branch_4 = torch.nn.Sequential(*list(features_tmp_4.children())[-3:-2])#conv1-3
    
    feature_branch_list = [features_branch_1,features_branch_2,features_branch_3,features_branch_4]
    
    model = ResNet(features_base, feature_branch_list, 512*4, grid_rate, num_classes, **kwargs)
    
    if pretrained:
        print('Load pre-trained Resnet50 suucess!')
    else:
        print('Load Resnet50 suucess!')
    
    return model
    
def main():
    gpu_envs_str = set_gpu_envs(CUDA_VISIBLE_DEVICES)
    time_str = time.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(model_path, model_name)
    lr = init_lr
    softmax = torch.nn.Softmax(dim=1)
    codepath = os.path.join(save_path, "code-{}".format(time_str))
    if os.path.exists(codepath):
        shutil.rmtree(codepath)
    os.makedirs(codepath)

    if 'CUB' in setname:
        trainset = CUB_H(input_size, transfer_mode, is_train=True, show_info=True, load_to_memory=False)
        testset = CUB_H(input_size, transfer_mode_test, is_train=False, show_info=True, load_to_memory=False)
        
        train_test_info = open('./data/CUB_200_2011/CUB_200_2011/CUB_200_2011_train_test_multi_level_info.pkl', 'rb')
        filelist_train = pickle.load(train_test_info)
        label_train = pickle.load(train_test_info)
        filelist_test = pickle.load(train_test_info)
        label_test = pickle.load(train_test_info)
        label_len = pickle.load(train_test_info)
        _ = pickle.load(train_test_info)
        _ = pickle.load(train_test_info)
        _ = pickle.load(train_test_info)
        _ = pickle.load(train_test_info)
        trans_0_to_1 = pickle.load(train_test_info)
        trans_1_to_2 = pickle.load(train_test_info)
        trans_2_to_3 = pickle.load(train_test_info)
        trans_3_to_4 = pickle.load(train_test_info)
        train_test_info.close()

    if 'Butterfly' in setname:
        trainset = Butterfly_H(input_size, transfer_mode, is_train=True, show_info=True, load_to_memory=False)
        testset = Butterfly_H(input_size, transfer_mode_test, is_train=False, show_info=True, load_to_memory=False)
        
        train_test_info = open('./data/Butterfly200/Butterfly200/Butterfly_train_test_multi_level_info.pkl', 'rb')
        filelist_train = pickle.load(train_test_info)
        label_train = pickle.load(train_test_info)
        filelist_test = pickle.load(train_test_info)
        label_test = pickle.load(train_test_info)
        label_len = pickle.load(train_test_info)
        _ = pickle.load(train_test_info)
        _ = pickle.load(train_test_info)
        _ = pickle.load(train_test_info)
        _ = pickle.load(train_test_info)
        trans_0_to_1 = pickle.load(train_test_info)
        trans_1_to_2 = pickle.load(train_test_info)
        trans_2_to_3 = pickle.load(train_test_info)
        trans_3_to_4 = pickle.load(train_test_info)
        train_test_info.close()
        
    #get the mapping matrix
    trans_1_to_2 = torch.from_numpy(trans_1_to_2).type(torch.float32)
    trans_2_to_3 = torch.from_numpy(trans_2_to_3).type(torch.float32)
    trans_3_to_4 = torch.from_numpy(trans_3_to_4).type(torch.float32)
    trans_1_to_4 = torch.matmul(trans_1_to_2, torch.matmul(trans_2_to_3, trans_3_to_4)).cuda()
    trans_2_to_4 = torch.matmul(trans_2_to_3, trans_3_to_4).cuda()
    trans_3_to_4 = trans_3_to_4.cuda()
        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                pin_memory=True, drop_last=False, prefetch_factor=2, persistent_workers=True)
                                    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                                pin_memory=True, drop_last=False, prefetch_factor=2, persistent_workers=True)
        
    num_classes = label_len
    model = resnet50(pretrained=pretrained, num_classes=num_classes)
    print(num_classes)

    def test(model, testloader, criterion):
        model.eval()
        raw_correct = np.zeros([5])
        with torch.no_grad():
            for i, data in enumerate(tqdm(testloader)):
                images, labels = data
                images, labels = images.cuda(), labels.type(torch.long).cuda()
                outputs_list = model(images)
                
                # correct num
                one_merged = torch.zeros_like(outputs_list[-1])
                for subidx in range(len(outputs_list)):
                    pred = outputs_list[subidx].max(1, keepdim=True)[1]
                    raw_correct[subidx] += pred.eq(labels[:,subidx].view_as(pred)).sum().item()
                one_merged += torch.matmul(softmax(outputs_list[0]), trans_1_to_4)
                one_merged += torch.matmul(softmax(outputs_list[1]), trans_2_to_4)
                one_merged += torch.matmul(softmax(outputs_list[2]), trans_3_to_4)
                one_merged += softmax(outputs_list[-1])
                pred = one_merged.max(1, keepdim=True)[1]
                raw_correct[-1] += pred.eq(labels[:,-1].view_as(pred)).sum().item()
        raw_accuracy = raw_correct / len(testloader.dataset)
        return raw_accuracy
        
    acc_list = []
    loss_list = []
    start_epoch = 0

    ################check the start_epoch###############
    assert start_epoch < end_epoch

    ################设置系统可见GPU ID 使用虚拟ID从0开始###############
    #set_gpu_envs(CUDA_VISIBLE_DEVICES)#set before import torch
    print('GPU available:', gpu_envs_str)
    print('Data mode:', transfer_mode)
    vitual_gpu_id = [i for i in range(len(CUDA_VISIBLE_DEVICES))]
    model = torch.nn.DataParallel(model, device_ids=vitual_gpu_id)
    model.cuda()

    ########################设置训练参数#######################
    criterion = nn.CrossEntropyLoss().cuda()

    ########################设置优化器#######################
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, end_epoch, eta_min=lrf, last_epoch=-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_acc = 0
    first_epoch_flag = True
    time_str = time.strftime("%Y%m%d-%H%M%S")
    acc_save_file = open(os.path.join(codepath, time_str+'-acc-list.txt'), 'w')

    #record the acc and loss changes
    acc_array_save = acc_list
    loss_array_save = loss_list
        
    for epoch in range(start_epoch+1, end_epoch+1):
        model.cuda()
        model.train()
        lr = next(iter(optimizer.param_groups))['lr']
        train_loss_avg = 0
        for i, data in enumerate(tqdm(trainloader)):
            images, labels = data
            images, labels = images.cuda(), labels.type(torch.long).cuda()
            
            outputs_list = model(images)
            
            loss_1 = criterion(outputs_list[0], labels[:,0])
            loss_2 = criterion(outputs_list[1], labels[:,1])
            loss_3 = criterion(outputs_list[2], labels[:,2])
            loss_4 = criterion(outputs_list[3], labels[:,3])
            
            ####################
            total_loss = loss_1+loss_2+loss_3+loss_4
            train_loss_avg += total_loss
            
            ###############add the loss protect check###############
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
        scheduler.step()
        train_loss_avg /= len(trainloader)
        
        # eval testset
        raw_acc_list = test(model, testloader, criterion)

        save_one_epoch_str = '[{:d}/{:d}] Test raw accuracy: {:.2f}%,{:.2f}%,{:.2f}%,{:.2f}%, fused: {:.2f}%, loss: {:.3f}'.format(epoch,
                  end_epoch, 100.*raw_acc_list[0],100.*raw_acc_list[1],100.*raw_acc_list[2],100.*raw_acc_list[3],\
                             100.*raw_acc_list[4], train_loss_avg)
        print(save_one_epoch_str)
        
        acc_array_save.append(raw_acc_list)
        loss_array_save.append(train_loss_avg.detach().cpu().numpy())
        acc_save_file.write(save_one_epoch_str+'\n') 
        
        #############################################################################################
        torch.save({'epoch': epoch,
                    'model_state_dict': model.cpu().module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.cpu().state_dict(),
                    'learning_rate': lr,
                    'acc_array_save': acc_array_save,
                    'loss_array_save': loss_array_save},
                    os.path.join(codepath, 'last-model.pth'))
        #############################################################################################
    acc_save_file.close()
    
if __name__ == '__main__':
    #########################info about the training############################
    setname = 'Butterfly_H' #'CUB_H', 'Butterfly_H'
    CUDA_VISIBLE_DEVICES = [1,3]
    batch_size = 64
    base_lr = 0.010
    input_size = 224 #224,448

    #####################################################
    pretrained = True
    transfer_mode = 'random'
    transfer_mode_test = 'center'

    model_name = 'resnet-50-coarse-fine-size-'
    model_name += str(input_size) + '-hier'
    model_path = './experiments/'+setname

    end_epoch = 90
    num_workers = 4
    #####################################################
    main()
    
    
    ##########################info about the training############################
    setname = 'CUB_H' #'CUB_H', 'Butterfly_H'
    CUDA_VISIBLE_DEVICES = [1,3]
    batch_size = 32
    base_lr = 0.010
    input_size = 448 #224,448

    #####################################################
    pretrained = True
    transfer_mode = 'random'
    transfer_mode_test = 'center'

    model_name = 'resnet-50-coarse-fine-size-'
    model_name += str(input_size) + '-hier'
    model_path = './experiments/'+setname

    end_epoch = 20
    num_workers = 4
    #####################################################
    main()
    
    
    ##########################info about the training############################
    # setname = 'Butterfly_H' #'CUB_H', 'Butterfly_H'
    # CUDA_VISIBLE_DEVICES = [1,3]
    # batch_size = 32
    # base_lr = 0.014
    # input_size = 448 #224,448

    # #####################################################
    # pretrained = True
    # transfer_mode = 'random'
    # transfer_mode_test = 'center'

    # model_name = 'resnet-50-coarse-fine-size-'
    # model_name += str(input_size) + '-hier'
    # model_path = './experiments/'+setname

    # end_epoch = 90
    # num_workers = 4
    # #####################################################
    # main()
    
    
    
    
    
    
    
    
    
    
    
    
    # ##########################info about the training############################
    # setname = 'Butterfly_H' #'CUB_H', 'Butterfly_H'
    # CUDA_VISIBLE_DEVICES = [1,3]
    # batch_size = 32
    # base_lr = 0.014
    # input_size = 448 #224,448

    # #####################################################
    # pretrained = True
    # transfer_mode = 'random'
    # transfer_mode_test = 'center'

    # model_name = 'resnet-50-coarse-fine-size-'
    # model_name += str(input_size) + '-hier'
    # model_path = './experiments/'+setname

    # end_epoch = 90
    # num_workers = 4
    # #####################################################
    # main()
    
    
    # ##########################info about the training############################
    # setname = 'Butterfly_H' #'CUB_H', 'Butterfly_H'
    # CUDA_VISIBLE_DEVICES = [1,3]
    # batch_size = 32
    # base_lr = 0.014
    # input_size = 448 #224,448

    # #####################################################
    # pretrained = True
    # transfer_mode = 'random'
    # transfer_mode_test = 'center'

    # model_name = 'resnet-50-coarse-fine-size-'
    # model_name += str(input_size) + '-hier'
    # model_path = './experiments/'+setname

    # end_epoch = 90
    # num_workers = 4
    # #####################################################
    # main()
    
    
    # ##########################info about the training############################
    # setname = 'Butterfly_H' #'CUB_H', 'Butterfly_H'
    # CUDA_VISIBLE_DEVICES = [1,3]
    # batch_size = 32
    # base_lr = 0.014
    # input_size = 448 #224,448

    # #####################################################
    # pretrained = True
    # transfer_mode = 'random'
    # transfer_mode_test = 'center'

    # model_name = 'resnet-50-coarse-fine-size-'
    # model_name += str(input_size) + '-hier'
    # model_path = './experiments/'+setname

    # end_epoch = 90
    # num_workers = 4
    # #####################################################
    # main()
    
    
    # ##########################info about the training############################
    # setname = 'Butterfly_H' #'CUB_H', 'Butterfly_H'
    # CUDA_VISIBLE_DEVICES = [1,3]
    # batch_size = 32
    # base_lr = 0.014
    # input_size = 448 #224,448

    # #####################################################
    # pretrained = True
    # transfer_mode = 'random'
    # transfer_mode_test = 'center'

    # model_name = 'resnet-50-coarse-fine-size-'
    # model_name += str(input_size) + '-hier'
    # model_path = './experiments/'+setname

    # end_epoch = 90
    # num_workers = 4
    # #####################################################
    # main()
    