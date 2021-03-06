import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import time
import random
import numpy as np
import sys
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from utils import readfile, set_seed
from dataset import ImgDataset
from efficientnet_pytorch import EfficientNet

# Set Random seed
SEED = 0
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
workspace_dir = sys.argv[1]
print("Reading data")
train_x, train_y, val_x, val_y  = readfile(os.path.join(workspace_dir, "train/train"), True)
print("Size of training data = {}".format(len(train_x)))
print("Size of validation data = {}".format(len(val_x)))

#training 時做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), #隨機將圖片水平翻轉
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomRotation(30), #隨機旋轉圖片
    transforms.ToTensor(), #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)    
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Parameters initialize
batch_size = 32
train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=42).to(device)
#model = models.resnet152(pretrained=True).to(device)
#model.fc = nn.Linear(2048, 42).to(device)
#model.classifier = nn.Linear(1664, 42).to(device)

loss = nn.CrossEntropyLoss() # 因為是 classification task，所以 loss 使用 CrossEntropyLoss
optimizer = torch.optim.SGD(model.parameters(), lr=0.0125, momentum=0.9, weight_decay=1e-4) # optimizer 使用 Adam
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5)
num_epoch = 50
total_para = sum(p.numel() for p in model.parameters())
print('Parameter total:{}'.format(total_para))
best_acc = 0.0

print("Start Training!")

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    
    model.train() # 確保 model 是在 train model (開啟 Dropout 等...)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0].to(device)) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].to(device)) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
        optimizer.step() # 以 optimizer 用 gradient 更新參數值
        
        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].to(device))
            batch_loss = loss(val_pred, data[1].to(device))

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()
        if val_acc > best_acc:
            torch.save(model.state_dict(), 'ckpt.model')
            print('saving model with acc {:.3f}'.format(val_acc/val_set.__len__()*100))
            best_acc = val_acc
        #將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))
    
    scheduler.step(val_acc/val_set.__len__())
