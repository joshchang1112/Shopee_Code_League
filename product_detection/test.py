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
from utils import read_test_file, set_seed
from dataset import ImgDataset
from efficientnet_pytorch import EfficientNet
# Set Random seed
SEED = 0
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
workspace_dir = sys.argv[1]

test_csv = pd.read_csv(os.path.join(workspace_dir, "test.csv"))
exist_filename = test_csv['filename'].tolist()

print("Reading data")
test_x, test_filename, = read_test_file(os.path.join(workspace_dir, "test/test"))
print("Size of testing data = {}".format(len(test_x)))


#testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Parameters initialize
batch_size = 32
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model_type = sys.argv[4]
model_num = int(sys.argv[5])

if model_type == 'r':
    if model_num == 152:
        model = torch.load('models/resnet152.model').to(device)
    elif model_num == 101:
        model = models.resnet101().to(device)
        model.fc = nn.Linear(2048, 42).to(device)
    else:
        model = models.resnet152().to(device)
        model.fc = nn.Linear(2048, 42).to(device)

elif model_type == 'd':
    model = models.densenet169().to(device)
    model.classifier = nn.Linear(1664, 42).to(device)

elif model_type == 'e':
    if model_num == 1:
        model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=42).to(device)
    elif model_num == 3:
        model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=42).to(device)
    elif model_num == 4:
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=42).to(device)
    elif model_num == 5:
        model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=42).to(device)
    elif model_num == 7:
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=42).to(device)

elif model_type == 'v':
    model = models.vgg16_bn().to(device)
    model.fc = nn.Linear(2048, 42).to(device)

if model_num != 152:
    model.load_state_dict(torch.load(sys.argv[2], map_location=device))
#model = torch.load('ckpt.model').to(device)
model.eval()

pred = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            pred.append(y)

with open(sys.argv[3], 'w') as f:
    f.write('filename,category\n')
    for i, y in enumerate(pred):
        if test_filename[i] in exist_filename:
            if y < 10:
                f.write('{},{}\n'.format(test_filename[i], '0'+str(y)))
            else:
                f.write('{},{}\n'.format(test_filename[i], str(y)))

