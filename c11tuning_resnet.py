# -*- encoding: utf-8 -*-
"""
@File    :   c04onnx_predict.py
@Contact :   zhujinchong@foxmail.com
@Author  :   zhujinchong
@Modify Time      @Version    @Desciption
------------      --------    -----------
2024/12/13 15:07    1.0        替换vgg为resnet
"""

import os
import random

import torch
import torch.nn as nn
import torch.utils
import torchvision.models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    x = "cuda"
    print('正在使用gpu训练')
else:
    x = 'cpu'
    print('正在使用cpu训练')

device = torch.device(x)
torch.manual_seed(2023)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # 自动下载模型权重
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        resnet_features = resnet.fc.in_features  # fc输入是一维，长度是2048
        resnet.fc = nn.Sequential()  # 把fc置空  # del resnet.fc报错
        resnet.eval()

        self.resnet = resnet

        self.fcc = nn.Sequential(
            nn.Linear(resnet_features, 1023),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(1023, 1)
        )
        self.sgm = nn.Sigmoid()

    def forward(self, x1, x2):
        output1 = self.resnet(x1)
        output2 = self.resnet(x2)
        output = torch.abs(output1 - output2)
        output = self.fcc(output)
        output = self.sgm(output)
        return output


def train(model, train_loader, optimizer, criterion, epoch=0):
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    with tqdm(total=len(train_loader), desc="Epoch: %d" % epoch) as pbar:
        for batch_idx, (data1, data2, label) in enumerate(train_loader):
            optimizer.zero_grad()
            data1, data2, label = data1.to(device), data2.to(device), label.to(device)
            output = model(data1, data2)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()  # Tensor -> float
            total += label.size(0)
            correct += torch.sum(torch.eq(torch.round(output), label)).item()
            pbar.update(1)
            pbar.set_postfix({
                'loss': '%.4f' % (train_loss / (batch_idx + 1)),
                'train acc': '%.3f' % (correct / total)
            })

    acc = 100 * correct / total
    avg_loss = train_loss / len(train_loader)

    return avg_loss, acc


def test(model, test_loader, criterion, epoch=0):
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Epoch: %d" % epoch) as pbar:
            for batch_idx, (data1, data2, label) in enumerate(test_loader):
                data1, data2, label = data1.to(device), data2.to(device), label.to(device)
                output = model(data1, data2)
                loss = criterion(output, label)

                test_loss += loss.item()  # Tensor -> float
                total += label.size(0)
                correct += torch.sum(torch.eq(torch.round(output), label)).item()
                pbar.update(1)
                pbar.set_postfix({
                    'test loss': '%.4f' % (test_loss / (batch_idx + 1)),
                    'test acc': '%.3f' % (correct / total)
                })
    acc = 100 * correct / total
    avg_loss = test_loss / len(test_loader)

    return avg_loss, acc


class SiameseDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.num_samples = len(self.data)
        # 图片增强
        self.tpzq = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(40),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        sample = self.data[item]
        a, b, label = sample[0], sample[1], sample[2]
        a, b = Image.open(a), Image.open(b)
        a, b = self.tpzq(a), self.tpzq(b)
        label = torch.tensor([label], dtype=torch.float)
        return a, b, label


def process_data(data_path: str):
    # 图片文件夹：[图片路径]
    data_dict_path = {}  # {label: [img_path]}
    for label in os.listdir(data_path):
        data_dict_path[label] = [f"{data_path}/{label}/{img}" for img in os.listdir(f"{data_path}/{label}")]

    # 组装成正负样本
    all_data = []
    for label in data_dict_path:
        for img_path in data_dict_path[label]:
            sample_positive = random.choice(data_dict_path[label])  # 从当前路径选择
            tmp_labels = list(data_dict_path.keys())
            tmp_labels.remove(label)
            tmp_label = random.choice(tmp_labels)
            sample_negative = random.choice(data_dict_path[tmp_label])  # 从其它路径选择
            a_sample = [[img_path, sample_positive, 1], [img_path, sample_negative, 0]]
            all_data.extend(a_sample)
    return all_data


def get_data_loader(batch_size: int = 32):
    data_chinese = process_data("./datasets/chinese")
    data_icons = process_data("./datasets/icons")
    all_data = data_chinese + data_icons
    dataset = SiameseDataset(all_data)

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths=[train_size, test_size])
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    # 配置模型及优化器
    model = SiameseNetwork()
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 每5个epoch将优化器的学习率乘以0.1

    # 数据
    train_loader, test_loader = get_data_loader(batch_size=32)

    # 模型训练与测试
    min_loss = 10000
    epochs = 10
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = test(model, test_loader, criterion, epoch)
        scheduler.step()
        print(
            f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Test Accuracy={test_acc:.4f}")
        if test_loss < min_loss:
            min_loss = test_loss
            print('保存模型===>', f'./models/model_{str(epoch)}.pth')
            torch.save(model, f'./models/model_{str(epoch)}.pth')
