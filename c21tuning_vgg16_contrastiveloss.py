# -*- encoding: utf-8 -*-
"""
@File    :   c04onnx_predict.py
@Contact :   zhujinchong@foxmail.com
@Author  :   zhujinchong
@Modify Time      @Version    @Desciption
------------      --------    -----------
2024/12/13 15:07    1.0         None
"""

import os
import random

import torch
import torch.nn.functional as F
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

        # 预先下载好模型，自定义加载权重
        # vgg = torchvision.models.vgg16()
        # vgg_weights = torch.load("./models/vgg16-397923af.pth", weights_only=True)
        # vgg.load_state_dict(vgg_weights)
        # 自动下载模型权重
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        del vgg.avgpool
        del vgg.classifier

        vgg = vgg.features
        self.vgg = vgg.eval()

        self.fc1 = nn.Sequential(
            nn.Linear(512 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 5)
        )

    def forward(self, x1, x2):
        x1, x2 = self.vgg(x1), self.vgg(x2)
        x1, x2 = torch.flatten(x1, 1), torch.flatten(x2, 1)
        o1, o2 = self.fc1(x1), self.fc1(x2)
        return o1, o2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    标签：同类为1， 不同类为0
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (label) * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


def train(model, train_loader, optimizer, criterion, epoch=0):
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    with tqdm(total=len(train_loader), desc="Epoch: %d" % (epoch + 1)) as pbar:
        for batch_idx, (data1, data2, label) in enumerate(train_loader):
            optimizer.zero_grad()
            data1, data2, label = data1.to(device), data2.to(device), label.to(device)
            output1, output2 = model(data1, data2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()  # Tensor -> float
            total += label.size(0)
            euclidean_distance = F.pairwise_distance(output1, output2)
            are_similar = euclidean_distance <= criterion.margin
            are_similar_int = are_similar.int()
            correct += torch.sum(torch.eq(are_similar_int, label.squeeze())).item()
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
                output1, output2 = model(data1, data2)
                loss = criterion(output1, output2, label)

                test_loss += loss.item()  # Tensor -> float
                total += label.size(0)
                euclidean_distance = F.pairwise_distance(output1, output2)
                are_similar = euclidean_distance <= criterion.margin
                are_similar_int = are_similar.int()
                correct += torch.sum(torch.eq(are_similar_int, label.squeeze())).item()

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
            transforms.Resize((105, 105)),
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

    # criterion = nn.BCELoss()
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 每5个epoch将优化器的学习率乘以0.1

    # 数据
    train_loader, test_loader = get_data_loader(batch_size=64)

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
