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

import torch.utils
import torchvision.transforms as transforms
from PIL import Image

from c11tuning_resnet import SiameseNetwork

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    x = "cuda"
    print('正在使用gpu识别')
else:
    x = 'cpu'
    print('正在使用cpu识别')

device = torch.device(x)

model = SiameseNetwork()
# model.load_state_dict(torch.load('./models/model_0.pth', weights_only=True, map_location=device))
model = torch.load("./models/model_0.pth")
model.eval()
model.to(device)
tpzq = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomRotation(40),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])


def predict(a: str, b: str):
    a, b = Image.open(a), Image.open(b)
    a, b = tpzq(a), tpzq(b)
    a, b = a.unsqueeze(0), b.unsqueeze(0)  # 扩展batch维
    a, b = a.to(device), b.to(device)
    output = model(a, b)
    print(output.item())


if __name__ == '__main__':
    a, b = './datasets/icons/00013_2/01.png', './datasets/icons/00013_2/02.png'
    predict(a, b)
    a, b = './datasets/icons/00013_2/01.png', './datasets/icons/00017_2/01.png'
    predict(a, b)
