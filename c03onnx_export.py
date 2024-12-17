# -*- encoding: utf-8 -*-
"""
@File    :   c03onnx_export.py
@Contact :   zhujinchong@foxmail.com
@Author  :   zhujinchong
@Modify Time      @Version    @Desciption
------------      --------    -----------
2024/12/13 15:06    1.0         None
"""

import torch
from c01train import SiameseNetwork


def export(model_path, save_path="./models/model.onnx"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dummy = (torch.randn(1, 3, 105, 105).to(device), torch.randn(1, 3, 105, 105).to(device))
    model = SiameseNetwork()
    # model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model = torch.load("./models/model_0.pth")
    model.eval()
    model.to(device)
    torch.onnx.export(model, dummy, save_path, input_names=["x1", "x2"])
    print("finish!")


if __name__ == '__main__':
    export("./models/model_0.pth", "./models/model.onnx")
