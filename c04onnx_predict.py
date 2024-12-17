# -*- encoding: utf-8 -*-
"""
@File    :   c04onnx_predict.py
@Contact :   zhujinchong@foxmail.com
@Author  :   zhujinchong
@Modify Time      @Version    @Desciption
------------      --------    -----------
2024/12/13 15:07    1.0         None
"""

import cv2
import numpy as np
import onnxruntime


def predict(img_path1, img_path2, onnx_model_path="./models/model.onnx"):
    # 创建ONNX Runtime推理会话
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    image1 = cv2.imread(img_path1)
    image2 = cv2.imread(img_path2)

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # 将图片转换为numpy数组
    input1 = cv2.resize(image1, (105, 105)).astype(np.float32) / 255
    input2 = cv2.resize(image2, (105, 105)).astype(np.float32) / 255

    # 确保输入的形状是 (C, H, W)
    input1 = np.transpose(input1, (2, 0, 1))
    input2 = np.transpose(input2, (2, 0, 1))

    # 添加 batch 维度
    input1 = np.expand_dims(input1, axis=0)
    input2 = np.expand_dims(input2, axis=0)

    outputs = ort_session.run(None, {'x1': input1, 'x2': input2})

    # 输出
    print(outputs[0][0][0])


if __name__ == '__main__':
    a, b = './datasets/icons/00013_2/01.png', './datasets/icons/00013_2/02.png'
    predict(a, b)
    a, b = './datasets/icons/00013_2/01.png', './datasets/icons/00017_2/01.png'
    predict(a, b)
