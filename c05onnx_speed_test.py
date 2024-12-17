import os
import time

import cv2
import numpy as np
import onnxruntime

# 创建ONNX Runtime推理会话
ort_session = onnxruntime.InferenceSession("./models/model.onnx")


def process(image_bytes):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    # 将图片转换为numpy数组
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 确保输入的形状是 (C, H, W)
    image = cv2.resize(image, (105, 105)).astype(np.float32) / 255
    # 确保输入的形状是 (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    # 添加 batch 维度
    image = np.expand_dims(image, axis=0)
    return image


def predict(img_path1, img_path2):
    img1 = open(img_path1, 'rb').read()
    img2 = open(img_path2, 'rb').read()
    input1, input2 = process(img1), process(img2)

    outputs = ort_session.run(None, {'x1': input1, 'x2': input2})

    # 输出
    print(outputs[0][0][0])


def test_predict():
    test_data = []
    dir_list = os.listdir("./datasets/chinese/")
    for dir in dir_list:
        dir = "./datasets/chinese/" + dir + "/"
        cur = []
        for img_path in os.listdir(dir):
            img_path = dir + img_path
            cur.append(img_path)
        cur = cur[:2]
        test_data.append(cur)
    test_data = test_data[:1000]

    s = time.time()
    for i, (a, b) in enumerate(test_data):
        print(i)
        predict(a, b)
    e = time.time()
    print(e - s)


if __name__ == '__main__':
    test_predict()
