# 1 目的

点选验证码，需要匹配两个相似图片，包括文字和图标

（两个图片的相似度计算）

# 2 方法

方法：训练一个孪生网络

参考教程：https://www.52pojie.cn/thread-1888314-1-1.html

参考代码：https://github.com/2833844911/dianxuan

# 3 环境

```
1. 安装torch，去官网 https://pytorch.org/ 选择合适的版本，如：
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

2. 安装其它
pip install Pillow tqdm

3. 转成onnx相关包
pip install onnx onnxruntime cv2 opencv-python
```

# 4 项目结构

```
datasets/       # 数据集
models/         # 模型存储
c01train.py     # vgg16 训练
c02predict.py   # vgg16 测试torch模型
c03onnx_export.py       # vgg16 导出为onnx模型
c04onnx_predict.py      # vgg16 测试onnx模型
c05onnx_speed_test.py   # vgg16 测试onnx模型速度
c11tuning_resnet.py     # resnet50 训练
c21tuning_vgg16_contrastiveloss.py    # 对比学习
data_crawler.py         # 使用selenium抓取图标数据
```

# 5 爬取数据（可选）

安装依赖

```shell
pip install selenium, webdriver_manager
```

Linux需要安装google浏览器：

```shell
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb
sudo apt-get install -f
google-chrome --version
```

step1. 适当修改 data_crawler.py 代码

step2. 爬取后手动分类

# 6 测试

数据集来源：

1. chinese文字来源于 https://github.com/2833844911/dianxuan  (**数据分类很多是错误的，需要数据清洗！！！**)
2. icons图标来源于 https://github.com/xinhaojin/click-based-captcha-dataset?tab=readme-ov-file
3. icons_craw来源：自己根据实际情况爬取的图片

数据量：

- 汉字：1829类 71088个
- 图标：812类 19848个
- 自己爬取图标：50类 100个

训练时显存大小：(分别在batch-size=32和64时)

- vgg16: 2.1 - 3.3G
- resnet50: 6.3 - 12G

最终模型大小：

- vgg16： 66M
- resnet50: 99M

效果：

参数：训练集:测试集=9:1, batch=64, lr=0.0001

1. vgg16, 在epoch=1; train acc=0.839, test acc=0.904
2. vgg16, 在epoch=5; train acc=0.960, test acc=0.953
3. resnet50, 在epoch=1; train acc=0.873, test acc=0.970
4. resnet50, 在epoch=5; train acc=0.994, test acc=0.991

速度：

1. vgg16_onnx，在window上，占用内存0.1G，速度：每秒~30次；
2. resnet50_onnx，在window上，占用内存0.6G，速度：每秒~22次；

# 7 调优

## 7.1 原方法

流程：
1. vgg16主要用于提取图像特征
2. 将两张图片的特征向量相减得到一个新的向量
3. 将相减得到的新向量输入到一个全连接层得到一个标量
4. 最后经过sigmoid函数，如果两张图片时同一个类别输出接近1，相反输出应该接近0

损失函数：
- 交叉熵损失

## 7.2 对比学习

将目标函数改成对比学习损失函数:

```python
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
```

结果：

- vgg16, 在epoch=1; train acc=0.750, test acc=0.823

- vgg16, 在epoch=5; train acc=0.848, test acc=0.837

