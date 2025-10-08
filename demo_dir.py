import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score

# 引入ResNet50模型
from networks.resnet import resnet50

# 引入进度条库
from tqdm import tqdm

class TestOptions:
    def __init__(self):
        self.model_path = "/root/autodl-tmp/cnndetection/CNNDetection/weights/blur_jpg_prob0.5.pth"
        self.isTrain = False
        self.serial_batches = False
        self.class_bal = False
        self.dataroot = "/root/autodl-tmp/cnndetection/CNNDetection/dataset/test/progan_testset/person"
        
        # 使用 os.scandir() 获取文件夹中的类别
        self.classes = [entry.name for entry in os.scandir(self.dataroot) if entry.is_dir()]
        print(f'Classes found in {self.dataroot}: {self.classes}')

# 创建 TestOptions 实例并直接使用
opt = TestOptions()

# 直接在代码中设置其他参数
batch_size = 32
workers = 4
use_cpu = False  # 如果设置为True，使用CPU；否则，使用GPU
crop = None  # 若为None则不裁剪

# 如果不只是查看图像尺寸，则进行模型加载和初始化
if True:
    # 设置计算设备：优先使用GPU，如果没有GPU或指定使用CPU，则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")

    # 初始化ResNet50模型，num_classes=1表示这是一个二分类任务
    model = resnet50(num_classes=1)

    # 如果指定了模型路径，则加载预训练模型权重
    if (opt.model_path is not None):
        state_dict = torch.load(opt.model_path, map_location=device)  # 自动适配 CPU/GPU
    model.load_state_dict(state_dict['model'])

    # 将模型转移到指定设备（CPU或GPU）
    model.to(device)

    # 将模型设置为评估模式，关闭训练时的随机行为（如Dropout）
    model.eval()

# 图像预处理
trans_init = []  # 存储图像预处理步骤的列表

# 如果裁剪参数opt.crop不为None，则进行中心裁剪
if crop is not None:
    trans_init = [transforms.CenterCrop(crop), ]
    print('Cropping to [%i]' % crop)
else:
    print('Not cropping')

# 定义一个图像转换操作：包括裁剪（如果指定了）以及后续的标准化处理
trans = transforms.Compose(trans_init + [
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 使用ImageNet的标准均值和标准差进行归一化
])

# 数据集加载
if type(opt.dataroot) == str:
    opt.dataroot = [opt.dataroot, ]  # 如果只有一个目录作为输入，转化为列表

print('Loading [%i] datasets' % len(opt.dataroot))

# 初始化数据加载器列表
data_loaders = []

# 对每个数据集目录进行处理
for dir in opt.dataroot:
    # 使用ImageFolder加载图像数据集，应用预处理
    dataset = datasets.ImageFolder(dir, transform=trans)

    # 创建数据加载器并加入列表
    data_loaders.append(torch.utils.data.DataLoader(dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=workers))

# 初始化标签和预测列表，Hs和Ws分别保存图像的高度和宽度
y_true, y_pred = [], []
Hs, Ws = [], []

# 使用`torch.no_grad()`进行推断，禁用梯度计算（推断时不需要反向传播）
with torch.no_grad():
    # 遍历数据加载器
    for data_loader in data_loaders:
        for data, label in tqdm(data_loader):  # 显示进度条
            # 记录图像的高度和宽度
            Hs.append(data.shape[2])
            Ws.append(data.shape[3])

            # 将标签展平并加入y_true列表
            y_true.extend(label.flatten().tolist())

            # 如果不只是查看图像尺寸，进行模型推断
            if True:
                if not use_cpu:
                    data = data.cuda()  # 如果使用GPU，将数据转移到GPU
                # 将数据输入模型，获取预测值，并存储到y_pred列表
                y_pred.extend(model(data).sigmoid().flatten().tolist())

# 将高度、宽度、标签和预测结果转换为numpy数组
Hs, Ws = np.array(Hs), np.array(Ws)
y_true, y_pred = np.array(y_true), np.array(y_pred)

# 输出图像的平均大小
print('Average sizes: [{:2.2f}+/-{:2.2f}] x [{:2.2f}+/-{:2.2f}] = [{:2.2f}+/-{:2.2f} Mpix]'.format(
    np.mean(Hs), np.std(Hs), np.mean(Ws), np.std(Ws), np.mean(Hs * Ws) / 1e6, np.std(Hs * Ws) / 1e6))

# 输出数据集中的真实图像和伪造图像的数量
print('Num reals: {}, Num fakes: {}'.format(np.sum(1 - y_true), np.sum(y_true)))

# 如果不只是查看图像尺寸，进行性能评估
if True:
    # 计算整体准确率
    acc = accuracy_score(y_true, y_pred > 0.5)

    # 计算真实图像的准确率
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)

    # 计算伪造图像的准确率
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)

    # 计算平均精度（Average Precision）
    ap = average_precision_score(y_true, y_pred)

    # 输出评估指标
    print('AP: {:2.2f}, Acc: {:2.2f}, Acc (real): {:2.2f}, Acc (fake): {:2.2f}'.format(
        ap * 100., acc * 100., r_acc * 100., f_acc * 100.))
