import os
import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义 TestOptions 类，包含模型路径和数据路径
class TestOptions:
    def __init__(self):
        # 直接在代码中设置路径和其他参数，不通过命令行传参
        self.model_path = "/root/autodl-tmp/cnndetection/CNNDetection/weights/blur_jpg_prob0.5.pth"
        self.isTrain = False
        self.serial_batches = False
        self.class_bal = False
        self.dataroot = "/root/autodl-tmp/cnndetection/CNNDetection/dataset/val/progan_val/person"
        self.mode = 'binary' 
        self.no_crop = False
        self.cropSize = 224
        self.no_resize = False
        # 使用 os.scandir() 获取文件夹中的类别
        self.classes = [entry.name for entry in os.scandir(self.dataroot) if entry.is_dir()]
        print(f'Classes found in {self.dataroot}: {self.classes}')

# 创建 TestOptions 实例并直接使用
opt = TestOptions()

# 创建数据加载器
def create_dataloader(opt):
    # 确保数据目录存在
    if not os.path.isdir(opt.dataroot):
        raise FileNotFoundError(f"Dataset directory {opt.dataroot} does not exist.")
    
    print(f"Loading data from {opt.dataroot}")

    # 图像预处理：根据cropSize裁剪图像，如果没有cropSize，则不裁剪
    data_transforms = []
    if opt.no_resize:
        data_transforms.append(transforms.Resize(256))  # 如果需要调整大小
    if not opt.no_crop:
        data_transforms.append(transforms.CenterCrop(opt.cropSize))  # 如果不禁止裁剪，则执行裁剪
    
    data_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform = transforms.Compose(data_transforms)

    # 使用 ImageFolder 加载数据集
    dataset = datasets.ImageFolder(opt.dataroot, transform=transform)
    
    # 打印类别信息
    print(f"Classes found: {dataset.classes}")
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    return data_loader

# 验证函数
def validate(model, opt):
    # 创建数据加载器
    data_loader = create_dataloader(opt)

    # 进行模型推断
    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            # 将数据转移到 GPU
            img = img.cuda()
            label = label.cuda()

            # 执行推断并将结果保存
            outputs = model(img).sigmoid().flatten()  # sigmoid 是用来将输出转化为 [0, 1] 的概率值
            y_pred.extend(outputs.tolist())  # 预测结果
            y_true.extend(label.flatten().tolist())  # 真实标签

    # 将结果转换为 numpy 数组
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # 计算真实图像的准确率
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)

    # 计算伪造图像的准确率
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)

    # 计算整体准确率
    acc = accuracy_score(y_true, y_pred > 0.5)

    # 计算平均精度
    ap = average_precision_score(y_true, y_pred)

    # 返回所有评估结果
    return acc, ap, r_acc, f_acc, y_true, y_pred

# 主函数
if __name__ == '__main__':
    # 检查是否有可用的 GPU
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found. This code requires a GPU to run.")

    # 加载模型
    model = resnet50(num_classes=1)
    
    # 将模型加载到 GPU
    state_dict = torch.load(opt.model_path, map_location=torch.device('cuda'))
    
    model.load_state_dict(state_dict['model'])
    model.cuda()  # 确保模型在 GPU 上
    model.eval()

    # 调用验证函数
    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    # 打印结果
    print("accuracy:", acc)
    print("average precision:", avg_precision)
    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
