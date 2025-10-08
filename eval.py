import os
import csv
import torch

from validate import validate  # 导入验证函数
from networks.resnet import resnet50  # 导入ResNet50模型
from options.test_options import TestOptions  # 导入测试选项的解析类
from eval_config import *  # 导入一些评估配置项，如dataroot，vals等




# Running tests
opt = TestOptions().parse(print_options=False)  # 解析测试选项参数
model_name = os.path.basename(model_path).replace('.pth', '')  # 获取模型名称并去掉.pth后缀
rows = [["{} model testing on...".format(model_name)],  # 初始化CSV文件内容，包含模型名称
        ['testset', 'accuracy', 'avg precision']]  # CSV文件表头

print("{} model testing on...".format(model_name))  # 打印模型名称

# 遍历不同的验证集进行测试
for v_id, val in enumerate(vals):  # 依次获取所有验证集
    opt.dataroot = '/root/autodl-tmp/cnndetection/CNNDetection/dataset/val/progan_val/person'  # 设置验证集路径
    opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']  # 如果是多分类，获取类别；否则只使用空字符串表示无类别
    opt.no_resize = True    # 默认不对图片进行resize

    # 加载ResNet50模型
    model = resnet50(num_classes=1)
    state_dict = torch.load(model_path, map_location='cpu')  # 加载模型的状态字典
    model.load_state_dict(state_dict['model'])  # 加载模型参数
    model.cuda()  # 将模型转移到GPU
    model.eval()  # 设置模型为评估模式

    # 进行验证，获取准确率（acc）和平均精度（ap）
    acc, ap, _, _, _, _ = validate(model, opt)
    rows.append([val, acc, ap])  # 将测试集的名称、准确率和平均精度加入到结果列表中
    print("({}) acc: {}; ap: {}".format(val, acc, ap))  # 打印当前测试集的准确率和平均精度

# 将测试结果保存到CSV文件中
csv_name = results_dir + '/{}.csv'.format(model_name)  # 生成CSV文件名，文件名为模型名称
with open(csv_name, 'w') as f:  # 打开CSV文件并写入
    csv_writer = csv.writer(f, delimiter=',')  # 创建CSV写入器
    csv_writer.writerows(rows)  # 写入所有行

