from util import mkdir  # 从util模块导入mkdir函数，用于创建目录

# directory to store the results
results_dir = './results/'  # 设置保存测试结果的目录路径
mkdir(results_dir)  # 如果目录不存在，创建该目录

# root to the testsets
dataroot = './dataset/test/'  # 设置测试集所在的根目录路径

# list of synthesis algorithms
vals = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan',
        'crn', 'imle', 'seeingdark', 'san', 'deepfake', 'stylegan2', 'whichfaceisreal']
# 这是一组假设用于生成伪造数据的合成算法名称列表（每个算法名称对应一个测试集）

# indicates if corresponding testset has multiple classes
multiclass = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
# 对应的测试集是否有多类标志。1 表示有多个类，0 表示是单一类。该列表长度与`vals`列表相同，每个元素对应一个算法的测试集。

# model
model_path = '/root/autodl-tmp/cnndetection/CNNDetection/weights/blur_jpg_prob0.5.pth'  # 训练好的模型的路径，假设是一个.pth文件，存储了模型的权重

