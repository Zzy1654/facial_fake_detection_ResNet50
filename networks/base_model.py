import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


# 基础模型类
class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt  # 配置参数
        self.total_steps = 0  # 训练步数
        self.isTrain = opt.isTrain  # 是否是训练模式
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # 模型保存目录
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')  # 使用 GPU 或 CPU

    # 保存模型
    def save_networks(self, epoch):
        save_filename = 'model_epoch_%s.pth' % epoch  # 生成保存的文件名
        save_path = os.path.join(self.save_dir, save_filename)  # 保存路径

        # 保存模型和优化器的状态字典
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'total_steps' : self.total_steps,
        }

        torch.save(state_dict, save_path)  # 保存模型

    # 从磁盘加载模型
    def load_networks(self, epoch):
        load_filename = 'model_epoch_%s.pth' % epoch  # 文件名
        load_path = os.path.join(self.save_dir, load_filename)  # 文件路径

        print('loading the model from %s' % load_path)  # 打印加载路径
        state_dict = torch.load(load_path, map_location=self.device)  # 加载模型

        if hasattr(state_dict, '_metadata'):  # 删除可能的冗余数据
            del state_dict._metadata

        self.model.load_state_dict(state_dict['model'],strict= False)  # 加载模型权重
        self.total_steps = state_dict['total_steps']  # 更新训练步数

        if self.isTrain and not self.opt.new_optim:  # 如果是训练模式，且没有使用新优化器
            self.optimizer.load_state_dict(state_dict['optimizer'])  # 加载优化器
            # 将优化器的状态迁移到GPU
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            for g in self.optimizer.param_groups:
                g['lr'] = self.opt.lr  # 更新学习率

    # 设置模型为评估模式
    def eval(self):
        self.model.eval()

    # 进行测试（不计算梯度）
    def test(self):
        with torch.no_grad():
            self.forward()


# 初始化模型权重函数
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        # 针对卷积层或全连接层进行初始化
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)  # 正态分布初始化
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)  # Xavier初始化
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # Kaiming初始化
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)  # 正交初始化
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            # 如果存在偏置，则初始化偏置为0
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # 针对BatchNorm层的初始化
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)  # 打印初始化类型
    net.apply(init_func)  # 应用初始化函数
