import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from torch.utils.tensorboard import SummaryWriter  # 用于将训练过程记录到 TensorBoard

from validate import validate  # 用于验证模型性能的函数
from data import create_dataloader  # 用于加载数据集的函数
from earlystop import EarlyStopping  # 早停类，用于防止过拟合
from networks.trainer import Trainer  # 模型训练器
from options.train_options import TrainOptions  # 用于处理训练参数的类


# 用于获取验证参数的函数
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)  # 获取训练的参数
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)  # 设置验证数据的根目录
    val_opt.isTrain = False  # 设置为验证模式
    val_opt.no_resize = False  # 不进行调整图像大小
    val_opt.no_crop = False  # 不进行裁剪
    val_opt.serial_batches = True  # 是否在整个数据集上进行迭代
    val_opt.jpg_method = ['pil']  # JPEG 方法
    if len(val_opt.blur_sig) == 2:  # 如果有两个模糊参数，取它们的平均值
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:  # 如果 JPEG 质量参数不为 1，取其平均值
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


# 主程序
if __name__ == '__main__':
    opt = TrainOptions().parse()  # 解析训练选项
    opt.dataroot = '/root/autodl-tmp/cnndetection/CNNDetection/dataset'.format(opt.dataroot, opt.train_split)  # 设置数据根目录
    val_opt = get_val_opt()  # 获取验证参数

    # 创建数据加载器
    data_loader = create_dataloader(opt)
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    # 创建 TensorBoard 记录器
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    # 初始化模型训练器
    model = Trainer(opt)
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)  # 初始化早停机制
    for epoch in range(opt.niter):  # 训练若干个 epoch
        epoch_start_time = time.time()  # 记录训练开始的时间
        iter_data_time = time.time()  # 记录每次迭代的数据加载时间
        epoch_iter = 0  # 记录每个 epoch 中的迭代次数

        for i, data in enumerate(data_loader):  # 遍历训练数据
            model.total_steps += 1  # 总步骤数加 1
            epoch_iter += opt.batch_size  # 更新当前 epoch 中已处理的样本数

            model.set_input(data)  # 设置当前输入数据
            model.optimize_parameters()  # 优化模型参数

            if model.total_steps % opt.loss_freq == 0:  # 每隔一定步数，输出当前训练损失
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)  # 记录损失到 TensorBoard

            if model.total_steps % opt.save_latest_freq == 0:  # 每隔一定步数保存最新模型
                print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                      (opt.name, epoch, model.total_steps))
                model.save_networks('latest')  # 保存模型

        if epoch % opt.save_epoch_freq == 0:  # 每隔一定 epoch 保存模型
            print('saving the model at the end of epoch %d, iters %d' % (epoch, model.total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)  # 保存模型

        # 验证模型
        model.eval()  # 切换到评估模式
        acc, ap = validate(model.model, val_opt)[:2]  # 获取模型的准确率和平均精度
        val_writer.add_scalar('accuracy', acc, model.total_steps)  # 记录验证准确率到 TensorBoard
        val_writer.add_scalar('ap', ap, model.total_steps)  # 记录验证平均精度到 TensorBoard
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

        early_stopping(acc, model)  # 使用早停机制进行检查
        if early_stopping.early_stop:  # 如果触发早停
            cont_train = model.adjust_learning_rate()  # 调整学习率
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)  # 重置早停
            else:
                print("Early stopping.")  # 停止训练
                break
        model.train()  # 继续训练模型


