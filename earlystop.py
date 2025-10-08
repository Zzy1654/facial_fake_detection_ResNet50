import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=1, verbose=False, delta=0):
        """
        初始化 EarlyStopping 类的实例。

        Args:
            patience (int): 在验证损失不再改进时，等待多少个 epochs 后停止训练。
                            默认值: 7
            verbose (bool): 如果为True，当验证损失改进时，打印相应的信息。
                            默认值: False
            delta (float): 监控的性能指标的最小变化，只有变化超过 delta 时才算作有改进。
                            默认值: 0
        """
        self.patience = patience  # 容忍的验证损失不改进的epoch数
        self.verbose = verbose  # 是否打印详细信息
        self.counter = 0  # 计数器，记录验证损失没有改进的次数
        self.best_score = None  # 记录最佳的验证损失或性能指标
        self.early_stop = False  # 是否触发早停机制
        self.score_max = -np.Inf  # 初始化最佳得分，设置为负无穷
        self.delta = delta  # 用于判断是否有足够的改进

    def __call__(self, score, model):
        """
        每次调用时检查模型的表现（验证集的损失或其他指标）是否有所改进，
        如果没有改进则增加计数器，并在计数器达到耐心值时停止训练。

        Args:
            score (float): 当前的性能指标值（如验证损失或准确度）。
            model (object): 当前训练的模型实例，用于保存最优模型。
        """
        # 如果是第一次调用，初始化最佳得分
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)  # 保存当前最好的模型
        # 如果当前得分小于最佳得分 - delta，表示没有改进
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # 如果没有改进的次数超过了耐心值，触发早停
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 如果当前得分改进了，更新最佳得分，保存模型，并重置计数器
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """当验证损失减少时，保存当前最好的模型。"""
        if self.verbose:
            print(f'Validation accuracy increased ({self.score_max:.6f} --> {score:.6f}).  Saving model ...')
        model.save_networks('best')  # 保存模型的权重（假设模型有save_networks方法）
        self.score_max = score  # 更新最佳得分

