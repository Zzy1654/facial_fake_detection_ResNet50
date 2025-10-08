import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib
from matplotlib.font_manager import FontProperties


# 设置中文字体
def set_chinese_font():
    # 尝试使用Windows系统中常见的中文字体
    font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi']

    for font_name in font_list:
        try:
            font_prop = FontProperties(fname=f"C:\\Windows\\Fonts\\{font_name}.ttf")
            matplotlib.rcParams['font.family'] = font_prop.get_name()
            return font_prop
        except:
            continue

    # 如果上述字体都不可用，使用matplotlib的默认配置
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    return None


def create_framework_flowchart():
    # 设置中文字体
    font_prop = set_chinese_font()

    # 创建图形和子图
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)

    # 设置样式
    plt.style.use('default')
    ax.set_axis_off()

    # 定义节点位置和大小
    node_width = 0.2
    node_height = 0.08

    # 根据流程图定义节点
    nodes = {
        # 输入阶段
        'input': (0.5, 0.95, '输入图像\n人脸图像预处理', 'lightblue'),

        # 特征提取阶段
        'resnet': (0.5, 0.8, 'ResNet模型\n特征提取与分类', 'lightsalmon'),

        # 三个并行分支
        'classification': (0.2, 0.65, '分类结果\n真实/伪造判断', 'lavender'),
        'gradcam': (0.5, 0.65, 'Grad-CAM\n可视化注意区域', 'lightgreen'),
        'unet': (0.8, 0.65, 'U-Net模型\n区域化分割', 'lightpink'),

        # 中间结果
        'heatmap': (0.5, 0.5, '热力图\n模型关注区域可视化', 'lightgreen'),
        'segmap': (0.8, 0.5, '分割图\n区域化特征表示', 'lightpink'),

        # 综合分析
        'fusion': (0.5, 0.35, '综合分析\n多维度结果融合', 'lavender'),

        # 最终输出
        'output': (0.5, 0.2, '最终检测结果\n真伪判断+区域化标记', 'lavender')
    }

    # 绘制节点
    for node_id, (x, y, text, color) in nodes.items():
        rect = patches.FancyBboxPatch(
            (x - node_width / 2, y - node_height / 2),
            node_width, node_height,
            boxstyle=patches.BoxStyle("Round", pad=0.02),
            facecolor=color, edgecolor='black', alpha=0.8, linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold', fontproperties=font_prop)

    # 定义箭头样式
    arrow_style = patches.ArrowStyle("Simple", head_length=10, head_width=8)

    # 绘制连接线
    connections = [
        # 主干连接
        ('input', 'resnet'),

        # 分支连接
        ('resnet', 'classification'),
        ('resnet', 'gradcam'),
        ('resnet', 'unet'),

        # 中间结果连接
        ('gradcam', 'heatmap'),
        ('unet', 'segmap'),

        # 综合分析连接
        ('classification', 'fusion'),
        ('heatmap', 'fusion'),
        ('segmap', 'fusion'),

        # 最终输出连接
        ('fusion', 'output')
    ]

    # 添加阶段标注
    stages = {
        'input_stage': (0.05, 0.95, '输入阶段'),
        'feature_stage': (0.05, 0.8, '特征提取阶段'),
        'branch_stage': (0.05, 0.65, '初步结果分析'),
        'mid_stage': (0.05, 0.5, '信息增强分析'),
        'fusion_stage': (0.05, 0.35, '信息融合阶段'),
        'output_stage': (0.05, 0.2, '输出阶段')
    }

    for stage_id, (x, y, text) in stages.items():
        ax.text(x, y, text, ha='left', va='center', fontsize=12, fontweight='bold', fontproperties=font_prop)

    # 绘制所有连接
    for start_id, end_id in connections:
        start_x, start_y = nodes[start_id][0], nodes[start_id][1] - node_height / 2
        end_x, end_y = nodes[end_id][0], nodes[end_id][1] + node_height / 2

        # 特殊处理从resnet到三个分支的连接
        if start_id == 'resnet' and end_id in ['classification', 'gradcam', 'unet']:
            # 计算中间点
            mid_y = (start_y + end_y) / 2

            # 创建路径点
            verts = [
                (start_x, start_y),  # 起点
                (start_x, mid_y),  # 中间点1
                (end_x, mid_y),  # 中间点2
                (end_x, end_y)  # 终点
            ]

            codes = [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO
            ]

            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', edgecolor='black', lw=1.5)
            ax.add_patch(patch)

            # 添加箭头
            arrow = patches.FancyArrowPatch(
                (end_x, mid_y + 0.01),
                (end_x, end_y - 0.01),
                arrowstyle=arrow_style,
                color='black',
                lw=1.5
            )
            ax.add_patch(arrow)

            # 添加分支说明文字
            if end_id == 'classification':
                ax.text((start_x + end_x) / 2, mid_y + 0.02, '特征提取结果', fontsize=8, ha='center',
                        fontproperties=font_prop)
            elif end_id == 'gradcam':
                ax.text((start_x + end_x) / 2, mid_y + 0.02, '提取可视化特征', fontsize=8, ha='center',
                        fontproperties=font_prop)
            elif end_id == 'unet':
                ax.text((start_x + end_x) / 2, mid_y + 0.02, '生成特征区域分割', fontsize=8, ha='center',
                        fontproperties=font_prop)

        # 特殊处理到融合节点的连接
        elif end_id == 'fusion':
            # 计算中间点
            mid_y = (start_y + end_y) / 2

            # 创建路径点
            verts = [
                (start_x, start_y),  # 起点
                (start_x, mid_y),  # 中间点1
                (end_x, mid_y),  # 中间点2
                (end_x, end_y)  # 终点
            ]

            codes = [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO
            ]

            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', edgecolor='black', lw=1.5)
            ax.add_patch(patch)

            # 添加箭头
            arrow = patches.FancyArrowPatch(
                (end_x, mid_y + 0.01),
                (end_x, end_y - 0.01),
                arrowstyle=arrow_style,
                color='black',
                lw=1.5
            )
            ax.add_patch(arrow)

            # 添加融合说明文字
            if start_id == 'classification':
                ax.text((start_x + end_x) / 2, mid_y - 0.02, '分类结果、热力图和分割图融合', fontsize=8, ha='center',
                        fontproperties=font_prop)
        else:
            # 直接连接
            arrow = patches.FancyArrowPatch(
                (start_x, start_y),
                (end_x, end_y),
                arrowstyle=arrow_style,
                color='black',
                lw=1.5
            )
            ax.add_patch(arrow)

    # 添加标题
    plt.title('深度人脸伪造检测系统框架', fontsize=18, fontweight='bold', pad=20, fontproperties=font_prop)

    # 添加框架说明
    description = (
        "本框架基于ResNet50主干网络，通过分类、Grad-CAM热力图和U-Net分割三个分支，\n"
        "实现对人脸伪造的高精度检测与可视化解释，最终通过多维度结果融合输出综合判断结果。"
    )
    plt.figtext(0.5, 0.05, description, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8),
                fontproperties=font_prop)

    # 设置图形边界
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 保存高清图像
    plt.savefig('e:\\xunlei\\Github\\CNNDetection\\framework_flowchart.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print("流程图已保存至: e:\\xunlei\\Github\\CNNDetection\\framework_flowchart.png")


if __name__ == '__main__':
    create_framework_flowchart()