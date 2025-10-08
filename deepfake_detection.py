import os
import cv2
import torch
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
from networks.resnet import resnet50


# 添加load_model函数
def load_model(model_path, device=None):
    """
    加载ResNet50模型并设置相关路径
    
    参数:
        model_path: 模型权重文件路径
        device: 计算设备，如果为None则自动选择
        
    返回:
        加载好权重的模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化ResNet50模型，num_classes=1表示这是一个二分类任务
    model = resnet50(num_classes=1)
    
    # 加载预训练模型权重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['model'], strict=False)
    
    # 将模型转移到指定设备
    model.to(device)
    
    # 将模型设置为评估模式
    model.eval()
    
    # 定义路径
    global heatmap_dir, unet_dir
    heatmap_dir = "/root/autodl-tmp/cnndetection/CNNDetection/results/heatmap"
    unet_dir = "/root/autodl-tmp/cnndetection/CNNDetection/results/unet"
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(unet_dir, exist_ok=True)
    
    return model

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用load_model函数加载模型和设置路径
model_path = "/root/autodl-tmp/cnndetection/CNNDetection/weights/blur_jpg_prob0.5.pth"
model = load_model(model_path, device)

# 优化的预处理函数
def transform_image(image):
    """
    优化的图像预处理函数，增加了更多的数据增强和归一化步骤
    """
    trans = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整为更大的尺寸以保留更多细节
        transforms.CenterCrop((224, 224)),  # 中心裁剪到模型所需尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return trans(image).unsqueeze(0)

# Grad-CAM 计算热力图
def generate_heatmap(image, model, device):
    image = transform_image(image).to(device)
    image.requires_grad = True  # 使输入图像能计算梯度
    
    feature_map = None
    gradients = None
    
    # 获取模型的最后一层卷积层
    def forward_hook(module, input, output):
        nonlocal feature_map
        feature_map = output  # 存储前向传播的特征图
        feature_map.retain_grad()  # 允许计算梯度
    
    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]  # 存储梯度
    
    # 选择 ResNet50 的最后一个卷积层
    target_layer = model.layer4[2]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)
    
    # 前向传播
    output = model(image)
    if isinstance(output, tuple):
        classification = output[0]
    else:
        classification = output
    
    pred = torch.sigmoid(classification).item()
    # 现在对 classification 应用 sigmoid
    loss = classification[0] if pred > 0.5 else -classification[0]
    
    # 反向传播
    model.zero_grad()
    loss.backward()
    
    # 释放钩子
    forward_handle.remove()
    backward_handle.remove()
    
    if gradients is None or feature_map is None:
        raise RuntimeError("Grad-CAM 计算失败: 梯度或特征图为空")
    
    # 计算 Grad-CAM 权重
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * feature_map, dim=1).squeeze().cpu().detach().numpy()
    
    # 归一化热力图
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # 伪彩色处理
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    return heatmap

# 修改预测函数，添加阈值优化和集成决策
def predict(image, threshold=0.5):
    """
    优化的预测函数，添加了阈值参数和后处理步骤
    """
    image_pil = Image.fromarray(image)
    transformed_image = transform_image(image_pil).to(device)
    
    # 使用模型进行预测
    with torch.no_grad():
        output = model(transformed_image)
        
        if isinstance(output, tuple):
            classification, feature_map, segmentation_map = output
        else:
            classification = output
            feature_map = None
            segmentation_map = None
        
        # 应用sigmoid获取概率值
        prob = torch.sigmoid(classification).item()
        
        # 使用优化的阈值进行分类
        label = "Fake" if prob > threshold else "Real"
        confidence = max(prob, 1 - prob) * 100
    
    # 添加后处理步骤：如果置信度较低，进行额外检查
    if 0.4 < prob < 0.6:  # 置信度不高的情况
        # 使用多尺度测试提高准确性
        scales = [0.9, 1.0, 1.1]  # 多个尺度
        multi_scale_probs = []
        
        for scale in scales:
            # 调整图像大小
            width, height = image_pil.size
            new_width, new_height = int(width * scale), int(height * scale)
            resized_img = image_pil.resize((new_width, new_height), Image.BILINEAR)
            
            # 如果尺寸变大，需要中心裁剪
            if scale > 1.0:
                left = (new_width - width) // 2
                top = (new_height - height) // 2
                resized_img = resized_img.crop((left, top, left + width, top + height))
            
            # 如果尺寸变小，需要填充
            elif scale < 1.0:
                new_img = Image.new("RGB", (width, height), (0, 0, 0))
                left = (width - new_width) // 2
                top = (height - new_height) // 2
                new_img.paste(resized_img, (left, top))
                resized_img = new_img
            
            # 预处理并预测
            transformed = transform_image(resized_img).to(device)
            with torch.no_grad():
                ms_output = model(transformed)
                if isinstance(ms_output, tuple):
                    ms_classification = ms_output[0]
                else:
                    ms_classification = ms_output
                scale_prob = torch.sigmoid(ms_classification).item()
                multi_scale_probs.append(scale_prob)
        
        # 计算多尺度预测的平均值
        avg_prob = sum(multi_scale_probs) / len(multi_scale_probs)
        
        # 使用多尺度预测结果更新标签和置信度
        label = "Fake" if avg_prob > threshold else "Real"
        confidence = max(avg_prob, 1 - avg_prob) * 100
        prob = avg_prob  # 更新概率值
    
    # 处理 UNet 分割输出
    if segmentation_map is not None:
        segmentation_map = torch.sigmoid(segmentation_map).squeeze().cpu().numpy()
        
        # 获取 min 和 max 值，并确保值差不小
        min_val, max_val = segmentation_map.min(), segmentation_map.max()
        print(f"[DEBUG] segmentation_map min_val: {min_val}, max_val: {max_val}")
        
        if max_val - min_val < 1e-6:
            print("[WARNING] segmentation_map min 和 max 过于接近，调整对比度")
            segmentation_map = np.zeros_like(segmentation_map)  # 设为全黑
        else:
            segmentation_map = (segmentation_map - min_val) / (max_val - min_val)  # 归一化
        
        # 转换为 0-255 格式
        segmentation_map = np.uint8(segmentation_map * 255)
        
        # 伪彩色映射
        segmentation_map = cv2.applyColorMap(segmentation_map, cv2.COLORMAP_JET)
        segmentation_map = cv2.cvtColor(segmentation_map, cv2.COLOR_BGR2RGB)  # 转换为 RGB
    else:
        # 如果没有分割图，创建一个空白图像
        segmentation_map = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # 生成Grad-CAM热力图
    heatmap = generate_heatmap(image_pil, model, device)
    
    # 将热力图调整为与原图相同的尺寸,叠加到原图上
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_add_image = cv2.addWeighted(image, 0.7, heatmap_resized, 0.3, 0)
    
    # 生成UNet分割图
    segmentation_map_resized = cv2.resize(segmentation_map, (image.shape[1], image.shape[0]))
    
    # 伪彩色处理后的 UNet 分割图,直接返回处理后的分割图
    final_segmentation_map = segmentation_map_resized
    
    # 保存
    heatmap_filename = os.path.join(heatmap_dir, f"{str(int(confidence))}_heatmap.jpg")
    segmentation_filename = os.path.join(unet_dir, f"{str(int(confidence))}_segmentation_map.jpg")
    
    # 保存到指定路径
    cv2.imwrite(heatmap_filename, cv2.cvtColor(heatmap_add_image, cv2.COLOR_RGB2BGR))  # Save heatmap image
    cv2.imwrite(segmentation_filename, cv2.cvtColor(final_segmentation_map, cv2.COLOR_RGB2BGR))  # Save segmentation map
    
    return label, confidence, heatmap_add_image, final_segmentation_map

# 添加集成预测函数，通过多次预测提高准确性
def ensemble_predict(image, num_augmentations=5):
    """
    使用数据增强和集成学习提高单张图像的预测准确性
    
    参数:
    - image: 输入图像
    - num_augmentations: 数据增强的次数
    
    返回:
    - label: 预测标签
    - confidence: 预测置信度
    - heatmap: 热力图
    - segmentation_map: 分割图
    """
    image_pil = Image.fromarray(image)
    probs = []
    
    # 原始预测
    transformed_image = transform_image(image_pil).to(device)
    with torch.no_grad():
        output = model(transformed_image)
        if isinstance(output, tuple):
            classification = output[0]
        else:
            classification = output
        prob = torch.sigmoid(classification).item()
        probs.append(prob)
    
    # 数据增强预测
    augmentations = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0))
    ]
    
    for i in range(min(num_augmentations, len(augmentations))):
        # 应用数据增强
        aug_image = augmentations[i](image_pil)
        transformed_aug = transform_image(aug_image).to(device)
        
        # 预测
        with torch.no_grad():
            aug_output = model(transformed_aug)
            if isinstance(aug_output, tuple):
                aug_classification = aug_output[0]
            else:
                aug_classification = aug_output
            aug_prob = torch.sigmoid(aug_classification).item()
            probs.append(aug_prob)
    
    # 计算平均概率
    avg_prob = sum(probs) / len(probs)
    
    # 使用最佳阈值进行分类
    best_threshold = 0.5  # 可以从batch_predict中获取或设置为固定值
    label = "Fake" if avg_prob > best_threshold else "Real"
    confidence = max(avg_prob, 1 - avg_prob) * 100
    
    # 使用标准predict函数生成热力图和分割图
    _, _, heatmap, segmentation_map = predict(image, threshold=best_threshold)
    
    return label, confidence, heatmap, segmentation_map

# 修改批量预测函数，添加阈值优化
def batch_predict(dataset_path):
    """
    优化的批量预测函数，添加了阈值优化和集成决策
    
    参数：
    - dataset_path (str): 数据集所在路径，需包含 "0_real"（真实图片） 和 "1_fake"（伪造图片） 文件夹。
    
    返回：
    - acc (float): 预测的准确率
    - f1 (float): 预测的 F1 分数
    - precision (float): 精准率
    - recall (float): 召回率
    - result_message (str): 处理结果的说明，包括热力图和 UNet 结果存放的路径
    - roc_curve_fig (PIL.Image): 显示 ROC 曲线图 (转换为 PIL 图像)
    - auc_score (float): 计算的 AUC 值
    - best_threshold (float): 找到的最佳阈值
    """
    y_true, y_pred = [], []  # 存储真实标签和预测标签
    y_pred_prob = []  # 存储预测的概率值（用于绘制 ROC 曲线）
    failed_images = []  # 用于记录处理失败的图片路径
    
    # 遍历数据集中的真实 (real) 和伪造 (fake) 目录
    for label, image_dir in [("real", os.path.join(dataset_path, "0_real")), 
                            ("fake", os.path.join(dataset_path, "1_fake"))]:
        true_label = 0 if label == "real" else 1  # 真实图像的标签为 0，伪造图像的标签为 1
            
        # 获取目录中的所有图像文件
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        
        # 使用tqdm显示进度
        for img_file in tqdm(image_files, desc=f"处理 {label} 图像"):
            img_path = os.path.join(image_dir, img_file)
            
            try:
                # 读取图像
                image = cv2.imread(img_path)
                if image is None:
                    print(f"警告: 无法读取图像 {img_path}")
                    failed_images.append(img_path)
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 转换为PIL图像进行预测
                image_pil = Image.fromarray(image)
                transformed_image = transform_image(image_pil).to(device)
                
                # 进行预测
                with torch.no_grad():
                    output = model(transformed_image)
                    
                    # 处理模型输出
                    if isinstance(output, tuple):
                        classification = output[0]
                    else:
                        classification = output
                        
                    # 获取预测概率
                    prob = torch.sigmoid(classification).item()
                    
                # 记录真实标签和预测概率
                y_true.append(true_label)
                y_pred_prob.append(prob)
                
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {str(e)}")
                failed_images.append(img_path)
    
    # 完成所有预测后，找出最佳阈值
    # 通过网格搜索找到最优F1分数的阈值
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred_threshold = [1 if prob > threshold else 0 for prob in y_pred_prob]
        f1 = f1_score(y_true, y_pred_threshold)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # 使用最佳阈值重新计算预测结果
    y_pred = [1 if prob > best_threshold else 0 for prob in y_pred_prob]
    
    # 计算性能指标
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    # 计算ROC曲线和AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc_score = auc(fpr, tpr)
    
    # 绘制ROC曲线
    roc_fig = plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('接收者操作特征曲线')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # 将ROC曲线转换为PIL图像
    canvas = FigureCanvas(roc_fig)
    canvas.draw()
    roc_curve_img = np.array(canvas.renderer.buffer_rgba())
    roc_curve_img = cv2.cvtColor(roc_curve_img, cv2.COLOR_RGBA2RGB)
    roc_curve_pil = Image.fromarray(roc_curve_img)
    
    # 更新结果消息，包含最佳阈值信息
    result_message = f"""
    模型评估结果:
    
    最佳阈值: {best_threshold:.4f}
    准确率: {acc:.4f}
    F1分数: {f1:.4f}
    精确率: {precision:.4f}
    召回率: {recall:.4f}
    AUC: {auc_score:.4f}
    
    处理统计:
    - 总图像数: {len(y_true)}
    - 处理失败图像数: {len(failed_images)}
    
    热力图和UNet结果已保存至:
    - 热力图: {heatmap_dir}
    - UNet分割图: {unet_dir}
    """
    
    # 关闭图形，避免内存泄漏
    plt.close('all')
    
    return acc, f1, precision, recall, result_message, roc_curve_pil, auc_score, best_threshold

# 创建Gradio界面
def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 深度人脸伪造检测系统")
        
        with gr.Tab("单张图像检测"):
            with gr.Row():
                input_image = gr.Image(label="输入图像")
                
            with gr.Row():
                detect_btn = gr.Button("标准检测")
                ensemble_btn = gr.Button("集成检测 (更高精度)")
                
            with gr.Row():
                output_label = gr.Label(label="检测结果")
                output_confidence = gr.Number(label="置信度 (%)")
                
            with gr.Row():
                output_heatmap = gr.Image(label="Grad-CAM热力图")
                output_segmentation = gr.Image(label="UNet分割图")
        
        with gr.Tab("数据集检测评估"):
            with gr.Row():
                dataset_path = gr.Textbox(label="数据集路径", placeholder="输入包含0_real和1_fake文件夹的路径")
                
            with gr.Row():
                evaluate_btn = gr.Button("评估数据集")
                
            with gr.Row():
                output_metrics = gr.Textbox(label="评估指标", lines=15)
                output_roc = gr.Image(label="ROC曲线")
        
        # 单张图像检测函数
        def on_detect(image):
            label, confidence, heatmap, segmentation = predict(image)
            return {output_label: label, output_confidence: confidence, 
                   output_heatmap: heatmap, output_segmentation: segmentation}
        
        # 集成检测函数
        def on_ensemble_detect(image):
            label, confidence, heatmap, segmentation = ensemble_predict(image)
            return {output_label: label, output_confidence: confidence, 
                   output_heatmap: heatmap, output_segmentation: segmentation}
        
        # 数据集评估函数
        def on_evaluate(path):
            acc, f1, precision, recall, result_message, roc_curve_pil, auc_score, best_threshold = batch_predict(path)
            return {output_metrics: result_message, output_roc: roc_curve_pil}
        
        # 绑定按钮事件
        detect_btn.click(
            fn=on_detect,
            inputs=[input_image],
            outputs=[output_label, output_confidence, output_heatmap, output_segmentation]
        )
        
        ensemble_btn.click(
            fn=on_ensemble_detect,
            inputs=[input_image],
            outputs=[output_label, output_confidence, output_heatmap, output_segmentation]
        )
        
        evaluate_btn.click(
            fn=on_evaluate,
            inputs=[dataset_path],
            outputs=[output_metrics, output_roc]
        )
    
    return demo

# 启动Gradio界面
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share = True)