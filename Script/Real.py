""" Real.py - 实际场景测试模块（图像/视频/摄像头推理）"""
import torch
import cv2
import numpy as np
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt


class RealTimeSegmentation:
    """实时语义分割推理器"""
    
    def __init__(self, model, device="cuda", img_size=(480, 640)):
        """
        Args:
            model: 分割模型 (Lightning模块，包含forward中的预处理)
            device: 运行设备
            img_size: 输入图像大小 (height, width)
        """
        self.device = device
        self.img_size = img_size
        
        # 直接使用传入的模型（Lightning模块）
        # Lightning模块的forward方法已经包含了正确的预处理
        self.model = model.to(device)
        self.model.eval()
        
        # 性能统计
        self.inference_times = []
        self.processed_frames = 0
        
    def preprocess_image(self, image):
        """
        预处理图像 - 与训练数据格式保持一致！
        
        训练时 Datasets.py 返回的是:
        - uint8 转为 float (值范围仍是 0-255)
        - 然后在模型 forward 中做 (image - mean) / std
        
        Args:
            image: BGR格式的图像 (H, W, 3)
            
        Returns:
            tensor: 预处理后的张量 (1, 3, H, W)，值范围与训练一致
        """
        # BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image_resized = cv2.resize(image_rgb, (self.img_size[1], self.img_size[0]))
        
        # 关键：与训练数据保持一致！
        # 训练时 Datasets.py 的 __getitem__ 返回的是 uint8 转为的 tensor
        # image = image.transpose(2, 0, 1) 直接返回，没有 /255
        # 所以这里也不要做 /255 归一化
        image_tensor = torch.from_numpy(image_resized.copy()).float()
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        return image_tensor.to(self.device)
    
    def predict(self, image):
        """
        对单张图像进行预测
        
        Args:
            image: BGR格式的图像
            
        Returns:
            mask: 预测的分割mask (H, W)
            inference_time: 推理时间（秒）
        """
        start_time = time.time()
        
        # 预处理 - 保持与训练数据格式一致
        image_tensor = self.preprocess_image(image)
        
        # 推理 - 模型forward会做 (image - mean) / std
        with torch.inference_mode():
            logits = self.model(image_tensor)
            pred_mask = logits.softmax(dim=1).argmax(dim=1).squeeze(0)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.processed_frames += 1
        
        # 转回numpy
        mask = pred_mask.cpu().numpy()
        
        # Resize回原始大小
        mask = cv2.resize(mask.astype(np.uint8), 
                         (image.shape[1], image.shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
        
        return mask, inference_time
    
    def get_stats(self):
        """获取性能统计"""
        if not self.inference_times:
            return {}
        
        return {
            "total_frames": self.processed_frames,
            "avg_inference_time": np.mean(self.inference_times),
            "min_inference_time": np.min(self.inference_times),
            "max_inference_time": np.max(self.inference_times),
            "avg_fps": 1.0 / np.mean(self.inference_times),
        }


def test_single_image(model, image_path, save_path=None, device="cuda"):
    """
    测试单张图像
    
    Args:
        model: 分割模型 (Lightning模块)
        image_path: 图像路径
        save_path: 结果保存路径（可选）
        device: 运行设备
        
    Returns:
        result_dict: 包含预测结果和统计信息的字典
    """
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    print(f"\n处理图像: {image_path}")
    print(f"图像大小: {image.shape}")
    
    # 创建推理器
    segmentor = RealTimeSegmentation(model, device)
    
    # 预测
    mask, inference_time = segmentor.predict(image)
    
    # 打印调试信息
    unique_classes = np.unique(mask)
    print(f"预测类别: {unique_classes}")
    print(f"类别数量: {len(unique_classes)}")
    
    # 获取类别数量用于可视化
    num_classes = 12  # 默认值
    if hasattr(model, 'hparams') and hasattr(model.hparams, 'out_classes'):
        num_classes = model.hparams.out_classes
    
    # 创建彩色mask可视化
    colored_mask = create_colored_mask(mask, num_classes)
    
    # 叠加显示 - 注意colored_mask已经是BGR格式
    overlay = cv2.addWeighted(image, 0.6, colored_mask, 0.4, 0)
    
    # 显示结果
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="tab20", vmin=0, vmax=num_classes-1)
    plt.colorbar(shrink=0.8)
    plt.title(f"Segmentation Mask\n(classes: {unique_classes})")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Overlay")
    plt.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"结果已保存到: {save_path}")
    
    plt.show()
    
    # 统计信息
    result = {
        "image_path": str(image_path),
        "image_size": image.shape,
        "inference_time_ms": inference_time * 1000,
        "fps": 1.0 / inference_time,
        "unique_classes": unique_classes.tolist(),
        "num_classes_detected": len(unique_classes),
    }
    
    print(f"推理时间: {result['inference_time_ms']:.2f} ms")
    print(f"FPS: {result['fps']:.2f}")
    
    return result


def test_video(model, video_path, output_path=None, device="cuda", show_window=True):
    """
    测试视频
    
    Args:
        model: 分割模型
        video_path: 视频文件路径
        output_path: 输出视频路径（可选）
        device: 运行设备
        show_window: 是否显示实时窗口
        
    Returns:
        stats: 性能统计字典
    """
    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n处理视频: {video_path}")
    print(f"分辨率: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
    
    # 创建推理器
    segmentor = RealTimeSegmentation(model, device)
    
    # 获取类别数量
    num_classes = 12
    if hasattr(model, 'hparams') and hasattr(model.hparams, 'out_classes'):
        num_classes = model.hparams.out_classes
    
    # 创建视频写入器
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 预测
            mask, inference_time = segmentor.predict(frame)
            
            # 创建可视化
            colored_mask = create_colored_mask(mask, num_classes)
            overlay = cv2.addWeighted(frame, 0.6, colored_mask, 0.4, 0)
            
            # 添加信息文本
            fps_text = f"FPS: {1.0/inference_time:.1f}"
            cv2.putText(overlay, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 保存
            if writer:
                writer.write(overlay)
            
            # 显示
            if show_window:
                cv2.imshow('Segmentation', overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"已处理: {frame_count}/{total_frames} 帧")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()
    
    # 统计信息
    stats = segmentor.get_stats()
    stats.update({
        "video_path": str(video_path),
        "resolution": f"{width}x{height}",
        "original_fps": fps,
    })
    
    print(f"\n视频处理完成!")
    print(f"总帧数: {stats['total_frames']}")
    print(f"平均推理时间: {stats['avg_inference_time']*1000:.2f} ms")
    print(f"平均FPS: {stats['avg_fps']:.2f}")
    
    if output_path:
        print(f"输出视频已保存到: {output_path}")
    
    return stats


def test_camera(model, camera_id=0, device="cuda", display_size=(640, 480), 
                capture_size=None, show_original_size=False):
    """
    测试摄像头实时分割
    
    Args:
        model: 分割模型
        camera_id: 摄像头ID
        device: 运行设备
        display_size: 显示窗口大小, 默认640x480
        capture_size: 摄像头捕获分辨率, None则使用摄像头默认分辨率
        show_original_size: 是否在窗口标题显示原始摄像头分辨率
    """
    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"无法打开摄像头: {camera_id}")
    
    # 设置摄像头捕获分辨率（如果指定）
    if capture_size is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_size[1])
    
    # 获取实际摄像头分辨率
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    window_title = 'Real-time Segmentation'
    if show_original_size:
        window_title += f' (Camera: {actual_width}x{actual_height}, Display: {display_size[0]}x{display_size[1]})'
    
    print(f"\n开始摄像头测试")
    print(f"摄像头分辨率: {actual_width}x{actual_height}")
    print(f"显示分辨率: {display_size[0]}x{display_size[1]}")
    print(f"按 'q' 退出, 按 's' 保存当前帧")
    
    # 创建推理器
    segmentor = RealTimeSegmentation(model, device)
    
    # 获取类别数量
    num_classes = 12
    if hasattr(model, 'hparams') and hasattr(model.hparams, 'out_classes'):
        num_classes = model.hparams.out_classes
    
    saved_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 预测
            mask, inference_time = segmentor.predict(frame)
            
            # 创建可视化
            colored_mask = create_colored_mask(mask, num_classes)
            overlay = cv2.addWeighted(frame, 0.6, colored_mask, 0.4, 0)
            
            # 缩放到显示大小
            overlay_resized = cv2.resize(overlay, display_size)
            
            # 添加信息文本
            fps = 1.0 / inference_time
            info_text = [
                f"FPS: {fps:.1f}",
                f"Inference: {inference_time*1000:.1f}ms",
                f"Frame: {actual_width}x{actual_height}",
            ]
            
            y_offset = 30
            for i, text in enumerate(info_text):
                cv2.putText(overlay_resized, text, (10, y_offset + i*30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示
            cv2.imshow(window_title, overlay_resized)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                saved_count += 1
                save_path = f"camera_capture_{saved_count}.png"
                cv2.imwrite(save_path, overlay)
                print(f"已保存: {save_path}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # 打印统计信息
    stats = segmentor.get_stats()
    print(f"\n摄像头测试统计:")
    print(f"总帧数: {stats['total_frames']}")
    print(f"平均推理时间: {stats['avg_inference_time']*1000:.1f}ms")
    print(f"平均FPS: {stats['avg_fps']:.2f}")
    if saved_count > 0:
        print(f"保存帧数: {saved_count}")
        

def test_camera_low_res(model, camera_id=0, device="cuda", resolution=(480, 360)):
    """
    低分辨率摄像头测试（更快的推理速度）
    
    Args:
        model: 分割模型
        camera_id: 摄像头ID
        device: 运行设备
        resolution: 处理分辨率, 默认480x360
    
    使用示例:
        # 以480x360分辨率运行, 推理速度更快
        test_camera_low_res(model, camera_id=0, resolution=(480, 360))
        
        # 更低分辨率以获得更高FPS
        test_camera_low_res(model, camera_id=0, resolution=(320, 240))
    """
    print(f"\n低分辨率模式: {resolution[0]}x{resolution[1]}")
    print(f"提示: 降低分辨率可以显著提升推理速度")
    
    test_camera(
        model=model,
        camera_id=camera_id,
        device=device,
        display_size=resolution,
        capture_size=resolution,  # 同时设置捕获分辨率，减少缩放开销
        show_original_size=False
    )


def create_colored_mask(mask, num_classes=12):
    """
    创建彩色分割mask
    
    Args:
        mask: 分割mask (H, W)
        num_classes: 类别数量
        
    Returns:
        colored_mask: BGR格式的彩色mask
    """
    # CamVid类别颜色 (BGR格式，用于OpenCV)
    # sky, building, pole, road, pavement, tree, signsymbol, fence, car, pedestrian, bicyclist, unlabelled
    CAMVID_COLORS_BGR = np.array([
        [128, 128, 128],  # sky - 灰色
        [0, 0, 128],      # building - 深红
        [192, 192, 192],  # pole - 浅灰
        [128, 64, 128],   # road - 紫色
        [0, 0, 60],       # pavement - 深棕
        [0, 128, 0],      # tree - 绿色
        [0, 192, 192],    # signsymbol - 黄色
        [128, 128, 64],   # fence - 青色
        [64, 0, 0],       # car - 深蓝
        [0, 0, 192],      # pedestrian - 红色
        [0, 128, 192],    # bicyclist - 橙色
        [0, 0, 0],        # unlabelled - 黑色
    ], dtype=np.uint8)
    
    # 如果类别数超过预定义颜色，使用tab20
    if num_classes > len(CAMVID_COLORS_BGR):
        colors_rgb = plt.cm.tab20(np.linspace(0, 1, num_classes))[:, :3] * 255
        colors_bgr = colors_rgb[:, ::-1].astype(np.uint8)
    else:
        colors_bgr = CAMVID_COLORS_BGR[:num_classes]
    
    # 创建彩色mask
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in range(num_classes):
        colored_mask[mask == class_id] = colors_bgr[class_id]
    
    return colored_mask


def batch_test_images(model, image_dir, output_dir, device="cuda"):
    """
    批量测试图像文件夹
    
    Args:
        model: 分割模型
        image_dir: 图像文件夹路径
        output_dir: 输出文件夹路径
        device: 运行设备
        
    Returns:
        all_stats: 所有图像的统计信息列表
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in image_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    print(f"\n找到 {len(image_files)} 张图像")
    
    all_stats = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 处理: {image_path.name}")
        
        try:
            save_path = output_dir / f"{image_path.stem}_result.png"
            stats = test_single_image(model, image_path, save_path, device)
            all_stats.append(stats)
        except Exception as e:
            print(f"处理失败: {e}")
            continue
    
    # 保存统计信息
    stats_path = output_dir / "batch_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=4, ensure_ascii=False, default=str)
    
    print(f"\n批量测试完成！统计信息已保存到: {stats_path}")
    
    # 打印汇总
    if all_stats:
        avg_time = np.mean([s['inference_time_ms'] for s in all_stats])
        avg_fps = np.mean([s['fps'] for s in all_stats])
        print(f"平均推理时间: {avg_time:.2f} ms")
        print(f"平均FPS: {avg_fps:.2f}")
    
    return all_stats


# ============ 调试工具函数 ============

def debug_model_output(model, image_path, device="cuda"):
    """
    调试模型输出，检查预测是否正常
    
    Args:
        model: Lightning模型
        image_path: 测试图像路径
        device: 设备
    """
    print("\n" + "="*50)
    print("模型输出调试")
    print("="*50)
    
    # 读取图像
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (640, 480))
    
    # 方式1: 与训练数据一致 (0-255)
    image_tensor = torch.from_numpy(image_resized.copy()).float()
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    print(f"输入tensor形状: {image_tensor.shape}")
    print(f"输入值范围: [{image_tensor.min():.4f}, {image_tensor.max():.4f}]")
    
    # 模型信息
    model = model.to(device)
    model.eval()
    
    print(f"\n模型mean: {model.mean.squeeze()}")
    print(f"模型std: {model.std.squeeze()}")
    
    # 前向传播
    with torch.inference_mode():
        logits = model(image_tensor)
        probs = logits.softmax(dim=1)
        pred = probs.argmax(dim=1)
    
    print(f"\nLogits形状: {logits.shape}")
    print(f"Logits值范围: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"Logits均值（每个类别）: {logits.mean(dim=(0,2,3))}")
    
    print(f"\n概率最大值（每个像素）: {probs.max(dim=1)[0].mean():.4f}")
    print(f"预测类别分布: {torch.bincount(pred.flatten(), minlength=12)}")
    print(f"预测的唯一类别: {torch.unique(pred).tolist()}")
    
    return logits, probs, pred