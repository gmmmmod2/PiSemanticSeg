# 语义分割项目 (Semantic Segmentation)

基于 PyTorch Lightning 和 segmentation_models_pytorch 的语义分割完整工程化实现。

## 📁 项目结构

```
project/
├── data/               # 数据集目录
│   └── CamVid/
│       ├── train/
│       ├── trainannot/
│       ├── val/
│       ├── valannot/
│       ├── test/
│       └── testannot/
├── models.py
│   ├── Model.py        # 模型定义和配置
├── Script
│   ├── Train.py        # 训练相关函数
│   ├── Test.py         # 测试相关函数
│   ├── Real.py         # 实际场景推理（图像/视频/摄像头）
│   ├── Other.py        # 工具函数
│   ├── Datasets.py     # 数据加载、预处理和可视化
├── Main.py             # 主训练脚本
├── checkpoints/        # 训练checkpoint保存
├── exports/            # 模型导出文件
└── experiments/        # 实验记录
```

## 🚀 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

将 CamVid 数据集放置在 `./data/CamVid/` 目录下。

### 3. 训练模型

```python
# 直接运行主训练脚本
python Main.py
```

或者自定义训练：

```python
from Datasets import get_dataloaders
from Model import create_model
from Train import train_model

# 加载数据
train_loader, valid_loader, test_loader, num_classes = get_dataloaders(
    data_dir="./data/CamVid/",
    batch_size=32,
    num_workers=4
)

# 创建模型
model = create_model(
    arch="Unet",
    encoder_name="mobileone_s4",
    in_channels=3,
    out_classes=num_classes,
    learning_rate=2e-4
)

# 训练
trainer = train_model(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    max_epochs=50
)
```

### 4. 测试模型

```python
from Model import load_model_from_checkpoint
from Test import test_from_checkpoint, visualize_test_results

# 从checkpoint测试
test_metrics = test_from_checkpoint(
    checkpoint_path="./checkpoints/best_model.ckpt",
    test_loader=test_loader
)

# 可视化结果
model = load_model_from_checkpoint("./checkpoints/best_model.ckpt")
visualize_test_results(model, test_loader, num_samples=5)
```

### 5. 实际场景推理

#### 单张图像测试

```python
from Model import load_model_from_checkpoint
from Real import test_single_image

model = load_model_from_checkpoint("./checkpoints/best_model.ckpt")

result = test_single_image(
    model=model,
    image_path="./test_image.jpg",
    save_path="./result.png"
)
```

#### 视频测试

```python
from Real import test_video

stats = test_video(
    model=model,
    video_path="./input_video.mp4",
    output_path="./output_video.mp4",
    show_window=True
)
```

#### 摄像头实时测试

```python
from Real import test_camera, test_camera_low_res

# 方式1：自定义分辨率（推荐）
test_camera(
    model=model,
    camera_id=0,
    display_size=(640, 480),      # 显示窗口大小
    capture_size=(640, 480),      # 摄像头捕获分辨率
    show_original_size=True       # 在标题显示分辨率信息
)

# 方式2：快捷低分辨率模式（更快推理速度）
test_camera_low_res(
    model=model,
    camera_id=0,
    resolution=(480, 360)  # 同时设置捕获和显示分辨率
)

# 方式3：超低分辨率获得最高FPS
test_camera_low_res(
    model=model,
    camera_id=0,
    resolution=(320, 240)  # 推理速度最快
)
```

提示：

- 降低分辨率可以显著提升推理速度（FPS）
- 摄像头 1600x800 → 480x360 可以提升约 3-4 倍推理速度
- 按 'q' 退出，按 's' 保存当前帧

#### 批量图像测试

```python
from Real import batch_test_images

all_stats = batch_test_images(
    model=model,
    image_dir="./test_images/",
    output_dir="./results/"
)
```

## 📊 模块说明

### Datasets.py - 数据处理模块

**主要功能：**

- `CamVidDataset`: 自定义数据集类
- `get_dataloaders()`: 获取训练/验证/测试数据加载器
- `get_training_augmentation()`: 训练数据增强
- `get_validation_augmentation()`: 验证数据增强
- `visualize_sample()`: 可视化单个样本
- `visualize_predictions()`: 可视化预测结果

### Model.py - 模型定义模块

**主要功能：**

- `SegmentationModel`: PyTorch Lightning 封装的分割模型
- `create_model()`: 创建新模型
- `load_model_from_checkpoint()`: 从 checkpoint 加载模型

**支持的模型架构：**

- Unet
- FPN
- DeepLabV3Plus
- PAN
- LinkNet
- PSPNet
- MAnet

### Train.py - 训练模块

**主要功能：**

- `create_trainer()`: 创建训练器（支持多种配置）
- `train_model()`: 训练模型
- `resume_training()`: 从 checkpoint 恢复训练
- `validate_model()`: 验证模型
- `get_best_checkpoint()`: 获取最佳 checkpoint

**特性：**

- 自动保存最佳模型
- 早停机制
- 学习率监控
- TensorBoard 日志
- 混合精度训练

### Test.py - 测试模块

**主要功能：**

- `test_model()`: 测试模型
- `test_from_checkpoint()`: 从 checkpoint 测试
- `visualize_test_results()`: 可视化测试结果
- `evaluate_metrics()`: 详细评估指标
- `save_test_results()`: 保存测试结果
- `batch_test()`: 批量测试多个 checkpoint

**评估指标：**

- IoU (Intersection over Union)
- F1 Score
- Accuracy
- Precision
- Recall

### Real.py - 实际场景推理模块

**主要功能：**

- `RealTimeSegmentation`: 实时分割推理器类
- `test_single_image()`: 单张图像测试（记录大小和速度）
- `test_video()`: 视频文件测试
- `test_camera()`: 摄像头实时测试
- `batch_test_images()`: 批量图像测试

**性能统计：**

- 推理时间（ms）
- FPS（帧率）
- 图像尺寸
- 平均/最小/最大推理时间

### Other.py - 工具函数模块

**主要功能：**

- `save_model_to_dir()`: 保存模型为目录格式
- `save_model_to_onnx()`: 导出 ONNX 模型
- `load_model_from_dir()`: 从目录加载模型
- `print_model_info()`: 打印模型信息
- `count_model_flops()`: 计算 FLOPs
- `compare_models()`: 比较多个模型性能
- `create_experiment_dir()`: 创建实验目录
- `get_device_info()`: 获取设备信息
- `cleanup_checkpoints()`: 清理旧的 checkpoint

## 🎯 使用示例

### 完整训练流程

```python
import pytorch_lightning as pl

# 设置随机种子
pl.seed_everything(42)

# 运行主训练脚本
from Main import main
model, metrics = main()
```

### 从 checkpoint 恢复训练

```python
from Train import resume_training
from Datasets import get_dataloaders

train_loader, valid_loader, _, _ = get_dataloaders("./data/CamVid/")

trainer = resume_training(
    checkpoint_path="./checkpoints/last.ckpt",
    train_loader=train_loader,
    valid_loader=valid_loader,
    max_epochs=100
)
```

### 模型对比实验

```python
from Other import compare_models
from Datasets import get_dataloaders

_, _, test_loader, _ = get_dataloaders("./data/CamVid/")

checkpoint_paths = [
    "./checkpoints/model_epoch_20.ckpt",
    "./checkpoints/model_epoch_30.ckpt",
    "./checkpoints/model_epoch_50.ckpt",
]

results = compare_models(checkpoint_paths, test_loader)
```

### 导出模型用于部署

```python
from Model import load_model_from_checkpoint
from Other import save_model_to_onnx, save_model_to_dir

# 加载模型
model = load_model_from_checkpoint("./checkpoints/best_model.ckpt")

# 导出为 ONNX（用于推理引擎）
save_model_to_onnx(
    model=model.model,
    save_path="./exports/model.onnx",
    input_shape=(1, 3, 480, 640)
)

# 导出为目录格式（用于继续训练或迁移学习）
save_model_to_dir(
    model=model.model,
    save_dir="./exports/saved_model",
    metrics={"test_iou": 0.85}
)
```

## 🔧 配置说明

### 主要配置参数

```python
CONFIG = {
    # 数据配置
    "data_dir": "./data/CamVid/",
    "batch_size": 32,
    "num_workers": 4,

    # 模型配置
    "arch": "Unet",              # 模型架构
    "encoder_name": "mobileone_s4",  # 编码器
    "in_channels": 3,             # 输入通道数

    # 训练配置
    "max_epochs": 50,
    "learning_rate": 2e-4,
    "scheduler_t_max": 50,
    "scheduler_eta_min": 1e-5,

    # 训练器配置
    "accelerator": "auto",        # 自动选择设备
    "devices": 1,                 # 使用设备数量
    "precision": "16-mixed",      # 混合精度训练
}
```

## 📈 实验追踪

### 查看 TensorBoard 日志

```bash
tensorboard --logdir ./checkpoints/camvid_segmentation
```

### 查看训练历史

训练过程中的所有指标都会自动记录到 TensorBoard，包括：

- 训练/验证损失
- IoU 指标
- 学习率变化

## 🐛 常见问题

### 1. CUDA 内存不足

**解决方案：**

- 减小 batch_size
- 使用梯度累积
- 使用混合精度训练（precision="16-mixed"）

### 2. 数据加载慢

**解决方案：**

- 增加 num_workers
- 使用 SSD 存储数据
- 减少数据增强的复杂度

### 3. 模型训练不收敛

**解决方案：**

- 检查学习率设置
- 使用学习率调度器
- 增加训练轮数
- 检查数据预处理是否正确

## 📝 开发建议

1. **实验管理**：使用 `create_experiment_dir()` 为每次实验创建独立目录
2. **版本控制**：使用 Git 管理代码，配置文件单独保存
3. **日志记录**：充分利用 TensorBoard 记录训练过程
4. **模型评估**：在多个数据集上测试模型泛化能力
5. **代码规范**：遵循 PEP 8 代码风格

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
