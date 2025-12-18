""" Datasets.py - 数据加载、预处理和可视化模块 """
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as A


class CamVidDataset(BaseDataset):
    """CamVid数据集类"""
    
    CLASSES = [
        "sky", "building", "pole", "road", "pavement", "tree",
        "signsymbol", "fence", "car", "pedestrian", "bicyclist", "unlabelled",
    ]

    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None):
        """
        Args:
            images_dir: 图像文件夹路径
            masks_dir: 标注文件夹路径
            classes: 指定使用的类别列表
            augmentation: 数据增强方法
        """
        self.ids = sorted(os.listdir(images_dir))  # 排序以保证一致性
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.background_class = self.CLASSES.index("unlabelled")

        if classes:
            self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        else:
            self.class_values = list(range(len(self.CLASSES)))

        # 创建类别映射
        self.class_map = {self.background_class: 0}
        self.class_map.update({
            v: i + 1 for i, v in enumerate(self.class_values)
            if v != self.background_class
        })
        self.augmentation = augmentation

    def __getitem__(self, i):
        """获取单个样本"""
        image = cv2.imread(self.images_fps[i])
        if image is None:
            raise ValueError(f"无法读取图像: {self.images_fps[i]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.masks_fps[i], 0)
        if mask is None:
            raise ValueError(f"无法读取标注: {self.masks_fps[i]}")
        
        # 重映射mask类别
        mask_remap = np.zeros_like(mask)
        for class_value, new_value in self.class_map.items():
            mask_remap[mask == class_value] = new_value

        # 数据增强
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_remap)
            image, mask_remap = sample["image"], sample["mask"]
        
        # 转换为float32并归一化到[0, 1]
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        
        return image, mask_remap

    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    """获取训练集数据增强"""
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True),
        A.RandomCrop(height=320, width=320, always_apply=True),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf([
            A.CLAHE(p=1),
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
        ], p=0.9),
        A.OneOf([
            A.Sharpen(p=1),
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ], p=0.9),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.HueSaturationValue(p=1),
        ], p=0.9),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """获取验证集数据增强(仅padding)"""
    test_transform = [
        A.PadIfNeeded(384, 480),
    ]
    return A.Compose(test_transform)


def get_dataloaders(data_dir, batch_size=32, num_workers=0):
    """
    获取训练、验证、测试数据加载器
    
    Args:
        data_dir: 数据集根目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        
    Returns:
        train_loader, valid_loader, test_loader, num_classes
    """
    # 定义数据路径
    x_train_dir = os.path.join(data_dir, "train")
    y_train_dir = os.path.join(data_dir, "trainannot")
    x_valid_dir = os.path.join(data_dir, "val")
    y_valid_dir = os.path.join(data_dir, "valannot")
    x_test_dir = os.path.join(data_dir, "test")
    y_test_dir = os.path.join(data_dir, "testannot")
    
    # 检查目录是否存在
    for dir_path, dir_name in [
        (x_train_dir, "训练图像"), (y_train_dir, "训练标注"),
        (x_valid_dir, "验证图像"), (y_valid_dir, "验证标注"),
        (x_test_dir, "测试图像"), (y_test_dir, "测试标注"),
    ]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"{dir_name}目录不存在: {dir_path}")
    
    # 创建数据集
    train_dataset = CamVidDataset(
        x_train_dir, y_train_dir,
        augmentation=get_training_augmentation()
    )
    valid_dataset = CamVidDataset(
        x_valid_dir, y_valid_dir,
        augmentation=get_validation_augmentation()
    )
    test_dataset = CamVidDataset(
        x_test_dir, y_test_dir,
        augmentation=get_validation_augmentation()
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    
    num_classes = len(train_dataset.CLASSES)
    
    return train_loader, valid_loader, test_loader, num_classes


def visualize_sample(image, mask, title="Sample"):
    """
    可视化单个样本
    
    Args:
        image: 图像 (C, H, W) or (H, W, C), 值范围[0,1]或[0,255]
        mask: 标注 (H, W)
        title: 图像标题
    """
    plt.figure(figsize=(12, 6))
    
    # 显示图像
    plt.subplot(1, 2, 1)
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    
    # 如果是[0,1]范围，转换为[0,255]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")
    
    # 显示mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="tab20")
    plt.title("Mask")
    plt.axis("off")
    
    plt.suptitle(title)
    plt.show()


def visualize_predictions(images, gt_masks, pred_masks, num_samples=5):
    """
    可视化预测结果
    
    Args:
        images: 图像批次 (B, C, H, W)
        gt_masks: 真实标注 (B, H, W)
        pred_masks: 预测结果 (B, H, W)
        num_samples: 显示样本数量
    """
    num_samples = min(num_samples, len(images))
    
    for idx in range(num_samples):
        image = images[idx].cpu().numpy().transpose(1, 2, 0)
        gt_mask = gt_masks[idx].cpu().numpy()
        pr_mask = pred_masks[idx].cpu().numpy()
        
        # 如果是[0,1]范围，转换为[0,255]用于显示
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        plt.figure(figsize=(15, 5))
        
        # 原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Image")
        plt.axis("off")
        
        # 真实标注
        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask, cmap="tab20")
        plt.title("Ground Truth")
        plt.axis("off")
        
        # 预测结果
        plt.subplot(1, 3, 3)
        plt.imshow(pr_mask, cmap="tab20")
        plt.title("Prediction")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
