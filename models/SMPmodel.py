""" SMPmodel.py - 模型定义和配置模块 """
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp


class SegmentationModel(pl.LightningModule):
    """语义分割模型"""
    
    def __init__(
        self,
        arch="Unet",
        encoder_name="mobileone_s4",
        in_channels=3,
        out_classes=12,
        learning_rate=2e-4,
        scheduler_t_max=50,
        scheduler_eta_min=1e-5,
        **kwargs
    ):
        """
        Args:
            arch: 模型架构 (Unet, FPN, DeepLabV3Plus等)
            encoder_name: 编码器名称
            in_channels: 输入通道数
            out_classes: 输出类别数
            learning_rate: 学习率
            scheduler_t_max: 学习率调度器周期
            scheduler_eta_min: 最小学习率
        """
        super().__init__()
        self.save_hyperparameters()
        
        # 创建分割模型
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        
        # 获取编码器预处理参数
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        
        # 损失函数
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        
        # 用于收集每个step的输出
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        """
        前向传播（包含预处理）
        
        Args:
            image: 输入图像 (B, C, H, W), 值范围应为[0, 1]
            
        Returns:
            mask: 预测的mask logits (B, classes, H, W)
        """
        # 图像归一化（使用编码器特定的mean/std）
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        """
        通用训练步骤
        
        Args:
            batch: (image, mask)
            stage: 'train' / 'valid' / 'test'
            
        Returns:
            dict: 包含loss和统计指标
        """
        image, mask = batch
        
        # 验证数据维度
        assert image.ndim == 4  # [B, C, H, W]
        assert mask.ndim == 3   # [B, H, W]
        
        # 确保image是float类型
        image = image.float()
        mask = mask.long()
        
        # 前向传播
        logits_mask = self.forward(image)
        
        # 确保输出维度正确
        assert logits_mask.shape[1] == self.hparams.out_classes
        
        # 计算损失
        logits_mask = logits_mask.contiguous()
        loss = self.loss_fn(logits_mask, mask)
        
        # 获取预测结果
        prob_mask = logits_mask.softmax(dim=1)
        pred_mask = prob_mask.argmax(dim=1)
        
        # 计算统计指标
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, 
            mode="multiclass", 
            num_classes=self.hparams.out_classes
        )
        
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        """
        每个epoch结束时的统计
        
        Args:
            outputs: 所有step的输出列表
            stage: 'train' / 'valid' / 'test'
        """
        if not outputs:
            return
            
        # 聚合所有step的指标
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        
        # 计算IoU
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        dataset_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro"
        )
        
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        self.log("train_loss", train_loss_info["loss"], prog_bar=True)
        return train_loss_info

    def on_train_epoch_end(self):
        """训练epoch结束"""
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        self.log("valid_loss", valid_loss_info["loss"], prog_bar=True)
        return valid_loss_info

    def on_validation_epoch_end(self):
        """验证epoch结束"""
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """测试步骤"""
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        """测试epoch结束"""
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.scheduler_t_max,
            eta_min=self.hparams.scheduler_eta_min
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def predict_step(self, batch, batch_idx):
        """预测步骤（用于推理）"""
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
        
        images = images.float()
        logits = self.forward(images)
        pred_masks = logits.softmax(dim=1).argmax(dim=1)
        return pred_masks


def load_model_from_checkpoint(checkpoint_path, map_location=None, **model_kwargs):
    """
    从checkpoint加载模型
    
    Args:
        checkpoint_path: checkpoint文件路径
        map_location: 设备映射（如 'cpu' 或 'cuda'）
        **model_kwargs: 模型初始化参数（如果需要覆盖）
        
    Returns:
        model: 加载的模型
    """
    model = SegmentationModel.load_from_checkpoint(
        checkpoint_path,
        map_location=map_location,
        **model_kwargs
    )
    model.eval()
    return model


def create_model(arch="Unet", encoder_name="mobileone_s4", in_channels=3, 
                out_classes=12, **kwargs):
    """
    创建新模型
    
    Args:
        arch: 模型架构
        encoder_name: 编码器名称
        in_channels: 输入通道数
        out_classes: 输出类别数
        **kwargs: 其他参数
        
    Returns:
        model: 创建的模型
    """
    model = SegmentationModel(
        arch=arch,
        encoder_name=encoder_name,
        in_channels=in_channels,
        out_classes=out_classes,
        **kwargs
    )
    return model
