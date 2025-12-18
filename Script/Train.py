""" Train.py - 训练相关函数模块 """
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import os


def create_trainer(
    max_epochs=50,
    accelerator="auto",
    devices=1,
    precision="16-mixed",
    save_dir="./checkpoints",
    project_name="segmentation",
    enable_early_stopping=True,
    patience=10,
    log_every_n_steps=10,
):
    """
    创建PyTorch Lightning训练器
    
    Args:
        max_epochs: 最大训练轮数
        accelerator: 加速器类型 ('auto', 'gpu', 'cpu', 'tpu')
        devices: 设备数量
        precision: 训练精度 ('32', '16-mixed', 'bf16-mixed')
        save_dir: checkpoint保存目录
        project_name: 项目名称
        enable_early_stopping: 是否启用早停
        patience: 早停耐心值
        log_every_n_steps: 日志记录频率
        
    Returns:
        trainer: Lightning训练器
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 配置callbacks
    callbacks = []
    
    # ModelCheckpoint - 保存最佳模型
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename=f"{project_name}-{{epoch:02d}}-{{valid_dataset_iou:.4f}}",
        monitor="valid_dataset_iou",
        mode="max",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)
    
    # 额外保存一个 best_model.ckpt（方便使用）
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="best_model",
        monitor="valid_dataset_iou",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    callbacks.append(best_checkpoint_callback)
    
    # EarlyStopping - 早停
    if enable_early_stopping:
        early_stop_callback = EarlyStopping(
            monitor="valid_dataset_iou",
            patience=patience,
            mode="max",
            verbose=True,
        )
        callbacks.append(early_stop_callback)
    
    # LearningRateMonitor - 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # TensorBoard Logger
    logger = TensorBoardLogger(
        save_dir=save_dir,
        name=project_name,
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        deterministic=False,
    )
    
    return trainer


def train_model(
    model,
    train_loader,
    valid_loader,
    max_epochs=50,
    save_dir="./checkpoints",
    project_name="segmentation",
    **trainer_kwargs
):
    """
    训练模型
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        valid_loader: 验证数据加载器
        max_epochs: 最大训练轮数
        save_dir: checkpoint保存目录
        project_name: 项目名称
        **trainer_kwargs: 传递给trainer的其他参数
        
    Returns:
        trainer: 训练完成的trainer对象
    """
    # 创建训练器
    trainer = create_trainer(
        max_epochs=max_epochs,
        save_dir=save_dir,
        project_name=project_name,
        **trainer_kwargs
    )
    
    # 开始训练
    print(f"开始训练，共 {max_epochs} 个epoch...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )
    
    print(f"训练完成！最佳模型保存在: {trainer.checkpoint_callback.best_model_path}")
    
    return trainer


def resume_training(
    checkpoint_path,
    train_loader,
    valid_loader,
    max_epochs=50,
    **trainer_kwargs
):
    """
    从checkpoint恢复训练
    
    Args:
        checkpoint_path: checkpoint文件路径
        train_loader: 训练数据加载器
        valid_loader: 验证数据加载器
        max_epochs: 最大训练轮数
        **trainer_kwargs: 传递给trainer的其他参数
        
    Returns:
        trainer: 训练完成的trainer对象
    """
    from models.SMPmodel import load_model_from_checkpoint
    
    # 加载模型
    model = load_model_from_checkpoint(checkpoint_path)
    
    # 创建训练器
    trainer = create_trainer(max_epochs=max_epochs, **trainer_kwargs)
    
    # 恢复训练
    print(f"从 {checkpoint_path} 恢复训练...")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=checkpoint_path,
    )
    
    return trainer


def validate_model(model, valid_loader, trainer=None):
    """
    验证模型
    
    Args:
        model: 要验证的模型
        valid_loader: 验证数据加载器
        trainer: 训练器（可选）
        
    Returns:
        metrics: 验证指标
    """
    if trainer is None:
        trainer = pl.Trainer(accelerator="auto", devices=1)
    
    print("开始验证...")
    metrics = trainer.validate(model, dataloaders=valid_loader, verbose=True)
    
    return metrics[0] if metrics else None


def get_best_checkpoint(save_dir, project_name="segmentation"):
    """
    获取最佳checkpoint路径
    
    Args:
        save_dir: checkpoint保存目录
        project_name: 项目名称
        
    Returns:
        best_checkpoint_path: 最佳checkpoint路径
    """
    import glob
    
    pattern = os.path.join(save_dir, f"{project_name}-*.ckpt")
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    # 按修改时间排序，返回最新的
    best_checkpoint = max(checkpoints, key=os.path.getmtime)
    return best_checkpoint