""" Main.py - 主训练脚本 """
import os
from Script.Datasets import get_dataloaders
from models.SMPmodel import create_model
from Script.Train import train_model, validate_model
from Script.Test import evaluate_metrics, save_test_results
from Script.Other import save_model_to_dir, save_model_to_onnx, print_model_info

def main():
    """主训练流程"""
    
    # ==================== 配置参数 ====================
    CONFIG = {
        "data_dir": "./data/CamVid/",          # 数据配置, 数据路径
        "batch_size": 16,                      # 数据配置, 批次大小
        "num_workers": 2,                      # 数据配置, 线程数量
        "arch": "Unet",                        # 模型配置, 架构选择
        "encoder_name": "mobileone_s4",        # 模型配置，编码器选择
        "in_channels": 3,                      # 模型配置, 输入通道数
        "max_epochs": 50,                      # 训练配置, 最大训练批次数
        "learning_rate": 2e-4,                 # 训练配置, 初始学习率
        "scheduler_t_max": 50,                 # 训练配置, 学习率调度器中的最大迭代次数
        "scheduler_eta_min": 1e-5,             # 训练配置, 学习率的最小值
        "accelerator": "auto",                 # 训练器配置, 训练设备选择
        "devices": 1,                          # 训练器配置, 训练设备数量
        "precision": "16-mixed",               # 训练器配置, 计算精度设置
        "save_dir": "./checkpoints",           # 保存配置, 训练checkpoint保存
        "project_name": "camvid_segmentation", # 保存配置, 模型保存名称
        "export_dir": "./exports",             # 保存配置, 模型导出文件
    }
    
    print("=" * 60)
    print("语义分割模型训练")
    print("=" * 60)
    
    # ==================== 1. 加载数据 ====================
    print("\n[1/6] 加载数据集...")
    train_loader, valid_loader, test_loader, num_classes = get_dataloaders(
        data_dir=CONFIG["data_dir"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"]
    )
    
    CONFIG["out_classes"] = num_classes
    
    print(f"  训练集: {len(train_loader)} batches")
    print(f"  验证集: {len(valid_loader)} batches")
    print(f"  测试集: {len(test_loader)} batches")
    print(f"  类别数: {num_classes}")
    
    # ==================== 2. 创建模型 ====================
    print("\n[2/6] 创建模型...")
    
    # 更新scheduler_t_max
    CONFIG["scheduler_t_max"] = CONFIG["max_epochs"] * len(train_loader)
    
    model = create_model(
        arch=CONFIG["arch"],
        encoder_name=CONFIG["encoder_name"],
        in_channels=CONFIG["in_channels"],
        out_classes=CONFIG["out_classes"],
        learning_rate=CONFIG["learning_rate"],
        scheduler_t_max=CONFIG["scheduler_t_max"],
        scheduler_eta_min=CONFIG["scheduler_eta_min"],
    )
    
    print_model_info(model)
    
    # ==================== 3. 训练模型 ====================
    print("\n[3/6] 开始训练...")
    trainer = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        max_epochs=CONFIG["max_epochs"],
        save_dir=CONFIG["save_dir"],
        project_name=CONFIG["project_name"],
        accelerator=CONFIG["accelerator"],
        devices=CONFIG["devices"],
        precision=CONFIG["precision"],
    )
    
    # ==================== 4. 验证模型 ====================
    print("\n[4/6] 验证最佳模型...")
    valid_metrics = validate_model(model, valid_loader, trainer)
    
    if valid_metrics:
        print("\n验证结果:")
        for key, value in valid_metrics.items():
            print(f"  {key}: {value:.4f}")
    
    # ==================== 5. 测试模型 ====================
    print("\n[5/6] 测试最佳模型...")
    test_metrics = evaluate_metrics(model, test_loader)
    
    # 保存测试结果
    os.makedirs(CONFIG["export_dir"], exist_ok=True)
    test_results_path = os.path.join(CONFIG["export_dir"], "test_results.json")
    save_test_results(test_metrics, test_results_path)
    
    # ==================== 6. 导出模型 ====================
    print("\n[6/6] 导出模型...")
    
    # 导出为目录格式
    model_dir = os.path.join(CONFIG["export_dir"], "saved_model")
    save_model_to_dir(model.model, model_dir, test_metrics)
    print(f"  模型已保存到目录: {model_dir}")
    
    # 导出为ONNX格式
    try:
        onnx_path = os.path.join(CONFIG["export_dir"], "model.onnx")
        save_model_to_onnx(
            model.model,
            onnx_path,
            input_shape=(1, 3, 320, 320),  # 使用训练时的输入大小
            opset_version=11  # 使用更稳定的版本
        )
        print(f"  ✓ ONNX模型已保存到: {onnx_path}")
    except Exception as e:
        print(f"  ⚠ ONNX导出失败: {e}")
        print(f"  提示: ONNX导出失败不影响模型使用, 可以继续使用checkpoint和目录格式的模型")
    
    # ==================== 训练完成 ====================
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"\n最佳checkpoint: {trainer.checkpoint_callback.best_model_path}")
    print(f"导出目录: {CONFIG['export_dir']}")
    print(f"测试IoU: {test_metrics.get('test_iou', 0):.4f}")
    
    return model, test_metrics

if __name__ == "__main__":
    # 设置随机种子
    import pytorch_lightning as pl
    pl.seed_everything(42, workers=True)
    
    # 运行训练
    model, metrics = main()
    
    # 可选：可视化测试结果
    from Script.Test import visualize_test_results
    from Script.Datasets import get_dataloaders
    
    _, _, test_loader, _ = get_dataloaders("./data/CamVid/", batch_size=8)
    print("\n可视化测试结果...")
    visualize_test_results(model, test_loader, num_samples=5)