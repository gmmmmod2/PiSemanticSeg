""" Test.py - 模型测试模块 """
import torch
import pytorch_lightning as pl
from Script.Datasets import visualize_predictions
import json
import os


def test_model(model, test_loader, trainer=None):
    """
    测试模型
    
    Args:
        model: 要测试的模型 (Lightning模块)
        test_loader: 测试数据加载器
        trainer: 训练器
        
    Returns:
        metrics: 测试指标字典
    """
    if trainer is None:
        trainer = pl.Trainer(accelerator="auto", devices=1)
    
    print("开始测试...")
    model.eval()
    
    # 运行测试
    test_metrics = trainer.test(model, dataloaders=test_loader, verbose=True)
    
    if test_metrics:
        print("\n测试结果:")
        for key, value in test_metrics[0].items():
            print(f"  {key}: {value:.4f}")
    
    return test_metrics[0] if test_metrics else None


def test_from_checkpoint(checkpoint_path, test_loader):
    """
    从checkpoint加载模型并测试
    
    Args:
        checkpoint_path: checkpoint文件路径
        test_loader: 测试数据加载器
        
    Returns:
        metrics: 测试指标字典
    """
    from models.SMPmodel import load_model_from_checkpoint
    
    print(f"从 {checkpoint_path} 加载模型...")
    model = load_model_from_checkpoint(checkpoint_path)
    
    return test_model(model, test_loader)


def visualize_test_results(model, test_loader, num_samples=5, device="cuda"):
    """
    可视化测试结果
    
    Args:
        model: 测试模型 (Lightning模块或纯smp模型)
        test_loader: 测试数据加载器
        num_samples: 可视化样本数量
        device: 运行设备
    """
    model.eval()
    model = model.to(device)
    
    # 获取一个batch
    images, masks = next(iter(test_loader))
    images = images.to(device)
    
    # 预测 - 处理Lightning模块和纯模型
    with torch.inference_mode():
        if hasattr(model, 'forward'):
            logits = model(images)
        else:
            logits = model.model(images)
        pred_masks = logits.softmax(dim=1).argmax(dim=1)
    
    # 可视化
    visualize_predictions(images, masks, pred_masks, num_samples)


def evaluate_metrics(model, test_loader, device="cuda"):
    """
    详细评估模型指标
    
    Args:
        model: 测试模型 (Lightning模块)
        test_loader: 测试数据加载器
        device: 运行设备
        
    Returns:
        metrics_dict: 详细指标字典
    """
    import segmentation_models_pytorch as smp
    
    model.eval()
    model = model.to(device)
    
    # 获取模型参数
    if hasattr(model, 'hparams'):
        num_classes = model.hparams.out_classes
    else:
        # 尝试从模型结构推断
        num_classes = 12  # 默认CamVid类别数
    
    # 获取损失函数
    if hasattr(model, 'loss_fn'):
        loss_fn = model.loss_fn
    else:
        loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
    
    all_tp, all_fp, all_fn, all_tn = [], [], [], []
    total_loss = 0
    num_batches = 0
    
    with torch.inference_mode():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device).long()
            
            # 预测 - 使用Lightning模块的forward方法（包含预处理）
            logits = model(images)
            pred_masks = logits.softmax(dim=1).argmax(dim=1)
            
            # 计算损失
            loss = loss_fn(logits.contiguous(), masks)
            total_loss += loss.item()
            num_batches += 1
            
            # 计算统计指标
            tp, fp, fn, tn = smp.metrics.get_stats(
                pred_masks, masks,
                mode="multiclass",
                num_classes=num_classes
            )
            
            all_tp.append(tp)
            all_fp.append(fp)
            all_fn.append(fn)
            all_tn.append(tn)
    
    # 聚合所有批次的指标
    tp = torch.cat(all_tp)
    fp = torch.cat(all_fp)
    fn = torch.cat(all_fn)
    tn = torch.cat(all_tn)
    
    # 计算各种指标
    metrics = {
        "test_loss": total_loss / num_batches,
        "test_iou": smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item(),
        "test_per_image_iou": smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item(),
        "test_f1_score": smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item(),
        "test_accuracy": smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro").item(),
        "test_recall": smp.metrics.recall(tp, fp, fn, tn, reduction="micro").item(),
        "test_precision": smp.metrics.precision(tp, fp, fn, tn, reduction="micro").item(),
    }
    
    print("\n详细测试指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return metrics


def save_test_results(metrics, save_path="test_results.json"):
    """
    保存测试结果到JSON文件
    
    Args:
        metrics: 测试指标字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    
    print(f"\n测试结果已保存到: {save_path}")


def batch_test(checkpoint_paths, test_loader, save_dir="./test_results"):
    """
    批量测试多个checkpoint
    
    Args:
        checkpoint_paths: checkpoint路径列表
        test_loader: 测试数据加载器
        save_dir: 结果保存目录
    """
    from models.SMPmodel import load_model_from_checkpoint
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = {}
    
    for ckpt_path in checkpoint_paths:
        print(f"\n{'='*50}")
        print(f"测试: {ckpt_path}")
        print(f"{'='*50}")
        
        try:
            model = load_model_from_checkpoint(ckpt_path)
            metrics = evaluate_metrics(model, test_loader)
            
            ckpt_name = os.path.basename(ckpt_path)
            all_results[ckpt_name] = metrics
            
            # 保存单个结果
            save_path = os.path.join(save_dir, f"{ckpt_name}_results.json")
            save_test_results(metrics, save_path)
            
        except Exception as e:
            print(f"测试失败: {e}")
            continue
    
    # 保存汇总结果
    summary_path = os.path.join(save_dir, "summary.json")
    save_test_results(all_results, summary_path)
    
    return all_results
