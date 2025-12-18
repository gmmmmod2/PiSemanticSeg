""" Other.py - 工具函数模块 """
import torch
import os
import json


def save_model_to_dir(model, save_dir, metrics=None):
    """
    保存模型到目录格式（使用segmentation_models_pytorch格式）
    
    Args:
        model: smp模型（非Lightning包装）
        save_dir: 保存目录路径
        metrics: 可选的指标字典
    """
    import segmentation_models_pytorch as smp
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(save_dir, metrics=metrics)
    else:
        # 如果是Lightning模型，提取内部模型
        if hasattr(model, 'model'):
            model.model.save_pretrained(save_dir, metrics=metrics)
        else:
            raise ValueError("模型不支持save_pretrained方法")
    
    print(f"模型已保存到: {save_dir}")
    
    # 保存metrics到单独的JSON文件
    if metrics:
        metrics_path = os.path.join(save_dir, "metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print(f"指标已保存到: {metrics_path}")


def save_model_to_onnx(
    model,
    save_path,
    input_shape=(1, 3, 224, 224),
    opset_version=17,
    dynamic_axes=None,
    use_dynamo=False
):
    """
    导出模型为ONNX格式
    
    Args:
        model: PyTorch模型
        save_path: ONNX文件保存路径
        input_shape: 输入形状 (B, C, H, W)
        opset_version: ONNX操作集版本
        dynamic_axes: 动态维度配置
        use_dynamo: 是否使用torch.export（新API，可能不稳定）
    """
    # 提取实际模型（如果是Lightning包装）
    if hasattr(model, 'model'):
        model = model.model
    
    # 将模型移到CPU并设置为评估模式
    model = model.cpu()
    model.eval()
    
    print(f"开始导出ONNX模型...")
    print(f"  输入形状: {input_shape}")
    print(f"  ONNX版本: opset {opset_version}")
    
    # 创建示例输入（CPU上）
    dummy_input = torch.randn(*input_shape)
    
    # 默认动态维度配置
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        }
    
    try:
        # 导出ONNX
        if use_dynamo:
            # 使用新的torch.export API（可能不稳定）
            torch.onnx.export(
                model,
                dummy_input,
                save_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
            )
        else:
            # 使用旧的稳定API
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    dummy_input,
                    save_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes=dynamic_axes,
                    verbose=False,
                    # 使用旧版本导出（更稳定）
                    dynamo=False,
                )
        
        print(f"✓ ONNX模型已保存到: {save_path}")
        
        # 验证ONNX模型
        try:
            import onnx
            onnx_model = onnx.load(save_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX模型验证通过")
            
            # 打印模型信息
            print(f"  输入: {[input.name for input in onnx_model.graph.input]}")
            print(f"  输出: {[output.name for output in onnx_model.graph.output]}")
            
        except ImportError:
            print("⚠ 警告: 未安装onnx库，跳过模型验证")
            print("  可以通过 'pip install onnx' 安装")
        except Exception as e:
            print(f"⚠ ONNX模型验证警告: {e}")
            
    except Exception as e:
        print(f"✗ ONNX导出失败: {e}")
        print("\n可能的解决方案:")
        print("1. 尝试降低opset_version (例如: opset_version=11)")
        print("2. 尝试使用更简单的模型架构")
        print("3. 检查模型中是否有不支持的操作")
        raise

def save_model_to_onnx_simple(model, save_path, input_shape=(1, 3, 320, 320)):
    """
    简化版ONNX导出（更稳定，不支持动态维度）
    
    Args:
        model: PyTorch模型
        save_path: ONNX文件保存路径
        input_shape: 固定输入形状 (B, C, H, W)
    """
    # 提取实际模型
    if hasattr(model, 'model'):
        model = model.model
    
    # 移到CPU并设置为评估模式
    model = model.cpu()
    model.eval()
    
    print(f"简化ONNX导出（固定输入大小）...")
    print(f"  输入形状: {input_shape}")
    
    dummy_input = torch.randn(*input_shape)
    
    try:
        with torch.no_grad():
            # 先测试一次前向传播
            _ = model(dummy_input)
            
            # 使用最简单的导出方式
            torch.onnx.export(
                model,
                dummy_input,
                save_path,
                export_params=True,
                opset_version=11,  # 使用较旧但更稳定的版本
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                verbose=False,
                dynamo=False,
            )
        
        print(f"✓ ONNX模型已保存到: {save_path}")
        
        # 简单验证
        try:
            import onnx
            onnx_model = onnx.load(save_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX模型验证通过")
        except ImportError:
            pass
        except Exception as e:
            print(f"⚠ 验证警告: {e}")
            
        return True
        
    except Exception as e:
        print(f"✗ ONNX导出失败: {e}")
        return False


def load_model_from_dir(model_dir, device="cuda"):
    """
    从目录加载模型
    
    Args:
        model_dir: 模型目录路径
        device: 加载到的设备
        
    Returns:
        model: 加载的模型
    """
    import segmentation_models_pytorch as smp
    
    model = smp.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()
    
    print(f"模型已从 {model_dir} 加载")
    
    return model


def print_model_info(model):
    """
    打印模型信息
    
    Args:
        model: PyTorch模型
    """
    # 提取实际模型
    actual_model = model.model if hasattr(model, 'model') else model
    
    # 计算参数量
    total_params = sum(p.numel() for p in actual_model.parameters())
    trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
    
    print(f"\n模型信息:")
    print(f"  架构: {model.hparams.arch if hasattr(model, 'hparams') else 'Unknown'}")
    print(f"  编码器: {model.hparams.encoder_name if hasattr(model, 'hparams') else 'Unknown'}")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  参数大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")


def count_model_flops(model, input_shape=(1, 3, 224, 224)):
    """
    计算模型FLOPs
    
    Args:
        model: PyTorch模型
        input_shape: 输入形状
        
    Returns:
        flops: FLOPs数量
        params: 参数量
    """
    try:
        from thop import profile
        
        # 提取实际模型
        actual_model = model.model if hasattr(model, 'model') else model
        actual_model.eval()
        
        dummy_input = torch.randn(*input_shape)
        flops, params = profile(actual_model, inputs=(dummy_input,), verbose=False)
        
        print(f"\n模型复杂度:")
        print(f"  FLOPs: {flops / 1e9:.2f} G")
        print(f"  Params: {params / 1e6:.2f} M")
        
        return flops, params
        
    except ImportError:
        print("警告: 未安装thop库，无法计算FLOPs")
        print("可以通过 'pip install thop' 安装")
        return None, None


def save_config(config_dict, save_path):
    """
    保存配置到JSON文件
    
    Args:
        config_dict: 配置字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)
    
    print(f"配置已保存到: {save_path}")


def load_config(config_path):
    """
    从JSON文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config_dict: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    print(f"配置已从 {config_path} 加载")
    
    return config_dict


def compare_models(checkpoint_paths, test_loader):
    """
    比较多个模型的性能
    
    Args:
        checkpoint_paths: checkpoint路径列表
        test_loader: 测试数据加载器
        
    Returns:
        comparison_results: 比较结果字典
    """
    from models.SMPmodel import load_model_from_checkpoint
    from Script.Test import evaluate_metrics
    
    results = {}
    
    print("\n开始模型比较...")
    print("=" * 60)
    
    for i, ckpt_path in enumerate(checkpoint_paths, 1):
        print(f"\n[{i}/{len(checkpoint_paths)}] 测试: {os.path.basename(ckpt_path)}")
        
        try:
            model = load_model_from_checkpoint(ckpt_path)
            metrics = evaluate_metrics(model, test_loader)
            
            results[os.path.basename(ckpt_path)] = metrics
            
        except Exception as e:
            print(f"  失败: {e}")
            continue
    
    # 打印比较结果
    print("\n" + "=" * 60)
    print("模型比较结果")
    print("=" * 60)
    
    for name, metrics in results.items():
        print(f"\n{name}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    return results


def create_experiment_dir(base_dir="./experiments", experiment_name=None):
    """
    创建实验目录
    
    Args:
        base_dir: 实验基础目录
        experiment_name: 实验名称（可选，默认使用时间戳）
        
    Returns:
        experiment_dir: 创建的实验目录路径
    """
    from datetime import datetime
    
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiment_dir = os.path.join(base_dir, experiment_name)
    
    # 创建子目录
    subdirs = ["checkpoints", "logs", "exports", "results"]
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    print(f"实验目录已创建: {experiment_dir}")
    
    return experiment_dir


def log_hyperparameters(hparams, log_path):
    """
    记录超参数
    
    Args:
        hparams: 超参数字典
        log_path: 日志文件路径
    """
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("Hyperparameters\n")
        f.write("=" * 50 + "\n\n")
        for key, value in hparams.items():
            f.write(f"{key}: {value}\n")
    
    print(f"超参数已记录到: {log_path}")


def get_device_info():
    """
    获取设备信息
    
    Returns:
        device_info: 设备信息字典
    """
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        device_info["cuda_device_name"] = torch.cuda.get_device_name(0)
        device_info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print("\n设备信息:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    return device_info


def cleanup_checkpoints(checkpoint_dir, keep_best_n=3):
    """
    清理旧的checkpoint文件, 只保留最好的N个
    
    Args:
        checkpoint_dir: checkpoint目录
        keep_best_n: 保留的最佳checkpoint数量
    """
    import glob
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    
    if len(checkpoints) <= keep_best_n:
        print(f"当前有 {len(checkpoints)} 个checkpoints，无需清理")
        return
    
    # 按修改时间排序
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    
    # 删除旧的checkpoints
    for ckpt in checkpoints[keep_best_n:]:
        try:
            os.remove(ckpt)
            print(f"已删除: {ckpt}")
        except Exception as e:
            print(f"删除失败 {ckpt}: {e}")
    
    print(f"清理完成，保留了 {keep_best_n} 个最新的checkpoints")