#!/usr/bin/env python3
"""
下载示例 ONNX 模型并转换为 NCNN 格式

用法:
    python download_models.py [--output-dir ../models]

依赖:
    pip install onnx onnxruntime

本脚本会:
1. 下载 MobileNetV2 ONNX 模型 (轻量级，适合端侧)
2. 使用 onnx2ncnn 转换为 NCNN 格式 (.param + .bin)
3. 生成验证用的随机输入
"""

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


MODELS = {
    "mobilenetv2": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
        "input_shape": [1, 3, 224, 224],
        "description": "MobileNetV2 - 轻量级图像分类模型，适合端侧部署"
    },
    "squeezenet": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx",
        "input_shape": [1, 3, 224, 224],
        "description": "SqueezeNet 1.1 - 更小的分类模型"
    },
}


def download_file(url: str, dest: Path):
    """下载文件，带进度显示"""
    print(f"  下载: {url}")
    print(f"  保存: {dest}")

    def report_hook(count, block_size, total_size):
        percent = min(count * block_size * 100 / total_size, 100) if total_size > 0 else 0
        sys.stdout.write(f"\r  进度: {percent:.1f}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, str(dest), reporthook=report_hook)
    print()


def convert_onnx_to_ncnn(onnx_path: Path, output_dir: Path):
    """使用 onnx2ncnn 转换模型"""
    param_path = output_dir / onnx_path.with_suffix(".param").name
    bin_path = output_dir / onnx_path.with_suffix(".bin").name

    # 尝试找到 onnx2ncnn
    onnx2ncnn = os.environ.get("ONNX2NCNN", "onnx2ncnn")

    try:
        result = subprocess.run(
            [onnx2ncnn, str(onnx_path), str(param_path), str(bin_path)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  转换成功: {param_path.name}, {bin_path.name}")
            return True
        else:
            print(f"  转换失败: {result.stderr}")
            return False
    except FileNotFoundError:
        print(f"  未找到 onnx2ncnn，请安装 NCNN 并设置 ONNX2NCNN 环境变量")
        print(f"  或者手动转换: onnx2ncnn {onnx_path} {param_path} {bin_path}")
        return False


def generate_test_input(output_dir: Path, shape: list):
    """生成测试用的随机输入数据"""
    try:
        import numpy as np
        data = np.random.randn(*shape).astype(np.float32)
        input_path = output_dir / "test_input.bin"
        data.tofile(str(input_path))
        print(f"  测试输入已保存: {input_path} (shape={shape})")
    except ImportError:
        print("  numpy 未安装，跳过测试输入生成")


def main():
    parser = argparse.ArgumentParser(description="下载示例模型并转换为 NCNN 格式")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认: 脚本所在目录的 ../models)")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()),
                        help="要下载的模型列表")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent.parent / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        info = MODELS[model_name]
        print(f"\n{'='*60}")
        print(f"模型: {model_name}")
        print(f"说明: {info['description']}")
        print(f"{'='*60}")

        onnx_path = output_dir / f"{model_name}.onnx"

        # 下载
        if onnx_path.exists():
            print(f"  已存在: {onnx_path}，跳过下载")
        else:
            download_file(info["url"], onnx_path)

        # 转换
        convert_onnx_to_ncnn(onnx_path, output_dir)

        # 生成测试输入
        generate_test_input(output_dir, info["input_shape"])

    print(f"\n完成! 模型保存在: {output_dir}")


if __name__ == "__main__":
    main()
