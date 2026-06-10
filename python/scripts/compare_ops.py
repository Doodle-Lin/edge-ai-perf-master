#!/usr/bin/env python3
"""
算子性能对比图

用法:
    python compare_ops.py --ops naive avx2 tiled winograd vulkan --sizes 128 256
    python compare_ops.py --data results.json
    python compare_ops.py --ops naive avx2 --times 2345 523 --output compare.png
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from edgeai.visualize import ProfilingVisualizer


def main():
    parser = argparse.ArgumentParser(description='算子性能对比')
    parser.add_argument('--ops', nargs='+', help='实现名称列表')
    parser.add_argument('--times', nargs='+', type=float, help='耗时列表 (ms)')
    parser.add_argument('--sizes', nargs='+', type=int, help='问题规模 (用于生成模拟数据)')
    parser.add_argument('--data', help='JSON 结果文件路径')
    parser.add_argument('--output', '-o', default='compare_ops.png', help='输出文件路径')
    args = parser.parse_args()

    if args.data:
        # 从 JSON 文件加载
        with open(args.data, 'r') as f:
            data = json.load(f)
        labels = [d['name'] for d in data]
        times = [d['timeMs'] for d in data]
    elif args.ops and args.times:
        labels = args.ops
        times = args.times
    elif args.ops and args.sizes:
        # 生成模拟数据用于演示
        import numpy as np
        labels = args.ops
        # 模拟不同实现的加速比
        speedups = {
            'naive': 1.0,
            'avx2': 6.0,
            'tiled': 8.0,
            'winograd': 12.0,
            'vulkan': 15.0,
        }
        times = []
        for op in labels:
            base_time = 2000  # ms for naive at size 128
            su = speedups.get(op.lower(), 1.0)
            times.append(base_time / su)
    else:
        print("请提供 --ops 和 --times，或 --data 参数")
        parser.print_help()
        sys.exit(1)

    viz = ProfilingVisualizer([])
    viz.plot_compare(labels, times, save_path=args.output)


if __name__ == '__main__':
    main()
