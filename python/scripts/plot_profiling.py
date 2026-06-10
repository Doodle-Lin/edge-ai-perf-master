#!/usr/bin/env python3
"""
绘制 profiling 结果

用法:
    python plot_profiling.py profiling_result.json
    python plot_profiling.py profiling_result.json --type timeline
    python plot_profiling.py profiling_result.json --type breakdown
    python plot_profiling.py profiling_result.json --type roofline
    python plot_profiling.py profiling_result.json --output result.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from edgeai.visualize import ProfilingVisualizer


def main():
    parser = argparse.ArgumentParser(description='绘制 profiling 结果')
    parser.add_argument('json_path', help='profiling JSON 文件路径')
    parser.add_argument('--type', choices=['timeline', 'breakdown', 'roofline', 'all'],
                        default='all', help='图表类型 (默认: all)')
    parser.add_argument('--output', '-o', help='输出文件路径 (默认: 自动命名)')
    parser.add_argument('--peak-gflops', type=float, default=500.0,
                        help='峰值算力 GFLOPS (默认: 500)')
    parser.add_argument('--peak-bw', type=float, default=40.0,
                        help='峰值内存带宽 GB/s (默认: 40)')
    args = parser.parse_args()

    viz = ProfilingVisualizer(args.json_path)

    base = Path(args.json_path).stem

    if args.type in ('timeline', 'all'):
        out = args.output or f'{base}_timeline.png'
        viz.plot_layer_timeline(save_path=out)

    if args.type in ('breakdown', 'all'):
        out = args.output or f'{base}_breakdown.png'
        viz.plot_flops_breakdown(save_path=out)

    if args.type in ('roofline', 'all'):
        out = args.output or f'{base}_roofline.png'
        viz.plot_roofline(peak_gflops=args.peak_gflops, peak_bw_gbs=args.peak_bw,
                          save_path=out)


if __name__ == '__main__':
    main()
