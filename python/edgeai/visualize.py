"""
性能数据可视化模块

提供以下可视化功能:
1. ProfilingVisualizer - 逐层性能剖析
   - plot_layer_timeline: 甘特图，每层耗时一目了然
   - plot_flops_breakdown: FLOPs 占比饼图
   - plot_roofline: Roofline Model (计算密集 vs 内存密集)
   - plot_compare: 多实现性能对比柱状图

2. PowerVisualizer - 功耗分析
   - plot_power_curve: 功耗随时间变化
   - plot_perf_power_tradeoff: 性能-功耗权衡散点图

依赖: numpy, matplotlib
"""

import json
import os
from typing import List, Dict, Optional, Union

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ProfilingVisualizer:
    """性能剖析数据可视化"""

    def __init__(self, json_path_or_data: Union[str, list, dict]):
        if isinstance(json_path_or_data, str):
            with open(json_path_or_data, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            # 兼容 {"records": [...]} 和 [...]
            self.records = raw if isinstance(raw, list) else raw.get('records', [])
        elif isinstance(json_path_or_data, list):
            self.records = json_path_or_data
        elif isinstance(json_path_or_data, dict):
            self.records = json_path_or_data.get('records', [])
        else:
            self.records = []

    def plot_layer_timeline(self, save_path: Optional[str] = None):
        """
        绘制逐层耗时时间线 (甘特图)
        每个算子是一个水平条，长度代表耗时
        """
        if not self.records:
            print("无 profiling 数据")
            return

        fig, ax = plt.subplots(figsize=(12, max(4, len(self.records) * 0.35)))

        names = [r.get('name', f'op{i}') for i, r in enumerate(self.records)]
        times = [r.get('timeMs', 0) for r in self.records]
        devices = [r.get('device', 'CPU') for r in self.records]

        # 累计偏移，展示流水线
        cumulative = 0
        colors = {'CPU': '#4C72B0', 'GPU': '#DD8452'}
        for i, (name, t, dev) in enumerate(zip(names, times, devices)):
            color = colors.get(dev, '#55A868')
            ax.barh(i, t, left=cumulative, height=0.6, color=color, edgecolor='white')
            # 在条内标注时间
            if t > max(times) * 0.05:
                ax.text(cumulative + t / 2, i, f'{t:.2f}ms',
                        ha='center', va='center', fontsize=7, color='white')
            cumulative += t

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel('Time (ms)')
        ax.set_title('Layer Timeline (甘特图)')
        ax.invert_yaxis()

        # 图例
        legend_patches = [mpatches.Patch(color=c, label=l) for l, c in colors.items()]
        ax.legend(handles=legend_patches, loc='lower right')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"已保存: {save_path}")
        else:
            plt.savefig('layer_timeline.png', dpi=150)
            print("已保存: layer_timeline.png")
        plt.close()

    def plot_flops_breakdown(self, save_path: Optional[str] = None):
        """绘制各算子类型的 FLOPs 占比饼图"""
        if not self.records:
            print("无 profiling 数据")
            return

        # 按 type 聚合 FLOPs
        type_flops: Dict[str, float] = {}
        for r in self.records:
            t = r.get('type', 'Unknown')
            f = r.get('flops', 0)
            type_flops[t] = type_flops.get(t, 0) + f

        labels = list(type_flops.keys())
        sizes = [type_flops[l] / 1e9 for l in labels]  # GFLOPs

        if sum(sizes) == 0:
            print("FLOPs 数据为 0，无法绘制")
            return

        fig, ax = plt.subplots(figsize=(8, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90
        )
        for t in autotexts:
            t.set_fontsize(9)
        ax.set_title('FLOPs Breakdown by Operator Type')

        plt.tight_layout()
        path = save_path or 'flops_breakdown.png'
        plt.savefig(path, dpi=150)
        print(f"已保存: {path}")
        plt.close()

    def plot_roofline(self, peak_gflops: float = 500.0,
                      peak_bw_gbs: float = 40.0,
                      save_path: Optional[str] = None):
        """
        绘制 Roofline Model

        X轴: Arithmetic Intensity (FLOPs/Byte) — 对数刻度
        Y轴: GFLOPS — 对数刻度
        两条线:
          - 计算屋顶: y = peak_gflops (水平线)
          - 内存屋顶: y = peak_bw * x (斜线)
        每个算子是散点，位置反映其计算密度和实际性能
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        # Roofline 边界
        ai_range = np.logspace(-1, 3, 200)
        mem_roof = peak_bw_gbs * ai_range
        compute_roof = np.full_like(ai_range, peak_gflops)
        attainable = np.minimum(mem_roof, compute_roof)

        ax.loglog(ai_range, attainable, 'k-', linewidth=2, label='Roofline')
        ax.loglog(ai_range, mem_roof, 'b--', alpha=0.5, linewidth=1, label=f'Mem BW: {peak_bw_gbs} GB/s')
        ax.loglog(ai_range, compute_roof, 'r--', alpha=0.5, linewidth=1, label=f'Peak: {peak_gflops} GFLOPS')

        # 标记拐点
        ridge = peak_gflops / peak_bw_gbs
        ax.axvline(x=ridge, color='gray', linestyle=':', alpha=0.5)
        ax.text(ridge, peak_gflops * 1.3, f'Ridge={ridge:.1f}', ha='center', fontsize=9)

        # 绘制各算子点
        if self.records:
            for r in self.records:
                ai = r.get('arithmeticIntensity', 0)
                gf = r.get('gflops', 0)
                name = r.get('name', '?')
                dev = r.get('device', 'CPU')
                if ai > 0 and gf > 0:
                    color = '#DD8452' if dev == 'GPU' else '#4C72B0'
                    ax.scatter(ai, gf, c=color, s=80, zorder=5, edgecolors='white')
                    ax.annotate(name, (ai, gf), textcoords='offset points',
                               xytext=(5, 5), fontsize=7, alpha=0.8)

        # 标注区域
        ax.fill_between(ai_range, 0, np.minimum(mem_roof, compute_roof),
                         alpha=0.05, color='green')
        ax.text(0.5, peak_gflops * 0.01, 'Memory-Bound\n(内存密集)', fontsize=11, alpha=0.5)
        ax.text(200, peak_gflops * 0.01, 'Compute-Bound\n(计算密集)', fontsize=11, alpha=0.5)

        ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)')
        ax.set_ylabel('GFLOPS')
        ax.set_title('Roofline Model')
        ax.set_xlim(0.1, 1000)
        ax.set_ylim(0.1, peak_gflops * 10)
        ax.legend(loc='upper left')
        ax.grid(True, which='both', alpha=0.3)

        plt.tight_layout()
        path = save_path or 'roofline.png'
        plt.savefig(path, dpi=150)
        print(f"已保存: {path}")
        plt.close()

    def plot_compare(self, labels: List[str], times_ms: List[float],
                     title: str = 'Operator Performance Comparison',
                     save_path: Optional[str] = None):
        """
        对比不同实现的性能柱状图

        labels = ["Naive", "AVX2", "Tiled", "Winograd", "Vulkan"]
        times_ms = [2345, 523, 412, 289, 156]
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(labels))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))

        # 左图: 耗时
        bars = ax1.bar(x, times_ms, color=colors, edgecolor='white')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=15)
        ax1.set_ylabel('Time (ms)')
        ax1.set_title(f'{title} - Latency')
        for bar, t in zip(bars, times_ms):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times_ms) * 0.01,
                     f'{t:.1f}', ha='center', va='bottom', fontsize=9)

        # 右图: 加速比 (相对最慢的实现)
        baseline = max(times_ms)
        speedups = [baseline / t if t > 0 else 0 for t in times_ms]
        bars2 = ax2.bar(x, speedups, color=colors, edgecolor='white')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=15)
        ax2.set_ylabel('Speedup (x)')
        ax2.set_title(f'{title} - Speedup vs Baseline')
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        for bar, s in zip(bars2, speedups):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f'{s:.1f}x', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        path = save_path or 'compare_ops.png'
        plt.savefig(path, dpi=150)
        print(f"已保存: {path}")
        plt.close()


class PowerVisualizer:
    """功耗数据可视化"""

    def __init__(self, json_path_or_data: Union[str, list, dict]):
        if isinstance(json_path_or_data, str):
            with open(json_path_or_data, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            self.samples = raw if isinstance(raw, list) else raw.get('samples', [])
        elif isinstance(json_path_or_data, list):
            self.samples = json_path_or_data
        elif isinstance(json_path_or_data, dict):
            self.samples = json_path_or_data.get('samples', [])
        else:
            self.samples = []

    def plot_power_curve(self, save_path: Optional[str] = None):
        """绘制功耗随时间变化曲线"""
        if not self.samples:
            print("无功耗数据")
            return

        timestamps = [s.get('timestamp', 0) for s in self.samples]
        cpu_power = [s.get('cpuPowerW', 0) for s in self.samples]
        gpu_power = [s.get('gpuPowerW', 0) for s in self.samples]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(timestamps, cpu_power, 'b-', label='CPU Power', linewidth=1.5)
        if any(p > 0 for p in gpu_power):
            ax.plot(timestamps, gpu_power, 'r-', label='GPU Power', linewidth=1.5)
        ax.fill_between(timestamps, cpu_power, alpha=0.2, color='blue')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power (W)')
        ax.set_title('Power Consumption Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 标注统计
        avg_cpu = np.mean(cpu_power)
        ax.axhline(y=avg_cpu, color='blue', linestyle='--', alpha=0.5)
        ax.text(timestamps[-1] * 0.02, avg_cpu * 1.05,
                f'Avg: {avg_cpu:.1f}W', fontsize=9, color='blue')

        plt.tight_layout()
        path = save_path or 'power_curve.png'
        plt.savefig(path, dpi=150)
        print(f"已保存: {path}")
        plt.close()

    def plot_perf_power_tradeoff(self, configs: List[str],
                                  gflops_list: List[float],
                                  power_list: List[float],
                                  save_path: Optional[str] = None):
        """
        绘制性能-功耗权衡散点图

        configs: 配置名称列表 (如 ["1T-2.0GHz", "2T-2.5GHz", ...])
        gflops_list: 对应的 GFLOPS
        power_list: 对应的功耗 (W)
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(power_list, gflops_list, c=range(len(configs)),
                             cmap='viridis', s=100, edgecolors='white', zorder=5)

        for i, cfg in enumerate(configs):
            ax.annotate(cfg, (power_list[i], gflops_list[i]),
                        textcoords='offset points', xytext=(8, 5), fontsize=8)

        # 计算并标注能效比
        for i in range(len(configs)):
            eff = gflops_list[i] / power_list[i] if power_list[i] > 0 else 0
            ax.annotate(f'{eff:.2f} GF/W', (power_list[i], gflops_list[i]),
                        textcoords='offset points', xytext=(8, -12), fontsize=7, color='gray')

        # Pareto 前沿线
        sorted_idx = np.argsort(power_list)
        pareto_x = [power_list[i] for i in sorted_idx]
        pareto_y = [gflops_list[i] for i in sorted_idx]
        ax.plot(pareto_x, pareto_y, 'k--', alpha=0.3, linewidth=1)

        ax.set_xlabel('Power (W)')
        ax.set_ylabel('Performance (GFLOPS)')
        ax.set_title('Performance-Power Tradeoff')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = save_path or 'perf_power_tradeoff.png'
        plt.savefig(path, dpi=150)
        print(f"已保存: {path}")
        plt.close()
