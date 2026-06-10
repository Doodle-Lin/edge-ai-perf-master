"""
功耗数据分析工具

提供:
- summary(): 摘要统计
- efficiency_metrics(): 能效指标 (GFLOPS/W)
- find_optimal_config(): 在多配置中找最优能效
"""

import json
from typing import List, Dict, Optional, Union


class PowerAnalyzer:
    """功耗数据分析"""

    def __init__(self, json_path_or_data: Union[str, list, dict]):
        if isinstance(json_path_or_data, str):
            with open(json_path_or_data, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            self.samples = raw if isinstance(raw, list) else raw.get('samples', [])
            self.metadata = raw if isinstance(raw, dict) else {}
        elif isinstance(json_path_or_data, list):
            self.samples = json_path_or_data
            self.metadata = {}
        elif isinstance(json_path_or_data, dict):
            self.samples = json_path_or_data.get('samples', [])
            self.metadata = json_path_or_data
        else:
            self.samples = []
            self.metadata = {}

    def summary(self) -> Dict:
        """返回摘要统计"""
        if not self.samples:
            return {"error": "无功耗数据"}

        cpu_powers = [s.get('cpuPowerW', 0) for s in self.samples]
        gpu_powers = [s.get('gpuPowerW', 0) for s in self.samples]

        import numpy as np

        result = {
            "avgCpuPowerW": float(np.mean(cpu_powers)),
            "peakCpuPowerW": float(np.max(cpu_powers)),
            "minCpuPowerW": float(np.min(cpu_powers)),
            "stdCpuPowerW": float(np.std(cpu_powers)),
            "avgGpuPowerW": float(np.mean(gpu_powers)) if any(p > 0 for p in gpu_powers) else 0.0,
            "peakGpuPowerW": float(np.max(gpu_powers)) if any(p > 0 for p in gpu_powers) else 0.0,
        }

        # 计算总能耗 (梯形积分)
        timestamps = [s.get('timestamp', 0) for s in self.samples]
        total_energy = 0.0
        for i in range(1, len(timestamps)):
            dt = timestamps[i] - timestamps[i - 1]
            avg_p = (cpu_powers[i] + gpu_powers[i] + cpu_powers[i-1] + gpu_powers[i-1]) / 2.0
            total_energy += avg_p * dt
        result["totalEnergyJ"] = total_energy
        result["durationS"] = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0

        return result

    def efficiency_metrics(self, gflops: float) -> Dict:
        """计算能效指标"""
        s = self.summary()
        if 'error' in s:
            return s

        avg_power = s['avgCpuPowerW'] + s['avgGpuPowerW']
        if avg_power <= 0:
            return {"error": "功耗数据无效"}

        return {
            "gflopsPerWatt": gflops / avg_power,
            "avgPowerW": avg_power,
            "totalEnergyJ": s.get('totalEnergyJ', 0),
            "joulesPerInference": s.get('totalEnergyJ', 0),  # 如果只跑了一次推理
        }

    def find_optimal_config(self, configs_data: List[Dict]) -> Dict:
        """
        在多个配置中找到能效比最高的

        configs_data: [
            {"name": "1T-2.0GHz", "gflops": 12.5, "avgPowerW": 15.2},
            {"name": "2T-2.5GHz", "gflops": 22.1, "avgPowerW": 28.7},
            ...
        ]
        """
        best = None
        best_eff = 0

        for cfg in configs_data:
            p = cfg.get('avgPowerW', 0)
            g = cfg.get('gflops', 0)
            if p <= 0:
                continue
            eff = g / p
            cfg['gflopsPerWatt'] = eff
            if eff > best_eff:
                best_eff = eff
                best = cfg

        return {
            "optimal": best,
            "all_configs": configs_data,
            "recommendation": f"最优配置: {best['name']} ({best_eff:.2f} GF/W)" if best else "无有效配置"
        }

    def print_summary(self):
        """打印摘要"""
        s = self.summary()
        if 'error' in s:
            print(s['error'])
            return

        print("=" * 50)
        print("功耗分析摘要")
        print("=" * 50)
        print(f"  平均 CPU 功耗: {s['avgCpuPowerW']:.1f} W")
        print(f"  峰值 CPU 功耗: {s['peakCpuPowerW']:.1f} W")
        if s['avgGpuPowerW'] > 0:
            print(f"  平均 GPU 功耗: {s['avgGpuPowerW']:.1f} W")
            print(f"  峰值 GPU 功耗: {s['peakGpuPowerW']:.1f} W")
        print(f"  总能耗: {s['totalEnergyJ']:.1f} J")
        print(f"  持续时间: {s['durationS']:.2f} s")
        print("=" * 50)
