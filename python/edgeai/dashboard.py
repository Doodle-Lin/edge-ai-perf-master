"""
Streamlit 仪表盘 - 交互式性能分析

运行方式:
    streamlit run dashboard.py

功能:
    - 性能剖析: 逐层耗时甘特图 + FLOPs 占比
    - 算子对比: 不同优化实现的性能对比
    - 功耗分析: 功耗曲线和能效指标
    - Roofline Model: 计算密集 vs 内存密集分析
"""

import streamlit as st
import json
import os
import sys
from pathlib import Path

# 确保可以导入 edgeai 包
sys.path.insert(0, str(Path(__file__).parent.parent))

from edgeai.visualize import ProfilingVisualizer, PowerVisualizer
from edgeai.power_analyzer import PowerAnalyzer


def load_json_file(uploaded_file):
    """从上传的文件加载 JSON"""
    if uploaded_file is not None:
        return json.loads(uploaded_file.read().decode('utf-8'))
    return None


def main():
    st.set_page_config(page_title="EdgeAI Profiler", layout="wide")
    st.title("EdgeAI 性能分析仪表盘")

    # 侧边栏
    mode = st.sidebar.selectbox(
        "分析模式",
        ["性能剖析", "算子对比", "功耗分析", "Roofline Model"]
    )

    # 峰值参数设置
    st.sidebar.markdown("### 硬件参数")
    peak_gflops = st.sidebar.number_input("峰值算力 (GFLOPS)", value=500.0, min_value=1.0)
    peak_bw = st.sidebar.number_input("峰值内存带宽 (GB/s)", value=40.0, min_value=1.0)

    if mode == "性能剖析":
        st.header("性能剖析")
        st.markdown("上传 Lab1 Step3 生成的 profiling JSON 文件")

        uploaded = st.file_uploader("选择 JSON 文件", type=['json'])
        if uploaded:
            data = load_json_file(uploaded)
            if data:
                viz = ProfilingVisualizer(data)

                # 逐层耗时表
                st.subheader("逐层耗时")
                records = data if isinstance(data, list) else data.get('records', [])
                if records:
                    import pandas as pd
                    df = pd.DataFrame(records)
                    display_cols = [c for c in ['name', 'type', 'timeMs', 'flops', 'memBytes', 'device'] if c in df.columns]
                    if display_cols:
                        st.dataframe(df[display_cols], use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("时间线")
                    viz.plot_layer_timeline(save_path='/tmp/edgeai_timeline.png')
                    if os.path.exists('/tmp/edgeai_timeline.png'):
                        st.image('/tmp/edgeai_timeline.png')

                with col2:
                    st.subheader("FLOPs 分布")
                    viz.plot_flops_breakdown(save_path='/tmp/edgeai_flops.png')
                    if os.path.exists('/tmp/edgeai_flops.png'):
                        st.image('/tmp/edgeai_flops.png')

    elif mode == "算子对比":
        st.header("算子性能对比")
        st.markdown("输入各实现的耗时 (ms)，自动生成对比图")

        impl_names = st.text_input("实现名称 (逗号分隔)", value="Naive,AVX2,Tiled,Winograd,Vulkan")
        times_str = st.text_input("耗时 (ms, 逗号分隔)", value="2345,523,412,289,156")

        if st.button("生成对比图"):
            names = [n.strip() for n in impl_names.split(',')]
            times = [float(t.strip()) for t in times_str.split(',') if t.strip()]

            if len(names) == len(times):
                viz = ProfilingVisualizer([])
                viz.plot_compare(names, times, save_path='/tmp/edgeai_compare.png')
                if os.path.exists('/tmp/edgeai_compare.png'):
                    st.image('/tmp/edgeai_compare.png')
            else:
                st.error("名称数量和耗时数量不匹配")

    elif mode == "功耗分析":
        st.header("功耗分析")
        uploaded = st.file_uploader("上传功耗 JSON 文件", type=['json'])
        if uploaded:
            data = load_json_file(uploaded)
            if data:
                analyzer = PowerAnalyzer(data)
                analyzer.print_summary()

                pviz = PowerVisualizer(data)
                pviz.plot_power_curve(save_path='/tmp/edgeai_power.png')
                if os.path.exists('/tmp/edgeai_power.png'):
                    st.image('/tmp/edgeai_power.png')

    elif mode == "Roofline Model":
        st.header("Roofline Model")
        st.markdown("""
        Roofline Model 可视化:
        - **斜线区域**: 内存密集 (Memory-Bound)，受内存带宽限制
        - **水平线区域**: 计算密集 (Compute-Bound)，受峰值算力限制
        - **拐点 (Ridge)**: 内存密集与计算密集的分界线
        """)

        uploaded = st.file_uploader("上传 profiling JSON", type=['json'])
        if uploaded:
            data = load_json_file(uploaded)
            if data:
                viz = ProfilingVisualizer(data)
                viz.plot_roofline(peak_gflops=peak_gflops, peak_bw_gbs=peak_bw,
                                  save_path='/tmp/edgeai_roofline.png')
                if os.path.exists('/tmp/edgeai_roofline.png'):
                    st.image('/tmp/edgeai_roofline.png')


if __name__ == "__main__":
    main()
