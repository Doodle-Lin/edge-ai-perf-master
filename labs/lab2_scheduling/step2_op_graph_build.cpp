/**
 * @file step2_op_graph_build.cpp
 * @brief Lab2 Step2: 算子依赖图构建 —— 将神经网络表示为 DAG，分析依赖与并行性
 *
 * ====== 本实验学习目标 ======
 * 1. 理解为什么用 DAG 表示神经网络推理过程
 * 2. 学会构建算子图：添加节点、添加依赖边
 * 3. 掌握拓扑排序算法（Kahn 算法）
 * 4. 学会分析并行性：找出可并行执行的算子对
 * 5. 计算关键路径：决定推理延迟下界
 * 6. 导出 DOT 格式用于 Graphviz 可视化
 *
 * 编译运行:
 *   g++ -std=c++17 -O2 -o step2_op_graph_build step2_op_graph_build.cpp
 *   ./step2_op_graph_build
 */

#include <algorithm>
#include <functional>
#include <iostream>
#include <iomanip>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ============================================================================
// 算子节点（与框架 include/scheduler/OpGraph.h 中的 OpNode 对齐）
// ============================================================================

struct OpNode {
    int id = -1;                          ///< 唯一标识符
    std::string name;                      ///< 算子实例名，如 "conv1"
    std::string type;                      ///< 算子类型，如 "Convolution"
    std::vector<int> inputs;               ///< 依赖的前驱算子 id 列表
    std::vector<int> outputs;              ///< 被哪些算子依赖
    int64_t estimatedFlops = 0;            ///< 估算 FLOPs
    int64_t estimatedMemBytes = 0;         ///< 估算内存访问量
};

// ============================================================================
// 算子依赖图（与框架 include/scheduler/OpGraph.h 对齐）
// ============================================================================

class OpGraph {
public:
    /// 添加算子节点，返回节点 id
    int addOp(const std::string& name, const std::string& type,
              const std::vector<int>& inputs = {},
              int64_t flops = 0, int64_t memBytes = 0) {
        int id = static_cast<int>(ops_.size());
        ops_.push_back({id, name, type, inputs, {}, flops, memBytes});
        // 建立反向边：前驱节点的 outputs 记录当前节点
        for (int dep : inputs) {
            ops_[dep].outputs.push_back(id);
        }
        return id;
    }

    const OpNode& getOp(int id) const { return ops_.at(id); }
    const std::vector<OpNode>& getAllOps() const { return ops_; }
    int numOps() const { return static_cast<int>(ops_.size()); }

    // ──────────────── 拓扑排序（Kahn 算法） ────────────────
    /**
     * Kahn 算法步骤：
     * 1. 计算每个节点的入度（被多少节点依赖）
     * 2. 入度为 0 的节点入队（没有前驱，可以最先执行）
     * 3. 出队一个节点，将其后继的入度减 1
     * 4. 后继入度变 0 则入队
     * 5. 重复直到队列为空
     *
     * 时间复杂度 O(V+E)，比 DFS 式更稳定（不会栈溢出）。
     * 如果输出序列长度 < V，说明图中有环——推理图中不应该出现环。
     */
    std::vector<int> topologicalSort() const {
        int n = numOps();
        std::vector<int> inDegree(n, 0);
        for (const auto& op : ops_) {
            for (int dep : op.inputs) {
                (void)dep;
                // op 依赖 dep，所以 op 的入度 = len(inputs)
                // 但我们在构建时已经记录了 inputs，直接统计
            }
            inDegree[op.id] = static_cast<int>(op.inputs.size());
        }

        std::queue<int> q;
        for (int i = 0; i < n; ++i) {
            if (inDegree[i] == 0) q.push(i);
        }

        std::vector<int> order;
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            order.push_back(u);
            for (int v : ops_[u].outputs) {
                if (--inDegree[v] == 0) {
                    q.push(v);
                }
            }
        }

        if (static_cast<int>(order.size()) < n) {
            std::cerr << "[警告] 图中存在环！拓扑排序不完整。\n";
        }
        return order;
    }

    // ──────────────── 并行性分析 ────────────────
    /**
     * 找出所有可并行执行的算子对。
     * 判定条件：两个算子之间不存在任何路径（既不是祖先也不是后代）。
     * 实现方法：对每对算子 (a, b)，检查 a 不可达 b 且 b 不可达 a。
     */
    std::vector<std::pair<int,int>> findParallelOps() const {
        std::vector<std::pair<int,int>> parallel;
        int n = numOps();
        for (int a = 0; a < n; ++a) {
            for (int b = a + 1; b < n; ++b) {
                if (!isReachable(a, b) && !isReachable(b, a)) {
                    parallel.push_back({a, b});
                }
            }
        }
        return parallel;
    }

    // ──────────────── 关键路径分析 ────────────────
    /**
     * 关键路径 = 图中的最长路径（按 FLOPs 加权）。
     * 它决定了推理延迟的理论下界：即使有无限并行资源，
     * 延迟也不可能低于关键路径长度。
     *
     * 算法：动态规划
     *   dist[v] = max(dist[u] + weight(u)) 对所有 u → v
     * 然后回溯得到完整路径。
     */
    std::vector<int> criticalPath() const {
        int n = numOps();
        auto order = topologicalSort();
        // dist[v] = 从源点到 v 的最长路径（以 FLOPs 加权）
        std::vector<double> dist(n, 0.0);
        std::vector<int> prev(n, -1);

        for (int u : order) {
            double w = static_cast<double>(ops_[u].estimatedFlops);
            for (int v : ops_[u].outputs) {
                if (dist[u] + w > dist[v]) {
                    dist[v] = dist[u] + w;
                    prev[v] = u;
                }
            }
        }

        // 找终点（dist 最大的节点）
        int endNode = 0;
        for (int i = 1; i < n; ++i) {
            double totalV = dist[i] + static_cast<double>(ops_[i].estimatedFlops);
            double totalEnd = dist[endNode] + static_cast<double>(ops_[endNode].estimatedFlops);
            if (totalV > totalEnd) {
                endNode = i;
            }
        }

        // 回溯路径
        std::vector<int> path;
        int cur = endNode;
        while (cur != -1) {
            path.push_back(cur);
            cur = prev[cur];
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    // ──────────────── DOT 格式导出 ────────────────
    /**
     * 导出为 Graphviz DOT 格式。
     * 用法：
     *   std::ofstream out("graph.dot");
     *   out << graph.toDot();
     *   // 命令行渲染: dot -Tpng graph.dot -o graph.png
     */
    std::string toDot() const {
        std::ostringstream oss;
        oss << "digraph OpGraph {\n";
        oss << "    rankdir=LR;\n";  // 从左到右布局，比从上到下更适合 CNN 图
        oss << "    node [shape=box, style=\"rounded,filled\", fontname=\"Arial\"];\n";
        oss << "    edge [color=\"#666666\"];\n\n";

        for (const auto& op : ops_) {
            // 根据算子类型着色
            std::string color;
            if (op.type == "Convolution") color = "#4A90D9";      // 蓝色 - 计算密集
            else if (op.type == "ReLU") color = "#7ED321";        // 绿色 - 逐元素
            else if (op.type == "Pooling") color = "#F5A623";     // 橙色 - 内存密集
            else if (op.type == "FullyConnected") color = "#D0021B"; // 红色 - 参数密集
            else color = "#9B9B9B";                                // 灰色 - 其他

            // FLOPs 用人类可读的格式显示
            std::string flopsStr;
            if (op.estimatedFlops >= 1'000'000'000)
                flopsStr = std::to_string(op.estimatedFlops / 1'000'000'000) + "G";
            else if (op.estimatedFlops >= 1'000'000)
                flopsStr = std::to_string(op.estimatedFlops / 1'000'000) + "M";
            else if (op.estimatedFlops >= 1'000)
                flopsStr = std::to_string(op.estimatedFlops / 1'000) + "K";
            else
                flopsStr = std::to_string(op.estimatedFlops);

            oss << "    op" << op.id << " [label=\""
                << op.name << "\\n" << op.type << "\\n"
                << flopsStr << " FLOPs\", fillcolor=\"" << color << "\"];\n";
        }

        oss << "\n";
        for (const auto& op : ops_) {
            for (int dep : op.inputs) {
                oss << "    op" << dep << " -> op" << op.id << ";\n";
            }
        }

        oss << "}\n";
        return oss.str();
    }

    // ──────────────── ASCII 可视化 ────────────────
    void printAsciiArt() const {
        std::cout << "\n  算子依赖图 (ASCII Art):\n\n";
        auto order = topologicalSort();

        // 按拓扑层级排列
        std::vector<int> inDegree(numOps(), 0);
        for (const auto& op : ops_) {
            inDegree[op.id] = static_cast<int>(op.inputs.size());
        }

        // 计算每个节点的"层级"（最长前驱路径 + 1）
        std::vector<int> level(numOps(), 0);
        for (int u : order) {
            for (int v : ops_[u].outputs) {
                level[v] = std::max(level[v], level[u] + 1);
            }
        }

        int maxLevel = 0;
        for (int l : level) maxLevel = std::max(maxLevel, l);

        // 按层级分组
        std::vector<std::vector<int>> stages(maxLevel + 1);
        for (int i = 0; i < numOps(); ++i) {
            stages[level[i]].push_back(i);
        }

        // 打印
        for (int s = 0; s <= maxLevel; ++s) {
            std::cout << "  Stage " << s << ":  ";
            for (size_t i = 0; i < stages[s].size(); ++i) {
                int id = stages[s][i];
                std::cout << "[" << ops_[id].name << " (" << ops_[id].type << ")]";
                if (i + 1 < stages[s].size()) std::cout << "  ||  ";
            }
            std::cout << "\n";
            if (s < maxLevel) {
                std::cout << "             |\n             v\n";
            }
        }
    }

private:
    std::vector<OpNode> ops_;

    /// DFS 判断从 src 是否可达 dst
    bool isReachable(int src, int dst) const {
        if (src == dst) return true;
        std::vector<bool> visited(numOps(), false);
        std::function<bool(int)> dfs = [&](int u) -> bool {
            if (u == dst) return true;
            visited[u] = true;
            for (int v : ops_[u].outputs) {
                if (!visited[v] && dfs(v)) return true;
            }
            return false;
        };
        return dfs(src);
    }
};

// ============================================================================
// 构建 CNN 算子图
// ============================================================================

/**
 * 构建一个典型的 CNN 模型图:
 *
 *   Input → Conv1 → ReLU1 → Pool1 → Conv2 → ReLU2 → Pool2 → FC
 *
 * 这是最经典的 CNN 结构，广泛用于 MNIST/CIFAR 等小模型。
 * 每个算子的 FLOPs 按典型参数估算：
 *   Conv2D FLOPs = 2 * outC * inC * kH * kW * outH * outW
 *   ReLU FLOPs   = outC * outH * outW (逐元素比较)
 *   Pool FLOPs   = outC * outH * outW * poolSize
 *   FC FLOPs     = 2 * inFeatures * outFeatures
 */
OpGraph buildCnnGraph() {
    OpGraph graph;

    // ── 添加算子节点 ──
    // 输入图像: 1x3x32x32 (NCHW, batch=1)

    // Conv1: 3→32 通道, 3x3 卷积, 输出 32x32 (padding=1)
    // FLOPs = 2 * 32 * 3 * 3 * 3 * 32 * 32 = 2 * 32 * 27 * 1024 ≈ 1.77M
    int conv1 = graph.addOp("conv1", "Convolution", {}, 1'770'000, 3'600'000);

    // ReLU1: 32x32x32 逐元素
    // FLOPs = 32 * 32 * 32 = 32,768 ≈ 33K
    int relu1 = graph.addOp("relu1", "ReLU", {conv1}, 33'000, 130'000);

    // Pool1: 2x2 最大池化, 输出 32x16x16
    // FLOPs = 32 * 16 * 16 * 4 = 32,768 ≈ 33K
    int pool1 = graph.addOp("pool1", "Pooling", {relu1}, 33'000, 130'000);

    // Conv2: 32→64 通道, 3x3 卷积, 输出 64x16x16 (padding=1)
    // FLOPs = 2 * 64 * 32 * 3 * 3 * 16 * 16 = 2 * 64 * 288 * 256 ≈ 9.44M
    int conv2 = graph.addOp("conv2", "Convolution", {pool1}, 9'440'000, 19'000'000);

    // ReLU2: 64x16x16 逐元素
    // FLOPs = 64 * 16 * 16 = 16,384 ≈ 16K
    int relu2 = graph.addOp("relu2", "ReLU", {conv2}, 16'000, 65'000);

    // Pool2: 2x2 最大池化, 输出 64x8x8
    // FLOPs = 64 * 8 * 8 * 4 = 16,384 ≈ 16K
    int pool2 = graph.addOp("pool2", "Pooling", {relu2}, 16'000, 65'000);

    // FC: 64*8*8=4096 → 10 分类
    // FLOPs = 2 * 4096 * 10 = 81,920 ≈ 82K
    int fc = graph.addOp("fc", "FullyConnected", {pool2}, 82'000, 164'000);

    return graph;
}

/**
 * 构建一个带残差连接的 CNN 图，展示并行性:
 *
 *         ┌─── Conv1 → ReLU1 → Conv2 → ReLU2 ───┐
 *  Input ─┤                                       ├─ Add → Output
 *         └─────── Identity (shortcut) ──────────┘
 *
 * ResNet 的核心思想：Conv1→ReLU→Conv2→ReLU 是主分支，
 * Identity 是捷径分支，两者可以并行执行！
 */
OpGraph buildResidualGraph() {
    OpGraph graph;

    // 输入: 1x64x16x16

    // 主分支
    // Conv1: 64→64, 3x3, FLOPs ≈ 9.4M
    int conv1 = graph.addOp("conv1", "Convolution", {}, 9'400'000, 19'000'000);
    int relu1 = graph.addOp("relu1", "ReLU", {conv1}, 16'000, 65'000);
    // Conv2: 64→64, 3x3, FLOPs ≈ 9.4M
    int conv2 = graph.addOp("conv2", "Convolution", {relu1}, 9'400'000, 19'000'000);
    int relu2 = graph.addOp("relu2", "ReLU", {conv2}, 16'000, 65'000);

    // 捷径分支（Identity: 零计算，只是数据传递）
    int shortcut = graph.addOp("shortcut", "Identity", {}, 0, 32'000);

    // 汇合：Add 将两个分支结果相加
    int add = graph.addOp("add", "Add", {relu2, shortcut}, 16'000, 65'000);

    return graph;
}

// ============================================================================
// 格式化 FLOPs 为可读字符串
// ============================================================================

static std::string formatFlops(int64_t flops) {
    std::ostringstream oss;
    if (flops >= 1'000'000'000)
        oss << std::fixed << std::setprecision(2) << (flops / 1e9) << "G";
    else if (flops >= 1'000'000)
        oss << std::fixed << std::setprecision(2) << (flops / 1e6) << "M";
    else if (flops >= 1'000)
        oss << std::fixed << std::setprecision(2) << (flops / 1e3) << "K";
    else
        oss << flops;
    return oss.str();
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    std::cout << "==============================================================\n";
    std::cout << "  Lab2 Step2: 算子依赖图构建 (Op Graph Construction)\n";
    std::cout << "==============================================================\n\n";

    // ── Part 1: 构建 CNN 算子图 ──
    std::cout << "=== Part 1: 简单 CNN 算子图 ===\n\n";
    std::cout << "构建图: Input → Conv1 → ReLU1 → Pool1 → Conv2 → ReLU2 → Pool2 → FC\n\n";

    OpGraph cnnGraph = buildCnnGraph();

    // 打印图中所有节点信息
    std::cout << "图中算子列表:\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << std::left
              << std::setw(4)  << "ID"
              << std::setw(12) << "名称"
              << std::setw(16) << "类型"
              << std::setw(12) << "FLOPs"
              << std::setw(8)  << "输入"
              << std::setw(8)  << "输出" << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (const auto& op : cnnGraph.getAllOps()) {
        std::string inputsStr = "[";
        for (size_t i = 0; i < op.inputs.size(); ++i) {
            inputsStr += std::to_string(op.inputs[i]);
            if (i + 1 < op.inputs.size()) inputsStr += ",";
        }
        inputsStr += "]";

        std::cout << std::left
                  << std::setw(4)  << op.id
                  << std::setw(12) << op.name
                  << std::setw(16) << op.type
                  << std::setw(12) << formatFlops(op.estimatedFlops)
                  << std::setw(8)  << inputsStr
                  << std::setw(8)  << std::to_string(op.outputs.size())
                  << "\n";
    }
    std::cout << "\n";

    // ── 拓扑排序 ──
    auto topoOrder = cnnGraph.topologicalSort();
    std::cout << "拓扑排序结果 (合法执行顺序):  ";
    for (int id : topoOrder) {
        std::cout << cnnGraph.getOp(id).name;
        if (id != topoOrder.back()) std::cout << " → ";
    }
    std::cout << "\n\n";

    // ── 并行性分析 ──
    auto parallelOps = cnnGraph.findParallelOps();
    std::cout << "可并行执行的算子对:  ";
    if (parallelOps.empty()) {
        std::cout << "无（这是一个纯串行链式图）\n";
        std::cout << "解释: Conv→ReLU→Pool→Conv→ReLU→Pool→FC 每个都依赖前一个，\n";
        std::cout << "       没有任何两个算子可以同时执行。\n";
    } else {
        for (auto& [a, b] : parallelOps) {
            std::cout << "(" << cnnGraph.getOp(a).name
                      << ", " << cnnGraph.getOp(b).name << ") ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // ── 关键路径 ──
    auto critPath = cnnGraph.criticalPath();
    int64_t critFlops = 0;
    for (int id : critPath) critFlops += cnnGraph.getOp(id).estimatedFlops;
    std::cout << "关键路径:  ";
    for (size_t i = 0; i < critPath.size(); ++i) {
        std::cout << cnnGraph.getOp(critPath[i]).name;
        if (i + 1 < critPath.size()) std::cout << " → ";
    }
    std::cout << "  (总 FLOPs = " << formatFlops(critFlops) << ")\n";
    std::cout << "解释: 关键路径覆盖了所有节点，因为这是链式图。\n";
    std::cout << "       关键路径决定了推理延迟的理论下界。\n";

    // ── ASCII 可视化 ──
    cnnGraph.printAsciiArt();

    // ── DOT 导出 ──
    std::cout << "\n\nDOT 格式输出（可保存为 .dot 文件，用 Graphviz 渲染）:\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << cnnGraph.toDot();
    std::cout << std::string(60, '-') << "\n";
    std::cout << "提示: 将上述内容保存为 cnn_graph.dot，然后运行:\n";
    std::cout << "       dot -Tpng cnn_graph.dot -o cnn_graph.png\n";

    // ════════════════════════════════════════════════════════════
    // Part 2: 带残差连接的 CNN 图（展示并行性）
    // ════════════════════════════════════════════════════════════
    std::cout << "\n\n==============================================================\n";
    std::cout << "=== Part 2: 带残差连接的 CNN 图（展示并行性） ===\n";
    std::cout << "==============================================================\n\n";

    OpGraph resGraph = buildResidualGraph();

    std::cout << "图结构:\n";
    std::cout << "         ┌─── Conv1 → ReLU1 → Conv2 → ReLU2 ───┐\n";
    std::cout << "  Input ─┤                                       ├─ Add\n";
    std::cout << "         └─────── Identity (shortcut) ──────────┘\n\n";

    // 拓扑排序
    auto resTopo = resGraph.topologicalSort();
    std::cout << "拓扑排序:  ";
    for (int id : resTopo) {
        std::cout << resGraph.getOp(id).name;
        if (id != resTopo.back()) std::cout << " → ";
    }
    std::cout << "\n\n";

    // 并行性分析 —— 这是重点！
    auto resParallel = resGraph.findParallelOps();
    std::cout << "可并行执行的算子对:\n";
    for (auto& [a, b] : resParallel) {
        std::cout << "  (" << resGraph.getOp(a).name
                  << ", " << resGraph.getOp(b).name << ")\n";
    }
    std::cout << "\n";
    std::cout << "重点发现: shortcut 和主分支的算子可以并行！\n";
    std::cout << "  - conv1 / shortcut: 无依赖 → 可并行\n";
    std::cout << "  - relu1 / shortcut: 无依赖 → 可并行\n";
    std::cout << "  - conv2 / shortcut: 无依赖 → 可并行\n";
    std::cout << "  - relu2 / shortcut: 无依赖 → 可并行\n";
    std::cout << "  → 这就是异构调度的机会：主分支放 GPU，shortcut 放 CPU！\n";

    // 关键路径
    auto resCritPath = resGraph.criticalPath();
    int64_t resCritFlops = 0;
    for (int id : resCritPath) resCritFlops += resGraph.getOp(id).estimatedFlops;
    std::cout << "\n关键路径:  ";
    for (size_t i = 0; i < resCritPath.size(); ++i) {
        std::cout << resGraph.getOp(resCritPath[i]).name;
        if (i + 1 < resCritPath.size()) std::cout << " → ";
    }
    std::cout << "  (总 FLOPs = " << formatFlops(resCritFlops) << ")\n";
    std::cout << "解释: shortcut 的 FLOPs=0，不在关键路径上。\n";
    std::cout << "       关键路径走的是主分支（Conv→ReLU→Conv→ReLU→Add）。\n";

    // ASCII 可视化
    resGraph.printAsciiArt();

    // DOT 导出
    std::cout << "\n\n残差图 DOT 格式:\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << resGraph.toDot();
    std::cout << std::string(60, '-') << "\n";

    // ── 总结 ──
    std::cout << "\n==============================================================\n";
    std::cout << "  关键收获\n";
    std::cout << "==============================================================\n\n";
    std::cout << "1. 神经网络 = DAG（有向无环图），节点=算子，边=数据依赖\n";
    std::cout << "2. 拓扑排序给出合法的执行顺序（Kahn 算法，O(V+E)）\n";
    std::cout << "3. 并行性 = 图中无依赖路径的算子对 → 可同时执行\n";
    std::cout << "4. 关键路径 = 最长路径 → 决定推理延迟下界\n";
    std::cout << "5. 残差连接引入了并行分支 → GPU+CPU 混合调度的机会\n";
    std::cout << "6. DOT 格式可将图可视化（dot -Tpng graph.dot -o graph.png）\n";
    std::cout << "\n下一步: step3_heuristic_dispatch.cpp —— 学习基于规则的启发式调度\n";

    return 0;
}
