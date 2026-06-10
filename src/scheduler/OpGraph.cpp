/**
 * @file OpGraph.cpp
 * @brief 算子依赖图的实现 —— 拓扑排序、并行性分析、关键路径、DOT 导出
 *
 * ====== 实现说明 ======
 *
 * 1. 拓扑排序使用 Kahn 算法（BFS 式）：
 *    - 维护入度表和零入度队列
 *    - 时间复杂度 O(V+E)，空间复杂度 O(V+E)
 *    - 比递归 DFS 更安全（不会栈溢出）
 *    - 输出稳定（按节点添加顺序的字典序）
 *
 * 2. 并行性分析使用传递闭包：
 *    - 用 DFS/BFS 计算每对节点之间的可达性
 *    - 不可达的节点对即为可并行对
 *    - 时间复杂度 O(V * (V+E))，适合中小规模图（<1000 节点）
 *
 * 3. 关键路径使用 DAG 上的动态规划（最长路径）：
 *    - 按拓扑序处理每个节点
 *    - dist[v] = max(dist[u] + weight(u)) for all u→v
 *    - 回溯得到路径
 *    - 时间复杂度 O(V+E)
 */

#include "scheduler/OpGraph.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <stack>

namespace edgeai {

// ============================================================================
// 图构建
// ============================================================================

int OpGraph::addOp(const std::string& name,
                    const std::string& type,
                    const std::vector<int>& inputs) {
    // 创建新节点
    OpNode node;
    node.id   = static_cast<int>(ops_.size());
    node.name = name;
    node.type = type;
    node.inputs = inputs;

    // 验证输入依赖的合法性（所有 input id 必须已存在）
    for (int inputId : inputs) {
        if (inputId < 0 || inputId >= static_cast<int>(ops_.size())) {
            throw std::out_of_range(
                "OpGraph::addOp: invalid input id " + std::to_string(inputId) +
                " for op '" + name + "' (graph has " +
                std::to_string(ops_.size()) + " nodes)");
        }
    }

    // 将节点加入列表
    ops_.push_back(node);

    // 更新前驱节点的 outputs 列表（建立反向边）
    int newId = node.id;
    for (int inputId : inputs) {
        ops_[inputId].outputs.push_back(newId);
    }

    return newId;
}

const OpNode& OpGraph::getOp(int id) const {
    if (id < 0 || id >= static_cast<int>(ops_.size())) {
        throw std::out_of_range(
            "OpGraph::getOp: id " + std::to_string(id) + " out of range [0, " +
            std::to_string(ops_.size()) + ")");
    }
    return ops_[id];
}

// ============================================================================
// 拓扑排序（Kahn 算法）
// ============================================================================

std::vector<int> OpGraph::topologicalSort() const {
    int n = static_cast<int>(ops_.size());
    if (n == 0) return {};

    // ── 步骤 1：计算每个节点的入度 ──
    // 入度 = 有多少条边指向该节点 = 该节点的 inputs 大小
    std::vector<int> inDegree(n, 0);
    for (const auto& op : ops_) {
        inDegree[op.id] = static_cast<int>(op.inputs.size());
    }

    // ── 步骤 2：将所有入度为 0 的节点入队 ──
    // 入度为 0 意味着没有前置依赖，可以最先执行
    std::queue<int> queue;
    for (int i = 0; i < n; ++i) {
        if (inDegree[i] == 0) {
            queue.push(i);
        }
    }

    // ── 步骤 3：BFS 式遍历 ──
    // 每出队一个节点，就"完成"了它，其所有后继的入度减 1
    // 如果后继入度变为 0，说明它的所有前置都完成了，可以入队
    std::vector<int> order;
    order.reserve(n);

    while (!queue.empty()) {
        int u = queue.front();
        queue.pop();
        order.push_back(u);

        // 遍历 u 的所有后继节点
        for (int v : ops_[u].outputs) {
            --inDegree[v];
            if (inDegree[v] == 0) {
                queue.push(v);
            }
        }
    }

    // ── 步骤 4：检测环 ──
    // 如果排序后的节点数 < 总节点数，说明图中有环
    // 因为环中的节点入度永远不为 0，无法出队
    if (static_cast<int>(order.size()) < n) {
        std::cerr << "[OpGraph::topologicalSort] 警告: 图中存在环！"
                  << " 已排序 " << order.size() << "/" << n << " 个节点\n";
    }

    return order;
}

// ============================================================================
// 可达性判断（DFS 辅助）
// ============================================================================

bool OpGraph::isReachable(int src, int dst) const {
    if (src == dst) return true;

    // 标准 DFS：从 src 出发，看能否到达 dst
    std::vector<bool> visited(ops_.size(), false);
    std::stack<int> stack;
    stack.push(src);
    visited[src] = true;

    while (!stack.empty()) {
        int u = stack.top();
        stack.pop();

        for (int v : ops_[u].outputs) {
            if (v == dst) return true;
            if (!visited[v]) {
                visited[v] = true;
                stack.push(v);
            }
        }
    }

    return false;
}

// ============================================================================
// 并行性分析
// ============================================================================

std::vector<std::pair<int,int>> OpGraph::findParallelOps() const {
    std::vector<std::pair<int,int>> parallelPairs;
    int n = static_cast<int>(ops_.size());

    // 对所有节点对 (i, j)，i < j，检查是否存在依赖路径
    // 如果 i 和 j 互相不可达（既不是祖先也不是后代），则可并行
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            // 双向检查：i→j 和 j→i 都不可达，则无依赖
            bool iToJ = isReachable(i, j);
            bool jToI = isReachable(j, i);
            if (!iToJ && !jToI) {
                parallelPairs.emplace_back(i, j);
            }
        }
    }

    return parallelPairs;
}

// ============================================================================
// 关键路径分析（最长路径 DP）
// ============================================================================

std::vector<int> OpGraph::criticalPath() const {
    int n = static_cast<int>(ops_.size());
    if (n == 0) return {};

    // ── 步骤 1：拓扑排序 ──
    auto order = topologicalSort();

    // ── 步骤 2：动态规划求最长路径 ──
    // dist[v] = 从源点到 v 的最长路径权重之和
    // 权重使用 estimatedFlops（如果为 0 则用默认权重 1，保证路径存在）
    std::vector<int64_t> dist(n, 0);
    // predecessor[v] = 最长路径上 v 的前驱节点
    std::vector<int> predecessor(n, -1);

    for (int u : order) {
        // 计算当前节点的"权重"——执行代价
        // estimatedFlops 为 0 时使用默认权重 1，避免所有权重为 0 导致路径长度全部为 0
        int64_t weight = ops_[u].estimatedFlops > 0 ? ops_[u].estimatedFlops : 1;

        // 松弛所有 u 的后继边
        for (int v : ops_[u].outputs) {
            if (dist[u] + weight > dist[v]) {
                dist[v] = dist[u] + weight;
                predecessor[v] = u;
            }
        }
    }

    // ── 步骤 3：找到终点（dist 最大的节点）──
    int endNode = 0;
    int64_t maxDist = 0;
    // 还需考虑终点自身的权重
    for (int i = 0; i < n; ++i) {
        int64_t weight = ops_[i].estimatedFlops > 0 ? ops_[i].estimatedFlops : 1;
        int64_t totalDist = dist[i] + weight;
        if (totalDist > maxDist) {
            maxDist = totalDist;
            endNode = i;
        }
    }

    // ── 步骤 4：回溯得到关键路径 ──
    // 从终点沿着 predecessor 链回溯到源点
    std::vector<int> path;
    int cur = endNode;
    while (cur != -1) {
        path.push_back(cur);
        cur = predecessor[cur];
    }

    // 路径是从终点到起点的，需要反转
    std::reverse(path.begin(), path.end());

    return path;
}

// ============================================================================
// 打印图结构
// ============================================================================

void OpGraph::printGraph() const {
    std::cout << "========== 算子依赖图 ==========\n";
    std::cout << "节点数: " << ops_.size() << "\n\n";

    for (const auto& op : ops_) {
        std::cout << "  [" << op.id << "] " << op.name
                  << " (type=" << op.type
                  << ", device=" << op.preferredDevice << ")\n";

        if (!op.inputs.empty()) {
            std::cout << "       依赖: ";
            for (int id : op.inputs) {
                std::cout << id << " ";
            }
            std::cout << "\n";
        }

        if (!op.outputs.empty()) {
            std::cout << "       被依赖: ";
            for (int id : op.outputs) {
                std::cout << id << " ";
            }
            std::cout << "\n";
        }
    }

    std::cout << "=================================\n";
}

// ============================================================================
// DOT 格式导出
// ============================================================================

std::string OpGraph::toDot() const {
    std::ostringstream oss;

    oss << "digraph OpGraph {\n";
    oss << "    rankdir=TB;\n";  // 从上到下布局
    oss << "    node [shape=box, style=filled, fontname=\"Arial\"];\n";
    oss << "    edge [color=\"#666666\"];\n\n";

    // ── 输出节点定义 ──
    // 根据算子类型选择颜色：
    //   蓝色 = 计算密集（Conv、Gemm）
    //   绿色 = 逐元素操作（ReLU、Add、Mul）
    //   橙色 = 内存密集（Pooling、Concat）
    //   灰色 = 其他
    for (const auto& op : ops_) {
        std::string color;
        std::string type = op.type;

        // 将类型统一转小写比较
        std::transform(type.begin(), type.end(), type.begin(), ::tolower);

        if (type.find("conv") != std::string::npos ||
            type.find("gemm") != std::string::npos ||
            type.find("matmul") != std::string::npos ||
            type.find("innerproduct") != std::string::npos) {
            color = "#4A90D9";  // 蓝色 - 计算密集型
        } else if (type.find("relu") != std::string::npos ||
                   type.find("add") != std::string::npos ||
                   type.find("mul") != std::string::npos ||
                   type.find("sigmoid") != std::string::npos ||
                   type.find("tanh") != std::string::npos ||
                   type.find("bn") != std::string::npos ||
                   type.find("batchnorm") != std::string::npos) {
            color = "#7BC67E";  // 绿色 - 逐元素操作
        } else if (type.find("pool") != std::string::npos ||
                   type.find("concat") != std::string::npos ||
                   type.find("split") != std::string::npos ||
                   type.find("slice") != std::string::npos ||
                   type.find("reshape") != std::string::npos) {
            color = "#F5A623";  // 橙色 - 内存密集/形状操作
        } else {
            color = "#BDC3C7";  // 灰色 - 其他
        }

        // DOT 节点定义，标签显示名称和类型
        oss << "    node" << op.id
            << " [label=\"" << op.name << "\\n" << op.type
            << "\", fillcolor=\"" << color << "\"];\n";
    }

    oss << "\n";

    // ── 输出边定义 ──
    // 从 inputs 推导边：如果 op 的 inputs 包含 id，则有 id → op.id 的边
    for (const auto& op : ops_) {
        for (int inputId : op.inputs) {
            oss << "    node" << inputId << " -> node" << op.id << ";\n";
        }
    }

    oss << "}\n";
    return oss.str();
}

} // namespace edgeai
