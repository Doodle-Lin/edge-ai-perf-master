/**
 * @file OpGraph.h
 * @brief 算子依赖图 —— 将神经网络表示为有向无环图 (DAG)，支持拓扑分析与调度
 *
 * ====== 学习要点 ======
 *
 * 1. 为什么用 DAG 表示神经网络？
 *    神经网络的推理过程本质上是数据流图：
 *    - 节点 = 算子（Conv、ReLU、Pooling 等）
 *    - 边   = 张量（一个算子的输出是另一个的输入）
 *    DAG（有向无环图）天然表达这种"先算谁、后算谁"的依赖关系。
 *    不用一般的有向图是因为推理不存在循环（训练中的循环通过展开处理）。
 *
 * 2. 拓扑排序（Topological Sort）
 *    将 DAG 的节点排成线性序列，使得每条边 (u→v) 中 u 排在 v 前面。
 *    这就是"合法的执行顺序"——保证每个算子的输入都已计算完毕。
 *    经典算法：Kahn 算法（BFS 式，用入度表）或 DFS 后序逆序。
 *
 * 3. 并行性分析
 *    如果两个算子之间没有依赖路径，它们就可以并行执行。
 *    例如 ResNet 中的残差分支和主分支，在分叉后、汇合前是独立的。
 *    findParallelOps() 找出所有这样的"可并行算子对"。
 *
 * 4. 关键路径（Critical Path）
 *    图中的最长路径，决定了推理延迟的下界。
 *    即使有无限多的计算资源，延迟也不可能低于关键路径长度。
 *    优化思路：优先加速关键路径上的算子（把它们分给 GPU）。
 *
 * 5. DOT 格式
 *    Graphviz 的图描述语言，用文本描述图结构，然后用 dot 命令渲染为图片。
 *    示例：digraph G { conv1 -> relu1; relu1 -> conv2; }
 */

#pragma once

#include <string>
#include <vector>
#include <utility>

namespace edgeai {

// ============================================================================
// 算子节点
// ============================================================================

/// 图中的单个算子节点
struct OpNode {
    int id = -1;                        ///< 唯一标识符，从 0 开始递增
    std::string name;                    ///< 算子实例名，如 "conv1"、"fc_out"
    std::string type;                    ///< 算子类型，如 "Convolution"、"Pooling"、"ReLU"
    std::vector<int> inputs;            ///< 依赖的算子 id 列表（数据来源）
    std::vector<int> outputs;           ///< 被哪些算子依赖（数据去向）
    std::string preferredDevice = "AUTO"; ///< 优先执行设备: "CPU" / "GPU" / "AUTO"

    /// 估算的 FLOPs（浮点运算次数），用于调度器判断算子规模
    /// 0 表示未设置，调度器会使用启发式规则估算
    int64_t estimatedFlops = 0;

    /// 估算的内存访问量（字节），包括输入 + 输出 + 权重
    int64_t estimatedMemBytes = 0;
};

// ============================================================================
// 算子依赖图
// ============================================================================

/**
 * @class OpGraph
 * @brief 算子依赖图（有向无环图 DAG）
 *
 * 构建图的典型流程：
 * @code
 *   OpGraph graph;
 *   int conv1 = graph.addOp("conv1", "Convolution");
 *   int relu1 = graph.addOp("relu1", "ReLU", {conv1});
 *   int conv2 = graph.addOp("conv2", "Convolution", {relu1});
 *   int pool  = graph.addOp("pool",  "Pooling",   {conv2});
 *
 *   // 拓扑排序，得到合法执行顺序
 *   auto order = graph.topologicalSort();
 * @endcode
 */
class OpGraph {
public:
    // ──────────────── 图构建 ────────────────

    /**
     * @brief 向图中添加一个算子节点
     * @param name   算子实例名
     * @param type   算子类型
     * @param inputs 该算子依赖的前驱算子 id 列表
     * @return 新算子的 id（等于 ops_.size() - 1）
     *
     * 添加边：对于 inputs 中的每个 id，自动建立 "id → 新节点" 的依赖边。
     * 如果 id 无效（越界），会抛出 std::out_of_range。
     */
    int addOp(const std::string& name,
              const std::string& type,
              const std::vector<int>& inputs = {});

    /// 按 id 获取算子节点（只读），越界则抛出 std::out_of_range
    const OpNode& getOp(int id) const;

    /// 获取所有算子节点（只读引用）
    const std::vector<OpNode>& getAllOps() const { return ops_; }

    /// 获取图中算子总数
    int numOps() const { return static_cast<int>(ops_.size()); }

    // ──────────────── 图分析 ────────────────

    /**
     * @brief 拓扑排序 —— 返回合法的执行顺序
     * @return 算子 id 的有序列表，保证依赖关系满足
     *
     * 使用 Kahn 算法（BFS 式）：
     * 1. 计算每个节点的入度
     * 2. 入度为 0 的节点入队
     * 3. 出队一个节点，将其后继的入度减 1
     * 4. 后继入度为 0 则入队
     * 5. 重复直到队列为空
     *
     * 时间复杂度 O(V+E)，比 DFS 式更稳定（不会栈溢出）。
     * 如果图中有环，返回的序列长度 < V，并打印警告。
     */
    std::vector<int> topologicalSort() const;

    /**
     * @brief 找出可以并行执行的算子对
     * @return 所有互不依赖的 (id1, id2) 对，保证 id1 < id2
     *
     * 判定方法：两个算子之间没有直接或间接的依赖关系（既不是祖先也不是后代）。
     * 通过计算传递闭包来确定：如果 op[a] 不是 op[b] 的祖先且 op[b] 也不是 op[a] 的祖先，
     * 则它们可以并行。
     */
    std::vector<std::pair<int,int>> findParallelOps() const;

    /**
     * @brief 关键路径分析 —— 找出图中的最长执行路径
     * @return 关键路径上的算子 id 列表（按执行顺序）
     *
     * 基于 estimatedFlops 估算权重，使用动态规划求最长路径：
     *   dist[v] = max(dist[u] + weight(u,v)) for all u → v
     * 然后回溯得到完整路径。
     *
     * 关键路径的意义：
     * - 它决定了推理延迟的理论下界
     * - 关键路径上的算子应优先分配给更快的设备（如 GPU）
     * - 非关键路径上的算子可以分配给较慢的设备而不影响总延迟
     */
    std::vector<int> criticalPath() const;

    // ──────────────── 可视化 ────────────────

    /// 打印图结构到 stdout（邻接表形式）
    void printGraph() const;

    /**
     * @brief 导出为 DOT 格式字符串
     * @return 可被 Graphviz 渲染的 DOT 文本
     *
     * 用法：
     * @code
     *   std::ofstream out("graph.dot");
     *   out << graph.toDot();
     *   out.close();
     *   // 命令行渲染: dot -Tpng graph.dot -o graph.png
     * @endcode
     *
     * 节点颜色根据算子类型着色：
     * - Convolution → 蓝色（计算密集）
     * - ReLU / Add  → 绿色（逐元素操作）
     * - Pooling     → 橙色（内存密集）
     * - 其他        → 灰色
     */
    std::string toDot() const;

private:
    std::vector<OpNode> ops_;  ///< 算子节点列表，id 等于下标

    /**
     * @brief 检测从 src 是否可达 dst（DFS 辅助）
     * @return true 表示存在路径 src → ... → dst
     *
     * 用于 findParallelOps() 判断两个节点之间是否有依赖关系。
     */
    bool isReachable(int src, int dst) const;
};

} // namespace edgeai
