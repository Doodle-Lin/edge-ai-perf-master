/**
 * @file _edgeai.cpp
 * @brief pybind11 绑定 - 将 C++ 核心模块暴露给 Python
 *
 * 本文件将 edgeai_core 库的核心类绑定到 Python，使 Python 脚本
 * 可以直接调用 C++ 的 profiling、scheduling 和 operator 功能。
 *
 * 绑定策略:
 * - OpProfiler: 完整绑定，支持从 Python 进行 profiling
 * - OpRecord: 只读绑定，用于 Python 消费 profiling 结果
 * - MemoryTracker: 简化绑定，只暴露查询方法
 * - JSON 序列化: 通过 toJson() 返回字符串，Python 端 json.loads 解析
 *   这比逐字段绑定更灵活，且避免了 C++/Python 数据结构的映射问题
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>

#include "profiler/OpProfiler.h"
#include "profiler/MemoryTracker.h"
#include "profiler/PowerMonitor.h"
#include "scheduler/DeviceInfo.h"
#include "scheduler/OpGraph.h"
#include "scheduler/DispatchStrategy.h"
#include "scheduler/HeuristicDispatch.h"
#include "scheduler/ProfileGuidedDispatch.h"
#include "scheduler/HybridExecutor.h"

namespace py = pybind11;

PYBIND11_MODULE(_edgeai, m) {
    m.doc() = "EdgeAI Performance Master C++ bindings";

    // ──── OpRecord ────
    py::class_<edgeai::OpRecord>(m, "OpRecord")
        .def_readonly("name", &edgeai::OpRecord::name)
        .def_readonly("type", &edgeai::OpRecord::type)
        .def_readonly("timeMs", &edgeai::OpRecord::timeMs)
        .def_readonly("flops", &edgeai::OpRecord::flops)
        .def_readonly("memBytes", &edgeai::OpRecord::memBytes)
        .def_readonly("device", &edgeai::OpRecord::device)
        .def("gflops", &edgeai::OpRecord::gflops)
        .def("bandwidth", &edgeai::OpRecord::bandwidth)
        .def("arithmeticIntensity", &edgeai::OpRecord::arithmeticIntensity);

    // ──── OpProfiler ────
    py::class_<edgeai::OpProfiler>(m, "OpProfiler")
        .def(py::init<>())
        .def("beginOp", &edgeai::OpProfiler::beginOp,
             py::arg("name"), py::arg("type"), py::arg("device") = "CPU")
        .def("endOp", &edgeai::OpProfiler::endOp,
             py::arg("flops") = 0, py::arg("memBytes") = 0)
        .def("records", &edgeai::OpProfiler::records, py::return_value_policy::reference)
        .def("clear", &edgeai::OpProfiler::clear)
        .def("totalTimeMs", &edgeai::OpProfiler::totalTimeMs)
        .def("findBottleneck", &edgeai::OpProfiler::findBottleneck)
        .def("computeBoundRatio", &edgeai::OpProfiler::computeBoundRatio)
        .def("toJson", &edgeai::OpProfiler::toJson)
        .def("saveJson", &edgeai::OpProfiler::saveJson)
        .def("printReport", &edgeai::OpProfiler::printReport);

    // ──── MemoryTracker ────
    py::class_<edgeai::MemoryTracker>(m, "MemoryTracker")
        .def_static("instance", &edgeai::MemoryTracker::instance,
                    py::return_value_policy::reference)
        .def("onAlloc", &edgeai::MemoryTracker::onAlloc)
        .def("onFree", &edgeai::MemoryTracker::onFree)
        .def("currentUsage", &edgeai::MemoryTracker::currentUsage)
        .def("peakUsage", &edgeai::MemoryTracker::peakUsage)
        .def("printReport", &edgeai::MemoryTracker::printReport)
        .def("toJson", &edgeai::MemoryTracker::toJson)
        .def("reset", &edgeai::MemoryTracker::reset);

    // ──── PowerMonitor ────
    py::class_<edgeai::PowerMonitor>(m, "PowerMonitor")
        .def(py::init<>())
        .def("start", &edgeai::PowerMonitor::start)
        .def("stop", &edgeai::PowerMonitor::stop)
        .def("isRunning", &edgeai::PowerMonitor::isRunning)
        .def("avgCpuPower", &edgeai::PowerMonitor::avgCpuPower)
        .def("peakCpuPower", &edgeai::PowerMonitor::peakCpuPower)
        .def("avgGpuPower", &edgeai::PowerMonitor::avgGpuPower)
        .def("totalEnergyJ", &edgeai::PowerMonitor::totalEnergyJ)
        .def("printReport", &edgeai::PowerMonitor::printReport)
        .def("toJson", &edgeai::PowerMonitor::toJson);

    // ──── DeviceInfo ────
    py::class_<edgeai::CpuInfo>(m, "CpuInfo")
        .def_readonly("numCores", &edgeai::CpuInfo::numCores)
        .def_readonly("numThreads", &edgeai::CpuInfo::numThreads)
        .def_readonly("hasAvx2", &edgeai::CpuInfo::hasAvx2)
        .def_readonly("hasFma", &edgeai::CpuInfo::hasFma)
        .def_readonly("l1CacheSize", &edgeai::CpuInfo::l1CacheSize)
        .def_readonly("l2CacheSize", &edgeai::CpuInfo::l2CacheSize)
        .def_readonly("l3CacheSize", &edgeai::CpuInfo::l3CacheSize)
        .def_readonly("modelName", &edgeai::CpuInfo::modelName);

    py::class_<edgeai::DeviceInfo>(m, "DeviceInfo")
        .def_static("queryCpu", &edgeai::DeviceInfo::queryCpu)
        .def_static("queryGpu", &edgeai::DeviceInfo::queryGpu)
        .def_static("printSummary", &edgeai::DeviceInfo::printSummary);

    // ──── OpGraph ────
    py::class_<edgeai::OpNode>(m, "OpNode")
        .def_readonly("id", &edgeai::OpNode::id)
        .def_readonly("name", &edgeai::OpNode::name)
        .def_readonly("type", &edgeai::OpNode::type)
        .def_readonly("inputs", &edgeai::OpNode::inputs)
        .def_readonly("outputs", &edgeai::OpNode::outputs)
        .def_readonly("preferredDevice", &edgeai::OpNode::preferredDevice);

    py::class_<edgeai::OpGraph>(m, "OpGraph")
        .def(py::init<>())
        .def("addOp", &edgeai::OpGraph::addOp,
             py::arg("name"), py::arg("type"), py::arg("inputs") = std::vector<int>{})
        .def("getOp", &edgeai::OpGraph::getOp, py::return_value_policy::reference)
        .def("getAllOps", &edgeai::OpGraph::getAllOps, py::return_value_policy::reference)
        .def("topologicalSort", &edgeai::OpGraph::topologicalSort)
        .def("findParallelOps", &edgeai::OpGraph::findParallelOps)
        .def("printGraph", &edgeai::OpGraph::printGraph)
        .def("toDot", &edgeai::OpGraph::toDot);

    // ──── Dispatch ────
    py::enum_<edgeai::Device>(m, "Device")
        .value("CPU", edgeai::Device::CPU)
        .value("GPU", edgeai::Device::GPU);

    py::class_<edgeai::DispatchDecision>(m, "DispatchDecision")
        .def_readonly("opId", &edgeai::DispatchDecision::opId)
        .def_readonly("device", &edgeai::DispatchDecision::device)
        .def_readonly("reason", &edgeai::DispatchDecision::reason);

    py::class_<edgeai::DispatchStrategy, std::shared_ptr<edgeai::DispatchStrategy>>(
        m, "DispatchStrategy")
        .def("dispatch", &edgeai::DispatchStrategy::dispatch)
        .def("name", &edgeai::DispatchStrategy::name);

    py::class_<edgeai::HeuristicDispatch, std::shared_ptr<edgeai::HeuristicDispatch>>(
        m, "HeuristicDispatch", py::base<edgeai::DispatchStrategy>())
        .def(py::init<>());

    py::class_<edgeai::ProfileGuidedDispatch, std::shared_ptr<edgeai::ProfileGuidedDispatch>>(
        m, "ProfileGuidedDispatch", py::base<edgeai::DispatchStrategy>())
        .def(py::init<>());

    // ──── HybridExecutor ────
    py::class_<edgeai::ExecutionPlan>(m, "ExecutionPlan")
        .def_readonly("decisions", &edgeai::ExecutionPlan::decisions)
        .def_readonly("stages", &edgeai::ExecutionPlan::stages);

    py::class_<edgeai::HybridExecutor>(m, "HybridExecutor")
        .def(py::init<>())
        .def("setStrategy", &edgeai::HybridExecutor::setStrategy)
        .def("compile", &edgeai::HybridExecutor::compile);
}
