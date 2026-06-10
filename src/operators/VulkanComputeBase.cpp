/**
 * @file VulkanComputeBase.cpp
 * @brief Vulkan 计算基类 —— GPU 通用计算的底层封装
 *
 * ====== 实现说明 ======
 * 1. 封装 Vulkan Compute 的完整生命周期：初始化 → 着色器加载 → 管线创建 → 缓冲区管理 → 调度 → 清理
 * 2. 所有代码包裹在 #ifdef ENABLE_VULKAN 中，无 Vulkan SDK 时提供 stub 实现
 * 3. 资源按创建的反序销毁，遵循 Vulkan 规范
 */

#include "operators/VulkanComputeBase.h"

// This file provides the VulkanComputeBase implementation.
// When ENABLE_VULKAN is not defined, the header's class definition is
// inside #ifdef ENABLE_VULKAN, so the class does not exist. The entire
// implementation below is correspondingly guarded.

#ifdef ENABLE_VULKAN

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace edgeai {

bool VulkanComputeBase::initialize()
{
    /*
     * Step 1: Create VkInstance
     * The instance is the top-level Vulkan object that loads the driver.
     */
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "EdgeAIPerf";
    appInfo.applicationVersion = 1;
    appInfo.pEngineName = "EdgeAI";
    appInfo.engineVersion = 1;
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        std::cerr << "[VulkanComputeBase] Failed to create VkInstance: " << result << std::endl;
        return false;
    }

    /*
     * Step 2: Find a physical device with compute capability
     * Enumerate all GPUs and pick the first one that supports a compute queue.
     */
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::cerr << "[VulkanComputeBase] No Vulkan-capable GPUs found" << std::endl;
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

    physicalDevice_ = VK_NULL_HANDLE;
    computeQueueFamilyIndex_ = 0;

    for (const auto& dev : devices) {
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, queueFamilies.data());

        for (uint32_t i = 0; i < queueFamilyCount; ++i) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                physicalDevice_ = dev;
                computeQueueFamilyIndex_ = i;
                break;
            }
        }
        if (physicalDevice_ != VK_NULL_HANDLE) break;
    }

    if (physicalDevice_ == VK_NULL_HANDLE) {
        std::cerr << "[VulkanComputeBase] No compute-capable GPU found" << std::endl;
        return false;
    }

    /*
     * Step 3: Create logical device with compute queue
     * The logical device is the application's view of the physical device.
     */
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex_;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

    result = vkCreateDevice(physicalDevice_, &deviceCreateInfo, nullptr, &device_);
    if (result != VK_SUCCESS) {
        std::cerr << "[VulkanComputeBase] Failed to create logical device: " << result << std::endl;
        return false;
    }

    vkGetDeviceQueue(device_, computeQueueFamilyIndex_, 0, &computeQueue_);

    /*
     * Step 4: Create command pool and command buffer
     * The command pool allocates command buffers that record GPU commands.
     */
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeQueueFamilyIndex_;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    result = vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_);
    if (result != VK_SUCCESS) {
        std::cerr << "[VulkanComputeBase] Failed to create command pool: " << result << std::endl;
        return false;
    }

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool_;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    result = vkAllocateCommandBuffers(device_, &cmdAllocInfo, &commandBuffer_);
    if (result != VK_SUCCESS) {
        std::cerr << "[VulkanComputeBase] Failed to allocate command buffer: " << result << std::endl;
        return false;
    }

    return true;
}

bool VulkanComputeBase::loadShader(const std::string& spvPath,
                                    const std::string& entryPoint)
{
    (void)entryPoint;

    // Read SPIR-V file (binary)
    std::ifstream file(spvPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[VulkanComputeBase] Failed to open shader file: " << spvPath << std::endl;
        return false;
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    spirvCode_.resize(fileSize / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(spirvCode_.data()), static_cast<std::streamsize>(fileSize));
    file.close();

    if (spirvCode_.empty()) {
        std::cerr << "[VulkanComputeBase] Shader file is empty: " << spvPath << std::endl;
        return false;
    }

    // Create shader module from SPIR-V code
    VkShaderModuleCreateInfo moduleInfo{};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = spirvCode_.size() * sizeof(uint32_t);
    moduleInfo.pCode = spirvCode_.data();

    VkResult result = vkCreateShaderModule(device_, &moduleInfo, nullptr, &shaderModule_);
    if (result != VK_SUCCESS) {
        std::cerr << "[VulkanComputeBase] Failed to create shader module: " << result << std::endl;
        return false;
    }

    return true;
}

bool VulkanComputeBase::createPipeline()
{
    /*
     * Create descriptor set layout: one binding per buffer
     * All buffers are storage buffers (read/write access from shader).
     * Binding 0, 1, 2, ... correspond to the order createBuffer() was called.
     */
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
    for (int i = 0; i < bufferCount_; ++i) {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = i;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        layoutBindings.push_back(binding);
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(layoutBindings.size());
    layoutInfo.pBindings = layoutBindings.data();

    VkResult result = vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_);
    if (result != VK_SUCCESS) {
        std::cerr << "[VulkanComputeBase] Failed to create descriptor set layout: " << result << std::endl;
        return false;
    }

    // Create pipeline layout with push constants for small parameters
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = 128;  // Maximum push constant size (typical)

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    result = vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_);
    if (result != VK_SUCCESS) {
        std::cerr << "[VulkanComputeBase] Failed to create pipeline layout: " << result << std::endl;
        return false;
    }

    // Create compute pipeline
    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = shaderModule_;
    shaderStageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = pipelineLayout_;

    result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_);
    if (result != VK_SUCCESS) {
        std::cerr << "[VulkanComputeBase] Failed to create compute pipeline: " << result << std::endl;
        return false;
    }

    // Shader module can be destroyed after pipeline creation
    vkDestroyShaderModule(device_, shaderModule_, nullptr);
    shaderModule_ = VK_NULL_HANDLE;

    // Create descriptor pool
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = static_cast<uint32_t>(bufferCount_);

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    result = vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_);
    if (result != VK_SUCCESS) {
        std::cerr << "[VulkanComputeBase] Failed to create descriptor pool: " << result << std::endl;
        return false;
    }

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo descAllocInfo{};
    descAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descAllocInfo.descriptorPool = descriptorPool_;
    descAllocInfo.descriptorSetCount = 1;
    descAllocInfo.pSetLayouts = &descriptorSetLayout_;

    result = vkAllocateDescriptorSets(device_, &descAllocInfo, &descriptorSet_);
    if (result != VK_SUCCESS) {
        std::cerr << "[VulkanComputeBase] Failed to allocate descriptor set: " << result << std::endl;
        return false;
    }

    // Update descriptor set to bind buffers
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    std::vector<VkWriteDescriptorSet> writes;

    for (int i = 0; i < bufferCount_; ++i) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = buffers_[i].buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = buffers_[i].size;
        bufferInfos.push_back(bufferInfo);

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptorSet_;
        write.dstBinding = i;
        write.dstArrayElement = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &bufferInfos[i];
        writes.push_back(write);
    }

    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    return true;
}

bool VulkanComputeBase::createBuffer(size_t size, VkBufferUsageFlags usage)
{
    /*
     * Create a GPU buffer and allocate device memory.
     * Memory type selection: prefer HOST_VISIBLE + HOST_COHERENT for zero-copy access.
     */
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    VkResult result = vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer);
    if (result != VK_SUCCESS) {
        std::cerr << "[VulkanComputeBase] Failed to create buffer: " << result << std::endl;
        return false;
    }

    // Get memory requirements
    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device_, buffer, &memReqs);

    // Find suitable memory type (HOST_VISIBLE + HOST_COHERENT)
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);

    uint32_t memoryTypeIndex = UINT32_MAX;
    VkMemoryPropertyFlags desiredProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReqs.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & desiredProps) == desiredProps) {
            memoryTypeIndex = i;
            break;
        }
    }

    if (memoryTypeIndex == UINT32_MAX) {
        std::cerr << "[VulkanComputeBase] No suitable memory type found" << std::endl;
        vkDestroyBuffer(device_, buffer, nullptr);
        return false;
    }

    // Allocate device memory
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;

    VkDeviceMemory memory;
    result = vkAllocateMemory(device_, &allocInfo, nullptr, &memory);
    if (result != VK_SUCCESS) {
        std::cerr << "[VulkanComputeBase] Failed to allocate device memory: " << result << std::endl;
        vkDestroyBuffer(device_, buffer, nullptr);
        return false;
    }

    // Bind memory to buffer
    result = vkBindBufferMemory(device_, buffer, memory, 0);
    if (result != VK_SUCCESS) {
        std::cerr << "[VulkanComputeBase] Failed to bind buffer memory: " << result << std::endl;
        vkFreeMemory(device_, memory, nullptr);
        vkDestroyBuffer(device_, buffer, nullptr);
        return false;
    }

    // Store buffer info
    buffers_.push_back({buffer, memory, size});
    bufferCount_ = static_cast<int>(buffers_.size());

    return true;
}

bool VulkanComputeBase::writeBuffer(int binding, const void* data, size_t size)
{
    if (binding < 0 || binding >= static_cast<int>(buffers_.size())) {
        std::cerr << "[VulkanComputeBase] Invalid binding: " << binding << std::endl;
        return false;
    }

    void* mapped = nullptr;
    VkResult result = vkMapMemory(device_, buffers_[binding].memory, 0, size, 0, &mapped);
    if (result != VK_SUCCESS) {
        std::cerr << "[VulkanComputeBase] Failed to map buffer memory: " << result << std::endl;
        return false;
    }

    std::memcpy(mapped, data, size);
    vkUnmapMemory(device_, buffers_[binding].memory);

    return true;
}

bool VulkanComputeBase::readBuffer(int binding, void* data, size_t size)
{
    if (binding < 0 || binding >= static_cast<int>(buffers_.size())) {
        std::cerr << "[VulkanComputeBase] Invalid binding: " << binding << std::endl;
        return false;
    }

    void* mapped = nullptr;
    VkResult result = vkMapMemory(device_, buffers_[binding].memory, 0, size, 0, &mapped);
    if (result != VK_SUCCESS) {
        std::cerr << "[VulkanComputeBase] Failed to map buffer memory for read: " << result << std::endl;
        return false;
    }

    std::memcpy(data, mapped, size);
    vkUnmapMemory(device_, buffers_[binding].memory);

    return true;
}

bool VulkanComputeBase::dispatch(uint32_t groupCountX, uint32_t groupCountY,
                                  uint32_t groupCountZ)
{
    // Begin command buffer recording
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkResult result = vkBeginCommandBuffer(commandBuffer_, &beginInfo);
    if (result != VK_SUCCESS) {
        std::cerr << "[VulkanComputeBase] Failed to begin command buffer: " << result << std::endl;
        return false;
    }

    // Bind pipeline and descriptor set
    vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE,
                             pipelineLayout_, 0, 1, &descriptorSet_, 0, nullptr);

    // Dispatch compute
    vkCmdDispatch(commandBuffer_, groupCountX, groupCountY, groupCountZ);

    // End command buffer recording
    result = vkEndCommandBuffer(commandBuffer_);
    if (result != VK_SUCCESS) {
        std::cerr << "[VulkanComputeBase] Failed to end command buffer: " << result << std::endl;
        return false;
    }

    // Submit command buffer to compute queue
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer_;

    result = vkQueueSubmit(computeQueue_, 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        std::cerr << "[VulkanComputeBase] Failed to submit to queue: " << result << std::endl;
        return false;
    }

    // Wait for completion
    result = vkQueueWaitIdle(computeQueue_);
    if (result != VK_SUCCESS) {
        std::cerr << "[VulkanComputeBase] Failed to wait for queue idle: " << result << std::endl;
        return false;
    }

    return true;
}

void VulkanComputeBase::cleanup()
{
    /*
     * Destroy Vulkan objects in reverse order of creation.
     * Vulkan resources have dependencies, so order matters.
     */
    if (device_ == VK_NULL_HANDLE) return;

    vkDeviceWaitIdle(device_);

    // Destroy buffers and free memory
    for (auto& buf : buffers_) {
        if (buf.buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, buf.buffer, nullptr);
        }
        if (buf.memory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, buf.memory, nullptr);
        }
    }
    buffers_.clear();
    bufferCount_ = 0;

    // Destroy pipeline
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }

    // Destroy pipeline layout
    if (pipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        pipelineLayout_ = VK_NULL_HANDLE;
    }

    // Destroy descriptor set layout
    if (descriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        descriptorSetLayout_ = VK_NULL_HANDLE;
    }

    // Destroy descriptor pool
    if (descriptorPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        descriptorPool_ = VK_NULL_HANDLE;
    }

    // Free command buffer and destroy command pool
    if (commandPool_ != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer_);
        vkDestroyCommandPool(device_, commandPool_, nullptr);
        commandPool_ = VK_NULL_HANDLE;
        commandBuffer_ = VK_NULL_HANDLE;
    }

    // Destroy logical device
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }

    // Destroy instance
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }

    physicalDevice_ = VK_NULL_HANDLE;
    computeQueue_ = VK_NULL_HANDLE;
}

} // namespace edgeai

#endif // ENABLE_VULKAN
