#pragma once
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cassert>
#include "vulkan_wrapper/vulkan_device.h"



struct RGContext {
    VkSandboxDevice* device = nullptr;
    VkCommandBuffer  cmd = VK_NULL_HANDLE;
    uint32_t         frameIndex = 0;

    // You can add global sets/UBOs if you want them accessible to passes
    VkDescriptorSet  globalSet = VK_NULL_HANDLE;
    // ... add more as needed
};

// Logical resource handle
using RGHandle = uint32_t;

enum class RGUsage : uint32_t {
    None = 0,
    ReadSampled = 1 << 0,
    ReadUniform = 1 << 1,
    WriteColor = 1 << 2,
    WriteDepth = 1 << 3,
    TransferSrc = 1 << 4,
    TransferDst = 1 << 5,
};
inline RGUsage operator|(RGUsage a, RGUsage b) { return RGUsage(uint32_t(a) | uint32_t(b)); }
inline bool    Any(RGUsage u, RGUsage bits) { return (uint32_t(u) & uint32_t(bits)) != 0; }

struct RGResourceDesc {
    std::string name;
    // In v1 these are logical only; in v2 you can add format, extent, etc.
    bool external = false; // swapchain/depth external
    VkImage        image = VK_NULL_HANDLE;  // only for external resources
    VkImageLayout  layout = VK_IMAGE_LAYOUT_UNDEFINED;  // track current layout
};

// Pass declaration
struct RGPass {
    std::string name;

    // Declared deps
    std::vector<RGHandle> reads;
    std::vector<RGHandle> writes;
    std::vector<RGUsage>  readUsages;   // parallel to reads
    std::vector<RGUsage>  writeUsages;  // parallel to writes

    // Execution callback
    std::function<void(const RGContext&)> execute;
};

class RenderGraph {
public:
    explicit RenderGraph() = default;

    // Resource creation
    RGHandle importExternal(const std::string& name);
    RGHandle createTransient(const std::string& name);

    // Pass creation
    // Usage: auto pass = graph.addPass("GltfPbr"); pass.read(color, RGUsage::WriteColor) ... ; pass.setExecute(...)
    struct PassBuilder {
        RGPass& pass;
        RenderGraph& graph;

        PassBuilder& read(RGHandle res, RGUsage usage);
        PassBuilder& write(RGHandle res, RGUsage usage);
        PassBuilder& setExecute(std::function<void(const RGContext&)> fn);
    };

    PassBuilder addPass(const std::string& name);

    void compile();  // builds execution order by deps
    void execute(const RGContext& ctx);

    // access
    const std::vector<RGPass>& getPasses() const { return m_passes; }

private:
    RGHandle newHandle();

    std::vector<RGResourceDesc> m_resources;
    std::vector<RGPass>         m_passes;

    // compiled order (indices into m_passes)
    std::vector<uint32_t>       m_execOrder;
};
