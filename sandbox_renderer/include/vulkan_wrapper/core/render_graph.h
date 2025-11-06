#pragma once
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cassert>
#include "vulkan_wrapper/vulkan_device.h"

struct FrameInfo;


struct RGContext {
    VkSandboxDevice* device = nullptr;
    VkCommandBuffer  cmd = VK_NULL_HANDLE;
    uint32_t         frameIndex = 0;

    VkDescriptorSet  globalSet = VK_NULL_HANDLE;
    FrameInfo* frame = nullptr;
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
    bool external = false;
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
    RGHandle importExternal(const std::string& name, VkImage image, VkImageView view, VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED);
   
    RGHandle createTransient(const std::string& name);

    // Pass creation
    struct PassBuilder {
        RGPass& pass;
        RenderGraph& graph;

        PassBuilder& read(RGHandle res, RGUsage usage);
        PassBuilder& write(RGHandle res, RGUsage usage);
        PassBuilder& setExecute(std::function<void(const RGContext&)> fn);
    };

    void emitPreBarriers(const RGContext& ctx);
    void executePasses(const RGContext& ctx);
    void emitPostBarriers(const RGContext& ctx);

    PassBuilder addPass(const std::string& name);

    void compile();  // builds execution order by deps
    void execute(const RGContext& ctx);

    // access
    const std::vector<RGPass>& getPasses() const { return m_passes; }

private:
    RGHandle newHandle();

    std::vector<RGResourceDesc> m_resources;
    std::vector<RGPass>         m_passes;

    std::vector<uint32_t>       m_execOrder;
};
