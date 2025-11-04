#include "vulkan_wrapper/core/render_graph.h"
#include <queue>
#include <spdlog/spdlog.h>


RGHandle RenderGraph::newHandle() { return static_cast<RGHandle>(m_resources.size()); }

RGHandle RenderGraph::importExternal(const std::string& name) {
    RGResourceDesc d; 
    d.name = name; 
    d.external = true;
    m_resources.push_back(d);
    return static_cast<RGHandle>(m_resources.size() - 1);
}

RGHandle RenderGraph::importExternal(const std::string& name, VkImage image, VkImageView view, VkImageLayout layout) {
    RGResourceDesc d; 
    d.name = name; 
    d.external = true;
    d.image = image;
    d.layout = layout;
    m_resources.push_back(d);
    return static_cast<RGHandle>(m_resources.size() - 1);
}

RGHandle RenderGraph::createTransient(const std::string& name) {
    RGResourceDesc d; d.name = name; d.external = false;
    m_resources.push_back(d);
    return static_cast<RGHandle>(m_resources.size() - 1);
}

RenderGraph::PassBuilder RenderGraph::addPass(const std::string& name) {
    RGPass p; p.name = name;
    m_passes.push_back(std::move(p));
    return PassBuilder{ m_passes.back(), *this };
}

RenderGraph::PassBuilder& RenderGraph::PassBuilder::read(RGHandle res, RGUsage usage) {
    pass.reads.push_back(res);
    pass.readUsages.push_back(usage);
    return *this;
}
RenderGraph::PassBuilder& RenderGraph::PassBuilder::write(RGHandle res, RGUsage usage) {
    pass.writes.push_back(res);
    pass.writeUsages.push_back(usage);
    return *this;
}
RenderGraph::PassBuilder& RenderGraph::PassBuilder::setExecute(std::function<void(const RGContext&)> fn) {
    pass.execute = std::move(fn);
    return *this;
}

void RenderGraph::emitPreBarriers(const RGContext& ctx) {
    std::vector<VkImageMemoryBarrier> barriers;
    VkPipelineStageFlags dstStages = 0;

    for (auto& p : m_passes) {
        for (size_t i = 0; i < p.writes.size(); ++i) {
            RGHandle h = p.writes[i];
            if (h >= m_resources.size()) continue;
            auto& res = m_resources[h];
            if (!res.external || res.image == VK_NULL_HANDLE) continue;

            RGUsage usage = (p.writeUsages.size() > i) ? p.writeUsages[i] : RGUsage::WriteColor;
            VkImageLayout newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            VkAccessFlags dstAccess = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

            if (Any(usage, RGUsage::WriteDepth)) {
                newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                dstAccess = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            }

            if (res.layout != newLayout) {
                VkImageMemoryBarrier b{};
                b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                b.oldLayout = res.layout;
                b.newLayout = newLayout;
                b.srcAccessMask = 0;
                b.dstAccessMask = dstAccess;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = res.image;
                b.subresourceRange.aspectMask = Any(usage, RGUsage::WriteDepth)
                    ? VK_IMAGE_ASPECT_DEPTH_BIT
                    : VK_IMAGE_ASPECT_COLOR_BIT;
                b.subresourceRange.levelCount = 1;
                b.subresourceRange.layerCount = 1;
                barriers.push_back(b);

                res.layout = newLayout;
                dstStages |= dstStage;
            }
        }
    }

    if (!barriers.empty()) {
        vkCmdPipelineBarrier(
            ctx.cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            dstStages,
            0,
            0, nullptr,
            0, nullptr,
            static_cast<uint32_t>(barriers.size()),
            barriers.data()
        );
    }
}

void RenderGraph::executePasses(const RGContext& ctx) {
    for (uint32_t idx : m_execOrder) {
        auto& p = m_passes[idx];
        if (p.execute) p.execute(ctx);
    }
}

void RenderGraph::emitPostBarriers(const RGContext& ctx) {
    std::vector<VkImageMemoryBarrier> postBarriers;
    VkPipelineStageFlags dstStages = 0;

    for (auto& p : m_passes) {
        for (size_t i = 0; i < p.writes.size(); ++i) {
            RGHandle h = p.writes[i];
            if (h >= m_resources.size()) continue;
            auto& res = m_resources[h];
            if (!res.external || res.image == VK_NULL_HANDLE) continue;

            RGUsage usage = (p.writeUsages.size() > i) ? p.writeUsages[i] : RGUsage::WriteColor;
            bool isSwapchain = res.name.find("Swapchain") != std::string::npos ||
                res.name.find("Present") != std::string::npos;
            VkImageLayout newLayout = isSwapchain
                ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                : res.layout;

            if (res.layout != newLayout) {
                VkImageMemoryBarrier b{};
                b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                b.oldLayout = res.layout;
                b.newLayout = newLayout;
                b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                b.dstAccessMask = 0;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = res.image;
                b.subresourceRange.aspectMask = Any(usage, RGUsage::WriteDepth)
                    ? VK_IMAGE_ASPECT_DEPTH_BIT
                    : VK_IMAGE_ASPECT_COLOR_BIT;
                b.subresourceRange.levelCount = 1;
                b.subresourceRange.layerCount = 1;
                postBarriers.push_back(b);

                res.layout = newLayout;
                dstStages |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            }
        }
    }

    if (!postBarriers.empty()) {
        vkCmdPipelineBarrier(
            ctx.cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            dstStages,
            0,
            0, nullptr,
            0, nullptr,
            static_cast<uint32_t>(postBarriers.size()),
            postBarriers.data()
        );
    }
}

// Build simple dependency graph: edge PassA -> PassB if A writes resource that B reads/writes later.
void RenderGraph::compile() {
    const uint32_t N = static_cast<uint32_t>(m_passes.size());
    m_execOrder.clear();
    if (N == 0) return;

    // For each resource, remember last writer pass index
    std::vector<int> lastWriter(m_resources.size(), -1);

    // Build adjacency + indegree for Kahn topo sort
    std::vector<std::vector<uint32_t>> adj(N);
    std::vector<uint32_t> indeg(N, 0);

    for (uint32_t i = 0; i < N; ++i) {
        auto& p = m_passes[i];

        // deps on prior writer for reads
        for (RGHandle r : p.reads) {
            int w = lastWriter[r];
            if (w >= 0) { adj[w].push_back(i); indeg[i]++; }
        }
        // deps on prior writer for writes (WAW ordering)
        for (RGHandle r : p.writes) {
            int w = lastWriter[r];
            if (w >= 0) { adj[w].push_back(i); indeg[i]++; }
            lastWriter[r] = static_cast<int>(i);
        }
    }

    // Kahn
    std::queue<uint32_t> q;
    for (uint32_t i = 0; i < N; ++i) if (indeg[i] == 0) q.push(i);

    while (!q.empty()) {
        uint32_t u = q.front(); q.pop();
        m_execOrder.push_back(u);
        for (uint32_t v : adj[u]) {
            if (--indeg[v] == 0) q.push(v);
        }
    }

    // Sanity: all nodes scheduled?
    assert(m_execOrder.size() == N && "Cycle in render graph!");
}
void RenderGraph::execute(const RGContext& ctx) {
    // Track current layout for all resources.
    for (auto& res : m_resources) {
        if (res.external) {
            continue;
        }
        else {
            res.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        }
    }

    for (uint32_t idx : m_execOrder) {
        auto& p = m_passes[idx];
        std::vector<VkImageMemoryBarrier> barriers;
        VkPipelineStageFlags dstStages = 0;

        auto addBarrier = [&](RGResourceDesc& res, VkImageLayout targetLayout, VkAccessFlags srcAccess, VkAccessFlags dstAccess,
            VkPipelineStageFlags dstStage, uint32_t layerCount = 1, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
                if (res.layout == targetLayout) return;
                VkImageMemoryBarrier b{};
                b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                b.oldLayout = res.layout;
                b.newLayout = targetLayout;
                b.srcAccessMask = srcAccess;
                b.dstAccessMask = dstAccess;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = res.image;
                b.subresourceRange.aspectMask = aspect;
                b.subresourceRange.baseMipLevel = 0;
                b.subresourceRange.levelCount = 1;
                b.subresourceRange.baseArrayLayer = 0;
                b.subresourceRange.layerCount = layerCount;
                barriers.push_back(b);
                dstStages |= dstStage;
                res.layout = targetLayout;
            };

        // Handle reads
        for (size_t i = 0; i < p.reads.size(); ++i) {
            RGHandle h = p.reads[i];
            if (h >= m_resources.size()) continue;
            auto& res = m_resources[h];
            if (!res.external || res.image == VK_NULL_HANDLE) continue;

            RGUsage usage = (p.readUsages.size() > i) ? p.readUsages[i] : RGUsage::ReadSampled;
            VkImageLayout targetLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkAccessFlags srcAccess = 0;
            VkAccessFlags dstAccess = 0;
            VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            uint32_t layers = 1;
            VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;

            if (Any(usage, RGUsage::ReadSampled | RGUsage::ReadUniform)) {
                targetLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                dstAccess = VK_ACCESS_SHADER_READ_BIT;
                if (res.layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
                    srcAccess = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                else if (res.layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
                    srcAccess = VK_ACCESS_TRANSFER_WRITE_BIT;
                // detect cubemap
                layers = 1;
            }
            else if (Any(usage, RGUsage::TransferSrc)) {
                targetLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                dstAccess = VK_ACCESS_TRANSFER_READ_BIT;
                if (res.layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
                    srcAccess = VK_ACCESS_TRANSFER_WRITE_BIT;
            }

            addBarrier(res, targetLayout, srcAccess, dstAccess, dstStage, layers, aspect);
        }

        // Handle writes
        for (size_t i = 0; i < p.writes.size(); ++i) {
            RGHandle h = p.writes[i];
            if (h >= m_resources.size()) continue;
            auto& res = m_resources[h];
            if (!res.external || res.image == VK_NULL_HANDLE) continue;

            RGUsage usage = (p.writeUsages.size() > i) ? p.writeUsages[i] : RGUsage::WriteColor;
            VkImageLayout targetLayout = VK_IMAGE_LAYOUT_GENERAL;
            VkAccessFlags srcAccess = 0;
            VkAccessFlags dstAccess = 0;
            VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            uint32_t layers = 1;
            VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;

            if (Any(usage, RGUsage::WriteColor)) {
                targetLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                dstAccess = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                if (res.layout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
                    srcAccess = VK_ACCESS_MEMORY_READ_BIT;
            }
            else if (Any(usage, RGUsage::WriteDepth)) {
                targetLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                dstAccess = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
                dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
            }
            else if (Any(usage, RGUsage::TransferDst)) {
                targetLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                dstAccess = VK_ACCESS_TRANSFER_WRITE_BIT;
            }

            layers = 1;
            addBarrier(res, targetLayout, srcAccess, dstAccess, dstStage, layers, aspect);
        }

        // Emit pre-pass barriers
        if (!barriers.empty()) {
            vkCmdPipelineBarrier(
                ctx.cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                dstStages ? dstStages : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0,
                0, nullptr,
                0, nullptr,
                static_cast<uint32_t>(barriers.size()),
                barriers.data()
            );
        }

        // Execute pass
        if (p.execute) p.execute(ctx);

        // Post-pass: transition written resources to final layout (swapchain PRESENT, etc.)
        std::vector<VkImageMemoryBarrier> postBarriers;
        VkPipelineStageFlags postDstStages = 0;

        for (size_t i = 0; i < p.writes.size(); ++i) {
            RGHandle h = p.writes[i];
            if (h >= m_resources.size()) continue;
            auto& res = m_resources[h];
            if (!res.external || res.image == VK_NULL_HANDLE) continue;

            VkImageLayout finalLayout = res.layout;
            RGUsage usage = (p.writeUsages.size() > i) ? p.writeUsages[i] : RGUsage::WriteColor;
            uint32_t layers = 1;
            VkImageAspectFlags aspect = (Any(usage, RGUsage::WriteDepth)) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
            VkAccessFlags srcAccess = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            VkAccessFlags dstAccess = 0;
            VkPipelineStageFlags dstStage = 0;

            if (Any(usage, RGUsage::WriteColor)) {
                // Only mark PRESENT for swapchain-like externals
                bool isSwapchain = res.name.find("Swapchain") != std::string::npos ||
                    res.name.find("Present") != std::string::npos;
                finalLayout = isSwapchain ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                    : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            }
            else if (Any(usage, RGUsage::WriteDepth)) {
                finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
                srcAccess = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                dstAccess = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            }
            else if (Any(usage, RGUsage::TransferDst)) {
                finalLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            }

            if (res.layout != finalLayout) {
                VkImageMemoryBarrier b{};
                b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                b.oldLayout = res.layout;
                b.newLayout = finalLayout;
                b.srcAccessMask = srcAccess;
                b.dstAccessMask = dstAccess;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = res.image;
                b.subresourceRange.aspectMask = aspect;
                b.subresourceRange.baseMipLevel = 0;
                b.subresourceRange.levelCount = 1;
                b.subresourceRange.baseArrayLayer = 0;
                b.subresourceRange.layerCount = layers;
                postBarriers.push_back(b);
                res.layout = finalLayout;
                postDstStages |= dstStage;
            }
        }

        if (!postBarriers.empty()) {
            vkCmdPipelineBarrier(
                ctx.cmd,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                postDstStages ? postDstStages : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0,
                0, nullptr,
                0, nullptr,
                static_cast<uint32_t>(postBarriers.size()),
                postBarriers.data()
            );
        }
    }
}

