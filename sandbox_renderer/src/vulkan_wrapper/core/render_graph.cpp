#include "vulkan_wrapper/core/render_graph.h"
#include <queue>
#include <spdlog/spdlog.h>


RGHandle RenderGraph::newHandle() { return static_cast<RGHandle>(m_resources.size()); }

RGHandle RenderGraph::importExternal(const std::string& name) {
    RGResourceDesc d; d.name = name; d.external = true;
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
    for (uint32_t idx : m_execOrder) {
        auto& p = m_passes[idx];

        // --- Automatic layout transitions for writes ---
        for (size_t i = 0; i < p.writes.size(); i++) {
            RGHandle h = p.writes[i];
            if (h >= m_resources.size()) {
                spdlog::error("Invalid RGHandle {} in pass '{}'", h, p.name);
                continue;
            }

            auto& res = m_resources[h];

            if (!res.external || res.image == VK_NULL_HANDLE) continue;  // skip invalid

            RGUsage usage = p.writeUsages[i];  // <-- FIX: get usage for this write

            VkImageLayout targetLayout = VK_IMAGE_LAYOUT_GENERAL;
            if (usage == RGUsage::WriteColor) targetLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            else if (usage == RGUsage::WriteDepth) targetLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            if (res.layout != targetLayout) {
                VkImageMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barrier.oldLayout = res.layout;
                barrier.newLayout = targetLayout;
                barrier.image = res.image;
                barrier.subresourceRange.aspectMask =
                    (usage == RGUsage::WriteDepth) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
                barrier.subresourceRange.levelCount = 1;
                barrier.subresourceRange.layerCount = 1;
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask =
                    (usage == RGUsage::WriteDepth) ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                    : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

                vkCmdPipelineBarrier(
                    ctx.cmd,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                    0, 0, nullptr, 0, nullptr, 1, &barrier
                );

                res.layout = targetLayout;
            }
        }

        // Execute pass callback
        if (p.execute) p.execute(ctx);
    }
}
