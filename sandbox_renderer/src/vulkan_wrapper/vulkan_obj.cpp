// obj_model.cpp
#include "common/renderer_pch.h"
#include "vulkan_wrapper/vulkan_obj.h"



// External
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

namespace std
{
    template <>
    struct hash<VkSandboxOBJmodel::Vertex> {
        size_t operator()(VkSandboxOBJmodel::Vertex const& vertex) const
        {
            size_t seed = 0;
            tools::hashCombine(seed, vertex.position, vertex.color, vertex.normal, vertex.uv);
            return seed;
        }
    };
}

VkSandboxOBJmodel::VkSandboxOBJmodel(VkSandboxDevice& device, const Builder& builder)
    : m_device{ device }, m_bIsSkyboxModel{ builder.isSkybox } {

    if (builder.isSkybox) {
        //createVertexBuffers(builder.skyboxVertices);
    }
    else {
        createVertexBuffers(builder.vertices);
        createIndexBuffers(builder.indices);
    }


}

VkSandboxOBJmodel::~VkSandboxOBJmodel() {}


std::shared_ptr<VkSandboxOBJmodel> VkSandboxOBJmodel::createModelFromFile(VkSandboxDevice& device, const std::string& filepath, bool isSkybox)
{
    Builder builder{};
    builder.loadModel(filepath, isSkybox);
    return std::make_shared<VkSandboxOBJmodel>(device, builder);
}



void VkSandboxOBJmodel::createVertexBuffers(const std::vector<Vertex>& vertices)
{
    m_vertexCount = static_cast<uint32_t>(vertices.size());
    assert(m_vertexCount >= 3 && "Vertex count must be at least 3");
    VkDeviceSize bufferSize = sizeof(vertices[0]) * m_vertexCount;
    uint32_t vertexSize = sizeof(vertices[0]);

    VkSandboxBuffer stagingBuffer{
        m_device,
        vertexSize,
        m_vertexCount,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    };

    stagingBuffer.map();
    stagingBuffer.writeToBuffer((void*)vertices.data());

    m_vertexBuffer = std::make_unique<VkSandboxBuffer>(
        m_device,
        vertexSize,
        m_vertexCount,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    m_device.copyBuffer(stagingBuffer.getBuffer(), m_vertexBuffer->getBuffer(), bufferSize);
}

void VkSandboxOBJmodel::createIndexBuffers(const std::vector<uint32_t>& indices)
{
    m_indexCount = static_cast<uint32_t>(indices.size());
    m_bHasIndexBuffer = m_indexCount > 0;
    if (!m_bHasIndexBuffer) return;

    VkDeviceSize bufferSize = sizeof(indices[0]) * m_indexCount;
    uint32_t indexSize = sizeof(indices[0]);

    VkSandboxBuffer stagingBuffer{
        m_device,
        indexSize,
        m_indexCount,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    };

    stagingBuffer.map();
    stagingBuffer.writeToBuffer((void*)indices.data());

    m_indexBuffer = std::make_unique<VkSandboxBuffer>(
        m_device,
        indexSize,
        m_indexCount,
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    m_device.copyBuffer(stagingBuffer.getBuffer(), m_indexBuffer->getBuffer(), bufferSize);
}

void VkSandboxOBJmodel::draw(VkCommandBuffer commandBuffer)
{

    VkCommandBuffer cmd = commandBuffer;

    if (m_bHasIndexBuffer) {
        vkCmdDrawIndexed(cmd, m_indexCount, 1, 0, 0, 0);
    }
    else {
        vkCmdDraw(cmd, m_vertexCount, 1, 0, 0);
    }
}


void VkSandboxOBJmodel::bind(VkCommandBuffer commandBuffer)
{

    VkCommandBuffer cmd = commandBuffer;

    VkBuffer buffers[] = { m_vertexBuffer->getBuffer() };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, buffers, offsets);

    if (m_bHasIndexBuffer) {
        vkCmdBindIndexBuffer(cmd, m_indexBuffer->getBuffer(), 0, VK_INDEX_TYPE_UINT32);
    }
}



std::vector<VkVertexInputBindingDescription> VkSandboxOBJmodel::Vertex::getBindingDescriptions()
{
    return {
        VkVertexInputBindingDescription{
            0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX
        }
    };
}

std::vector<VkVertexInputAttributeDescription> VkSandboxOBJmodel::Vertex::getAttributeDescriptions()
{
    return {
        { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
        { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
        { 2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) }
        //{ 3, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv) }
    };
}


void VkSandboxOBJmodel::Builder::loadModel(const std::string& filepath, bool isSkybox) {
    this->isSkybox = isSkybox;
    vertices.clear();
    //skyboxVertices.clear();
    indices.clear();

    // 1) Check extension
    auto lastDot = filepath.find_last_of('.');
    if (lastDot == std::string::npos) {
        throw std::runtime_error("Model file has no extension: " + filepath);
    }
    std::string ext = filepath.substr(lastDot + 1);
    for (auto& c : ext) c = static_cast<char>(::tolower(c));

    if (ext == "obj") {
        // tinyobj loader ---
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t>    shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str())) {
            throw std::runtime_error("Failed to load OBJ: " + warn + err);
        }


        else {
            std::unordered_map<Vertex, uint32_t> uniqueVertices{};
            for (auto const& shape : shapes) {
                for (auto const& index : shape.mesh.indices) {
                    Vertex vertex{};
                    // copy position, color, normal, uv exactly as before
                    if (index.vertex_index >= 0) {
                        vertex.position = {
                            attrib.vertices[3 * index.vertex_index + 0],
                            attrib.vertices[3 * index.vertex_index + 1],
                            attrib.vertices[3 * index.vertex_index + 2],
                        };
                    }
                    if (!attrib.colors.empty()) {
                        vertex.color = {
                            attrib.colors[3 * index.vertex_index + 0],
                            attrib.colors[3 * index.vertex_index + 1],
                            attrib.colors[3 * index.vertex_index + 2],
                        };
                    }
                    if (index.normal_index >= 0) {
                        vertex.normal = {
                            attrib.normals[3 * index.normal_index + 0],
                            attrib.normals[3 * index.normal_index + 1],
                            attrib.normals[3 * index.normal_index + 2],
                        };
                    }
                    if (index.texcoord_index >= 0) {
                        vertex.uv = {
                            attrib.texcoords[2 * index.texcoord_index + 0],
                            1.0f - attrib.texcoords[2 * index.texcoord_index + 1],
                        };
                    }

                    if (uniqueVertices.count(vertex) == 0) {
                        uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                        vertices.push_back(vertex);
                    }
                    indices.push_back(uniqueVertices[vertex]);
                }
            }
        }
        return;
    }
    else {
        throw std::runtime_error("Unsupported model format: " + ext);
    }
}
