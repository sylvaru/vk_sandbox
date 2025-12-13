#pragma once
#include <glm/glm.hpp>

#define MAX_NUM_JOINTS 128u

struct alignas(16) ShaderMeshData {
	glm::mat4 matrix;
	glm::mat4 jointMatrix[MAX_NUM_JOINTS]{};
	uint32_t jointCount{ 0 };
};

struct alignas(16) ShaderMaterial {
	glm::vec4 baseColorFactor;
	glm::vec4 emissiveFactor;
	glm::vec4 diffuseFactor;
	glm::vec4 specularFactor;
	float workflow;
	int colorTextureSet;
	int PhysicalDescriptorTextureSet;
	int normalTextureSet;
	int occlusionTextureSet;
	int emissiveTextureSet;
	float metallicFactor;
	float roughnessFactor;
	float alphaMask;
	float alphaMaskCutoff;
	float emissiveStrength;
};

struct MeshPushConstantBlock {
	int32_t meshIndex;
	int32_t materialIndex;
};

struct alignas(16) MaterialPushConstants {
	glm::vec4 params0; // x=albedoIntensity, y=metallicMultiplier, z=roughnessMultiplier, w=aoMultiplier
	glm::vec4 params1; // x=normalStrength, y=iblIntensity, z=directLightIntensity, w=emissiveIntensity
	float exposure;

	float _pad0;
	float _pad1;
	float _pad2;
};