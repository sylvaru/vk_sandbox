#pragma once
#include <memory>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <unordered_map>
#include <spdlog/spdlog.h>
#include <PxPhysicsAPI.h>
#include <cooking/PxCooking.h>



struct CollisionMeshData {
    std::vector<glm::vec3> vertices;
    std::vector<uint32_t> indices;
};


class PhysicsEngine {
public:
    PhysicsEngine() = default;
    ~PhysicsEngine();
    PhysicsEngine(const PhysicsEngine&) = delete;
    PhysicsEngine& operator=(const PhysicsEngine&) = delete;
    PhysicsEngine(PhysicsEngine&&) = delete;
    PhysicsEngine& operator=(PhysicsEngine&&) = delete;

    void initPhysx();
    void stepSimulation(float deltaTime);

    physx::PxRigidStatic* createStaticTriangleMesh(
        const std::vector<glm::vec3>& vertices,
        const std::vector<uint32_t>& indices,
        const glm::mat4& transform);

    CollisionMeshData extractCollisionMesh(
        const std::vector<glm::vec3>& inVerts,
        const std::vector<uint32_t>& inIndices);


    void createFPScontroller(const glm::vec3& startPos, float radius = 0.4f, float height = 1.8f);
    void moveFPSController(const glm::vec3& displacement, float deltaTime);
    glm::vec3 getFPScontrollerPosition() const;
    physx::PxCapsuleController* getFPScontroller() const { return m_fpsController; }

private:

    physx::PxFoundation* m_foundation = nullptr;
    physx::PxPhysics* m_physics = nullptr;
    physx::PxScene* m_scene = nullptr;
    physx::PxDefaultCpuDispatcher* m_dispatcher = nullptr;

    physx::PxControllerManager* m_controllerMgr = nullptr;
    physx::PxCapsuleController* m_fpsController = nullptr;

    physx::PxMaterial* m_defaultMaterial = nullptr;

    physx::PxDefaultAllocator m_allocator;
    physx::PxDefaultErrorCallback m_errorCallback;
};