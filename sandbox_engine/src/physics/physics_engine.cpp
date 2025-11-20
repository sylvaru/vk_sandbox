#include "physics/physics_engine.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <PxPhysicsAPI.h>
#include <geometry/PxTriangleMeshGeometry.h>
#include <extensions/PxRigidActorExt.h>
#include <cooking/PxCooking.h>

using namespace physx;


static inline PxTransform glmMat4ToPxTransform(const glm::mat4& m) {
    // Extract translation
    glm::vec3 translation = glm::vec3(m[3]);

    // Extract rotation matrix (remove scale by normalizing columns)
    glm::mat3 rotMat;
    rotMat[0] = glm::normalize(glm::vec3(m[0]));
    rotMat[1] = glm::normalize(glm::vec3(m[1]));
    rotMat[2] = glm::normalize(glm::vec3(m[2]));

    // Convert to quaternion
    glm::quat q = glm::quat_cast(rotMat);

    return PxTransform(PxVec3(translation.x, translation.y, translation.z),
        PxQuat(q.x, q.y, q.z, q.w));
}


void PhysicsEngine::initPhysx() {
    // Foundation
    m_foundation = PxCreateFoundation(PX_PHYSICS_VERSION, m_allocator, m_errorCallback);
    if (!m_foundation) {
        throw std::runtime_error("PxCreateFoundation failed");
    }

    // Physics
    PxTolerancesScale toleranceScale;
    m_physics = PxCreatePhysics(PX_PHYSICS_VERSION, *m_foundation, toleranceScale);
    if (!m_physics)
        throw std::runtime_error("PxCreatePhysics failed");

    // Cooking (uses physics tolerances)
    PxCookingParams cookParams(m_physics->getTolerancesScale());

    // Dispatcher (worker threads) - choose your thread count
    m_dispatcher = PxDefaultCpuDispatcherCreate(2);

    // Scene
    PxSceneDesc sceneDesc(m_physics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
    sceneDesc.cpuDispatcher = m_dispatcher;
    sceneDesc.filterShader = PxDefaultSimulationFilterShader;
    m_scene = m_physics->createScene(sceneDesc);
    if (!m_scene) throw std::runtime_error("PxPhysics::createScene failed");

    // Controller manager (character controllers)
    m_controllerMgr = PxCreateControllerManager(*m_scene);
    if (!m_controllerMgr) throw std::runtime_error("PxCreateControllerManager failed");

    // Default material
    m_defaultMaterial = m_physics->createMaterial(0.6f, 0.6f, 0.0f);

    spdlog::info("PhysX initialized (foundation/physics/cooking/scene/controllerMgr)");
}

void PhysicsEngine::stepSimulation(float deltaTime) {
    if (!m_scene) return;

    // Simulate and fetch results synchronously
    m_scene->simulate(deltaTime);
    m_scene->fetchResults(true);
    
}

// Filters triangles for physics accuracy
CollisionMeshData PhysicsEngine::extractCollisionMesh(
    const std::vector<glm::vec3>& inVerts,
    const std::vector<uint32_t>& inIndices)
{
    CollisionMeshData out;

    out.vertices.reserve(inVerts.size());
    out.indices.reserve(inIndices.size());

    const float MIN_TRI_SIZE = 0.001f;   // Very tiny triangles removed
    const float MAX_TRI_SIZE = 20.0f;   // Too big
    const float MIN_NORMAL_Y = -0.2f;
    const float MAX_NORMAL_Y = 1.0f;

    auto triArea = [](const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
        return glm::length(glm::cross(b - a, c - a)) * 0.5f;
        };

    for (size_t i = 0; i < inIndices.size(); i += 3)
    {
        uint32_t i0 = inIndices[i];
        uint32_t i1 = inIndices[i + 1];
        uint32_t i2 = inIndices[i + 2];

        glm::vec3 a = inVerts[i0];
        glm::vec3 b = inVerts[i1];
        glm::vec3 c = inVerts[i2];

        float area = triArea(a, b, c);
        if (area < MIN_TRI_SIZE || area > MAX_TRI_SIZE)
            continue; // skip tiny or massive triangles

        glm::vec3 n = glm::normalize(glm::cross(b - a, c - a));
        if (n.y < MIN_NORMAL_Y || n.y > MAX_NORMAL_Y)
            continue; // remove weird or vertical triangles

        // Passed filters — include triangle
        uint32_t base = (uint32_t)out.vertices.size();
        out.vertices.push_back(a);
        out.vertices.push_back(b);
        out.vertices.push_back(c);

        out.indices.push_back(base + 0);
        out.indices.push_back(base + 1);
        out.indices.push_back(base + 2);
    }

    return out;
}
PxRigidStatic* PhysicsEngine::createStaticTriangleMesh(
    const std::vector<glm::vec3>& vertices,
    const std::vector<uint32_t>& indices,
    const glm::mat4& transform)
{
    if (!m_physics) {
        spdlog::error("PhysicsEngine::createStaticTriangleMesh failed: m_physics is null");
        return nullptr;
    }

    // Extract scale from the transform
    glm::vec3 scale(
        glm::length(glm::vec3(transform[0])),
        glm::length(glm::vec3(transform[1])),
        glm::length(glm::vec3(transform[2]))
    );

    // Bake scale into vertices
    std::vector<PxVec3> pxVerts;
    pxVerts.reserve(vertices.size());
    for (const auto& v : vertices)
        pxVerts.emplace_back(v.x * scale.x, v.y * scale.y, v.z * scale.z);

    PxTriangleMeshDesc desc{};
    desc.points.count = static_cast<PxU32>(pxVerts.size());
    desc.points.stride = sizeof(PxVec3);
    desc.points.data = pxVerts.data();

    desc.triangles.count = static_cast<PxU32>(indices.size() / 3);
    desc.triangles.stride = sizeof(uint32_t) * 3;
    desc.triangles.data = indices.data();

    // Cooking
    PxDefaultMemoryOutputStream outBuf;
    PxTriangleMeshCookingResult::Enum result;
    PxCookingParams params(m_physics->getTolerancesScale());
    params.suppressTriangleMeshRemapTable = true;

    if (!PxCookTriangleMesh(params, desc, outBuf, &result)) {
        spdlog::error("PxCookTriangleMesh failed (ok=false). MeshVerts={}, MeshTris={}",
            desc.points.count, desc.triangles.count);
        return nullptr;
    }

    if (result != PxTriangleMeshCookingResult::eSUCCESS) {
        spdlog::error("PxCookTriangleMesh returned error result: {}", static_cast<int>(result));
        return nullptr;
    }

    PxDefaultMemoryInputData inBuf(outBuf.getData(), outBuf.getSize());
    PxTriangleMesh* mesh = m_physics->createTriangleMesh(inBuf);
    if (!mesh) {
        spdlog::error("createTriangleMesh failed after cooking.");
        return nullptr;
    }

    // Create PxRigidStatic with translation + rotation
    PxTransform pxT = glmMat4ToPxTransform(transform);
    PxRigidStatic* actor = m_physics->createRigidStatic(pxT);
    if (!actor) {
        spdlog::error("createRigidStatic failed.");
        return nullptr;
    }

    PxTriangleMeshGeometry geom(mesh);
    PxShape* shape = m_physics->createShape(geom, *m_defaultMaterial, true);
    if (!shape) {
        spdlog::error("createShape failed for PxTriangleMeshGeometry.");
        return nullptr;
    }

    actor->attachShape(*shape);
    shape->release();
    m_scene->addActor(*actor);

    spdlog::info("Successfully created static PhysX triangle mesh. Verts={}, Tris={}",
        desc.points.count, desc.triangles.count);

    return actor;
}


void PhysicsEngine::createFPScontroller(const glm::vec3& startPos, float radius, float height) {
    if (!m_controllerMgr || !m_defaultMaterial) {
        spdlog::error("Controller manager or material not initialized");
        return;
    }

    PxCapsuleControllerDesc desc;
    desc.setToDefault();
    desc.height = height;
    desc.radius = radius;
    desc.position = PxExtendedVec3(startPos.x, startPos.y, startPos.z);
    desc.material = m_defaultMaterial;
    desc.upDirection = PxVec3(0, 1, 0);
    desc.slopeLimit = 0.707f;
    desc.contactOffset = 0.02f;
    desc.stepOffset = 0.3f;
    desc.nonWalkableMode = PxControllerNonWalkableMode::ePREVENT_CLIMBING;

    PxController* c = m_controllerMgr->createController(desc);
    if (!c) {
        spdlog::error("Failed to create capsule controller");
        m_fpsController = nullptr;
        return;
    }

 
    m_fpsController = static_cast<PxCapsuleController*>(c);
    spdlog::info("FPS controller created at ({:.2f}, {:.2f}, {:.2f})", startPos.x, startPos.y, startPos.z);
}

void PhysicsEngine::moveFPSController(const glm::vec3& displacement, float deltaTime) {
    if (!m_fpsController) return;

    PxVec3 move(displacement.x, displacement.y, displacement.z);
    const float minDist = 0.001f;
    m_fpsController->move(move, minDist, deltaTime, nullptr, nullptr);
}

glm::vec3 PhysicsEngine::getFPScontrollerPosition() const {
    if (!m_fpsController) return glm::vec3(0.0f);
    PxExtendedVec3 p = m_fpsController->getPosition();
    return glm::vec3((float)p.x, (float)p.y, (float)p.z);
}

PhysicsEngine::~PhysicsEngine() {
    if (m_fpsController) {
        m_fpsController->release();
        m_fpsController = nullptr;
    }
    if (m_controllerMgr) {
        m_controllerMgr->purgeControllers();
        m_controllerMgr->release();
        m_controllerMgr = nullptr;
    }
    if (m_scene) {
        m_scene->release();
        m_scene = nullptr;
    }
    if (m_dispatcher) {
        m_dispatcher->release();
        m_dispatcher = nullptr;
    }
    if (m_physics) {
        m_physics->release();
        m_physics = nullptr;
    }
    if (m_foundation) {
        m_foundation->release();
        m_foundation = nullptr;
    }
}