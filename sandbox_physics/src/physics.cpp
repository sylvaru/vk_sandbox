#include "physics.h"
#include <spdlog/spdlog.h>


using namespace physx;

static inline PxVec3 glmToPx(const glm::vec3& v) { return PxVec3(v.x, v.y, v.z); }
static inline glm::vec3 pxToGlm(const PxVec3& v) { return glm::vec3(v.x, v.y, v.z); }

SandboxPhysics::SandboxPhysics() = default;

SandboxPhysics::~SandboxPhysics() {
    shutdown();
}

bool SandboxPhysics::init(int workerThreads) {
    if (m_pFoundation) return true; // already initialized

    spdlog::info("SandboxPhysics: creating foundation");

    // Default allocator / error callback types are provided by PhysX headers.
    static PxDefaultErrorCallback s_errorCallback;
    static PxDefaultAllocator s_allocatorCallback;

    m_pFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, s_allocatorCallback, s_errorCallback);
    if (!m_pFoundation) {
        spdlog::error("SandboxPhysics: PxCreateFoundation failed");
        return false;
    }

    PxTolerancesScale tolerances;
    m_pPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *m_pFoundation, tolerances);
    if (!m_pPhysics) {
        spdlog::error("SandboxPhysics: PxCreatePhysics failed");
        return false;
    }

    // CPU dispatcher
    const int threads = std::max(1, workerThreads);
    m_pDispatcher = PxDefaultCpuDispatcherCreate(threads);
    if (!m_pDispatcher) {
        spdlog::error("SandboxPhysics: PxDefaultCpuDispatcherCreate failed");
        return false;
    }

    // create scene
    PxSceneDesc sceneDesc(m_pPhysics->getTolerancesScale());
    sceneDesc.gravity = PxVec3(0.f, -9.81f, 0.f);
    sceneDesc.cpuDispatcher = m_pDispatcher;
    sceneDesc.filterShader = PxDefaultSimulationFilterShader;
    m_pScene = m_pPhysics->createScene(sceneDesc);
    if (!m_pScene) {
        spdlog::error("SandboxPhysics: createScene failed");
        return false;
    }

    // default material
    m_pMaterial = m_pPhysics->createMaterial(0.5f, 0.5f, 0.1f);
    if (!m_pMaterial) {
        spdlog::error("SandboxPhysics: createMaterial failed");
        return false;
    }

    spdlog::info("SandboxPhysics: initialized (threads={})", threads);
    return true;
}

PxRigidStatic* SandboxPhysics::createGroundPlane() {
    if (!m_pPhysics || !m_pScene) return nullptr;
    PxPlane plane(PxVec3(0, 1, 0), 0.0f); // y = 0
    PxRigidStatic* ground = PxCreatePlane(*m_pPhysics, plane, *m_pMaterial);
    if (ground) m_pScene->addActor(*ground);
    return ground;
}

bool SandboxPhysics::createDefaultScene() {
    if (!m_pPhysics || !m_pScene) {
        spdlog::error("SandboxPhysics: createDefaultScene called before initialize");
        return false;
    }

    // ground
    createGroundPlane();

    // controller manager
    m_pControllerManager = PxCreateControllerManager(*m_pScene);
    if (!m_pControllerManager) {
        spdlog::error("SandboxPhysics: PxCreateControllerManager failed");
        return false;
    }

    // capsule controller descriptor
    PxCapsuleControllerDesc desc;
    desc.radius = 0.35f;
    desc.height = 1.6f;                 // standing height without caps
    desc.position = PxExtendedVec3(0.0, 2.0, 0.0);
    desc.stepOffset = 0.25f;
    desc.slopeLimit = cosf(glm::radians(50.0f));
    desc.material = m_pMaterial;
    desc.density = 10.0f;
    desc.contactOffset = 0.1f;
    desc.reportCallback = nullptr;

    // create controller (returns PxController*)
    m_pController = m_pControllerManager->createController(desc);
    if (!m_pController) {
        spdlog::error("SandboxPhysics: createController failed");
        return false;
    }

    spdlog::info("SandboxPhysics: default scene + controller created");
    return true;
}

void SandboxPhysics::simulate(float dt) {
    if (!m_pScene) return;

    // step scene
    m_pScene->simulate(static_cast<PxReal>(dt));
    m_pScene->fetchResults(true);

    // Note: PxControllerManager does not provide updateControllers() in current PhysX 5.x API.
    // Controller behavior is driven by PxController::move() and the scene simulate/fetchResults above.
}

void SandboxPhysics::moveController(const glm::vec3& displacement, float minDistance, float elapsedTime) {
    if (!m_pController) return;
    PxControllerFilters filters; // default: no special filtering
    // move returns PxControllerCollisionFlags
    m_lastCollisionFlags = m_pController->move(glmToPx(displacement), minDistance, elapsedTime, filters);
}

void SandboxPhysics::jump(float velocity) {
    if (!m_pController) return;
    // naive immediate upward move - good starting point; consider per-frame vertical velocity for better feel
    PxControllerFilters filters;
    PxVec3 up(0.0f, velocity, 0.0f);
    m_lastCollisionFlags = m_pController->move(up, 0.001f, 0.0f, filters);
}

glm::vec3 SandboxPhysics::getControllerPosition() const {
    if (!m_pController) return glm::vec3(0.0f);
    const PxExtendedVec3& p = m_pController->getPosition();
    return glm::vec3(static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z));
}

bool SandboxPhysics::isControllerGrounded() const {
    // correct, modern test using PxFlags::isSet(...)
    return m_lastCollisionFlags.isSet(PxControllerCollisionFlag::eCOLLISION_DOWN);
}

void SandboxPhysics::shutdown() {
    // Purge controllers and release manager (purgeControllers releases controllers)
    if (m_pControllerManager) {
        m_pControllerManager->purgeControllers(); // releases controllers managed by manager
        m_pControllerManager->release();
        m_pControllerManager = nullptr;
        m_pController = nullptr; // controllers are released by purge
    }

    // Release scene (this will remove any actors)
    if (m_pScene) {
        // Do not call removeActors/removeAllActors with wrong signature; release() will clean up
        m_pScene->release();
        m_pScene = nullptr;
    }

    if (m_pDispatcher) {
        m_pDispatcher->release();
        m_pDispatcher = nullptr;
    }

    if (m_pPhysics) {
        m_pPhysics->release();
        m_pPhysics = nullptr;
    }

    if (m_pFoundation) {
        m_pFoundation->release();
        m_pFoundation = nullptr;
    }

    spdlog::info("SandboxPhysics: shutdown complete");
}