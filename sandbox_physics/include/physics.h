#pragma once
#include <PxPhysicsAPI.h>

#include <glm/glm.hpp>
#include <memory>


class SandboxPhysics {
public:
	SandboxPhysics();
	~SandboxPhysics();

	bool init(int workerThreads = 2);

	bool createDefaultScene();

	void simulate(float dt);

	void moveController(const glm::vec3& displacement, float minDistance = 0.001f, float elapsedTime = 0.0f);
	// Jump: give upward impulse (works by moving controller vertically)
	void jump(float velocity);

	// Get controller world position (center)
	glm::vec3 getControllerPosition() const;

	// Ground check (true if controller flags indicate grounded)
	bool isControllerGrounded() const;

	// Shutdown & free PhysX objects
	void shutdown();
private:

	physx::PxFoundation* m_pFoundation = nullptr;
	physx::PxPhysics* m_pPhysics = nullptr;
	physx::PxDefaultCpuDispatcher* m_pDispatcher = nullptr;
	physx::PxScene* m_pScene = nullptr;
	physx::PxMaterial* m_pMaterial = nullptr;

	// Controller manager + controller
	physx::PxControllerManager* m_pControllerManager = nullptr;
	physx::PxController* m_pController = nullptr;

	// Keep last move/flags for queries
	physx::PxControllerCollisionFlags       m_lastCollisionFlags;

	// internal helpers
	physx::PxRigidStatic* createGroundPlane();

};