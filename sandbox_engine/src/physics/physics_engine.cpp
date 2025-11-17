#include "physics/physics_engine.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <PxPhysicsAPI.h>
#include <iostream>

static physx::PxDefaultAllocator gAllocator;
static physx::PxDefaultErrorCallback gErrorCallback;

physx::PxFoundation* gFoundation = nullptr;
physx::PxPhysics* gPhysics = nullptr;



PhysicsEngine::PhysicsEngine() {
   
}

void PhysicsEngine::stepSimulation(float deltaTime) {

}


void PhysicsEngine::initPhysx() {
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, physx::PxTolerancesScale());

	if (!gPhysics)
		throw std::runtime_error("Failed to initialize PhysX!");
}