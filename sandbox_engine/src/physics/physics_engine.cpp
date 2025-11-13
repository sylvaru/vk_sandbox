#include <btBulletDynamicsCommon.h>
#include "physics/physics_engine.h"
#include <spdlog/spdlog.h>


PhysicsEngine::PhysicsEngine() {
    m_broadphase = std::make_unique<btDbvtBroadphase>();
    m_collisionConfiguration = std::make_unique<btDefaultCollisionConfiguration>();
    m_dispatcher = std::make_unique<btCollisionDispatcher>(m_collisionConfiguration.get());
    m_solver = std::make_unique<btSequentialImpulseConstraintSolver>();

    m_dynamicsWorld = std::make_unique<btDiscreteDynamicsWorld>(
        m_dispatcher.get(),
        m_broadphase.get(),
        m_solver.get(),
        m_collisionConfiguration.get()
    );

    // Gravity (tweak as needed for your world)
    m_dynamicsWorld->setGravity(btVector3(0.0f, -9.81f, 0.0f));

    spdlog::info("Bullet Physics Engine initialized");
}

PhysicsEngine::~PhysicsEngine() {
    spdlog::info("Bullet Physics Engine shutting down");
    // Bullet smart pointers handle cleanup, but you might clear any rigid bodies here if owned externally
}

void PhysicsEngine::stepSimulation(float deltaTime) {
    if (m_dynamicsWorld) {
        // Cap max substeps for stability
        m_dynamicsWorld->stepSimulation(deltaTime, 10, 1.0f / 240.0f);
    }
}