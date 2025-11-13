#pragma once
#include <memory>

class btBroadphaseInterface;
class btDefaultCollisionConfiguration;
class btCollisionDispatcher;
class btSequentialImpulseConstraintSolver;
class btDiscreteDynamicsWorld;
class btRigidBody;
class btCapsuleShape;

class PhysicsEngine {
public:
    PhysicsEngine();
    ~PhysicsEngine();

    void stepSimulation(float deltaTime);

    btDiscreteDynamicsWorld* getWorld() const { return m_dynamicsWorld.get(); }

private:
    std::unique_ptr<btBroadphaseInterface> m_broadphase;
    std::unique_ptr<btDefaultCollisionConfiguration> m_collisionConfiguration;
    std::unique_ptr<btCollisionDispatcher> m_dispatcher;
    std::unique_ptr<btSequentialImpulseConstraintSolver> m_solver;
    std::unique_ptr<btDiscreteDynamicsWorld> m_dynamicsWorld;
};