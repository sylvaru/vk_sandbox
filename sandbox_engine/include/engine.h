#pragma once
#include <memory>
#include "vulkan_wrapper/core/vulkan_renderer.h"
#include "vulkan_wrapper/vulkan_device.h"
#include "vulkan_wrapper/vulkan_instance.h"
#include "interfaces/layer_i.h"
#include "interfaces/camera_i.h"
#include "interfaces/window_i.h"
#include "common/GLFWwindowAndInput.h"
#include "asset_manager.h"
#include "physics/physics_engine.h"


namespace Core {

    struct EngineSpecification {
        std::string name = "";
        WindowSpecification windowSpec;
    };

    class SandboxEngine {
    public:
        SandboxEngine();
        explicit SandboxEngine(const EngineSpecification& appSpec);

        ~SandboxEngine();

        void initialize();
        void runApp();

        template<typename T, typename... Args>
        void pushLayer(Args&&... args) {
            static_assert(std::is_base_of<ILayer, T>::value, "T must derive from IGameLayer");
            auto layer = std::make_unique<T>(std::forward<Args>(args)...);
            // give layer a chance to access engine services
            layer->onAttach(this);
            m_layers.push_back(std::move(layer));
        }

        IWindow& getWindow() { return *m_window; }
        AssetManager& getAssetManager() { return m_assetManager; }
        VkSandboxDevice& getDevice() { return m_device; }
        VkSandboxInstance& getInstance() { return m_vkinstance; }
        ISandboxRenderer& renderer() { return m_renderer; }
        PhysicsEngine& getPhysicsEngine() { return *m_physicsEngine; }

        void toggleCursorLock();
        void setupInputCallbacks();
        void processInput();

        // Active scene API: layers can notify engine which scene should be considered active
        void setActiveScene(IScene* scene, ILayer* owner = nullptr);

        IScene* getActiveScene() const { return m_activeScene; }

        bool isCursorLocked() const { return m_cursorLocked; }

    private:
        EngineSpecification                      m_engineSpec;
        std::unique_ptr<GLFWwindowAndInput>      m_window;
        VkSandboxInstance                        m_vkinstance;
        VkSandboxDevice                          m_device;
        AssetManager                             m_assetManager;
        VkSandboxRenderer                        m_renderer;
        
        std::unique_ptr<PhysicsEngine>           m_physicsEngine;

        std::vector<std::unique_ptr<ILayer>>     m_layers;

        IScene*                                  m_activeScene = nullptr;
        ILayer*                                  m_activeSceneOwner = nullptr;

        // Misc
        bool                                     m_cursorLocked = true;
    };

 
}