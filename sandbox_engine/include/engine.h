#pragma once
#include <memory>
#include "vulkan_wrapper/core/window.h"
#include "vulkan_wrapper/core/vulkan_renderer.h"
#include "vulkan_wrapper/vulkan_device.h"
#include "vulkan_wrapper/vulkan_instance.h"
#include "interfaces/layer_i.h"
#include "interfaces/camera_i.h"
#include "interfaces/window_input_i.h"
#include "common/glfw_input.h"
#include "asset_manager.h"
#include "physics/physics_engine.h"

#include <spdlog/spdlog.h>

namespace Core {

    struct EngineSpecification {
        std::string Name = "";

        struct WindowSpecification {
            uint32_t Width = 1920;
            uint32_t Height = 1080;
        } windowSpec;
    };

    class SandboxEngine {
    public:
       
        explicit SandboxEngine(const EngineSpecification& appSpec = EngineSpecification());
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

        AssetManager& getAssetManager() { return m_assetManager; }
        VkSandboxDevice& getDevice() { return m_device; }
        VkSandboxInstance& getInstance() { return m_vkinstance; }
        ISandboxRenderer& renderer() { return m_renderer; }

        std::shared_ptr<IWindowInput> getInputSharedPtr() { return m_windowInput; }
        std::unique_ptr<PhysicsEngine> takePhysicsEngine() { return std::move(m_physicsEngine); }

        void toggleCursorLock();
        void setupInputCallbacks();
        void processInput();

        // Active scene API: layers can notify engine which scene should be considered active
        void setActiveScene(IScene* scene, ILayer* owner = nullptr);

        IScene* getActiveScene() const { return m_activeScene; }

        const bool isCursorLocked() { return m_cursorLocked; }

    private:
        EngineSpecification                      m_engineSpec;
        SandboxWindow                            m_window;
        VkSandboxInstance                        m_vkinstance;
        VkSandboxDevice                          m_device;
        AssetManager                             m_assetManager;
        VkSandboxRenderer                        m_renderer;
        std::shared_ptr<IWindowInput>            m_windowInput;
        std::unique_ptr<PhysicsEngine>           m_physicsEngine;

        std::vector<std::unique_ptr<ILayer>>     m_layers;

        IScene*                                  m_activeScene = nullptr;
        ILayer*                                  m_activeSceneOwner = nullptr;

        // Misc
        bool                                     m_cursorLocked = true;
    };

 
}