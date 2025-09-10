#pragma once
#include <memory>
#include "window.h"
#include "vulkan_wrapper/vulkan_device.h"
#include "vulkan_wrapper/vulkan_instance.h"
#include "vulkan_wrapper/vulkan_renderer.h"
#include "interfaces/game_layer_i.h"
#include "interfaces/camera_i.h"
#include "interfaces/window_input_i.h"
#include "common/glfw_input.h"
#include "asset_manager.h"



class SandboxEngine {
public:
    static constexpr uint32_t WIDTH = 1440;
    static constexpr uint32_t HEIGHT = 810;

    SandboxEngine();
    ~SandboxEngine();

    void initialize();

    void initLayer(IGameLayer* game);

    void run(std::unique_ptr<IGameLayer> game);
private:
    SandboxWindow                 m_window;
    VkSandboxInstance             m_vkinstance;
    VkSandboxDevice               m_device;
    AssetManager                  m_assetManager;
    VkSandboxRenderer             m_renderer;
    std::shared_ptr<IWindowInput> m_windowInput;
 

public:
    std::shared_ptr<IWindowInput> getInputSharedPtr() {
        return m_windowInput;
    }
    AssetManager& getAssetManager() { return m_assetManager; }

    ISandboxRenderer& renderer();
    void toggleCursorLock();
    void setupInputCallbacks();
    void processInput();

    bool m_cursorLocked = true;


};
