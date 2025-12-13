#pragma once
#include "interfaces/window_i.h"
#include "GLFW/glfw3.h"
#include "imgui.h"

class GLFWwindowAndInput final : public IWindow {
public:
    GLFWwindowAndInput(const WindowSpecification& spec, const std::string& title);
    ~GLFWwindowAndInput();
    // Window
    void pollEvents() override;
    void waitEvents() override;
    bool isWindowShouldClose() const override;
    void requestWindowClose() override;
    void getFramebufferSize(int& width, int& height) const override;

    bool wasWindowResized() const { return m_framebufferResized; }
    void resetWindowResizedFlag() { m_framebufferResized = false; }

    void setUserPointer(void* ptr) override;
    void* getWindowUserPointer() const override;

    // Native access
    void* getNativeHandle() const override { return m_window; }
    GLFWwindow* nativeGLFW() const { return m_window; }

    // Input
    void lockCursor(bool lock) override;
    bool isCursorLocked() const override { return m_cursorLocked; }
    bool isKeyPressed(SandboxKey key) const override;
    bool isMouseButtonPressed(int button) const override;
    void consumeMouseDelta(double& dx, double& dy) override;
    void setKeyCallback(SandboxKeyCallback callback) override;

private:
    void initGLFW(const WindowSpecification& spec, const std::string& title);
    int mapKeyToGLFW(SandboxKey key) const;

    static void framebufferResizeCallback(GLFWwindow* window, int w, int h);
    static void cursorPosCallback(GLFWwindow* window, double x, double y);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

private:
    GLFWwindow* m_window = nullptr;

    uint32_t m_width = 0;
    uint32_t m_height = 0;
    bool m_framebufferResized = false;
    bool m_cursorLocked = false;

    // mouse
    double m_lastX = 0.0;
    double m_lastY = 0.0;
    double m_accumDX = 0.0;
    double m_accumDY = 0.0;
    bool m_firstMouse = true;

    SandboxKeyCallback m_keyCallback;


};
