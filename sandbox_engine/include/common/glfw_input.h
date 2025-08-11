#pragma once
#include "interfaces/window_input_i.h"
#include "GLFW/glfw3.h"
#include "window.h"


class GLFWWindowInput : public IWindowInput {
public:
    explicit GLFWWindowInput(GLFWwindow* window);

    void lockCursor(bool lock) override;
    void setCursorCallback(void (*callback)(double, double)) override;
    void getFramebufferSize(int& width, int& height) const override;
    bool isKeyPressed(SandboxKey key) const override;
    bool isMouseButtonPressed(int button) const override;
    void getMouseDelta(double& dx, double& dy) override;

    void* getWindowUserPointer()const override {
        return glfwGetWindowUserPointer(m_pwindow);
    }


    inline void setUserPointer(void* ptr) override{
        glfwSetWindowUserPointer(m_pwindow, ptr);
    }

    inline void setKeyCallback(SandboxKeyCallback callback) override {
        m_keyCallback = std::move(callback);
    }

    bool isWindowShouldClose() const override{
        return glfwWindowShouldClose(m_pwindow);
    }

    void requestWindowClose() {
        glfwSetWindowShouldClose(m_pwindow, GLFW_TRUE);
    }

    void pollEvents()override {
        glfwPollEvents();
    }
    void setGLFWwindow(GLFWwindow* window) {
        m_pwindow = window;
    }
private:
    int mapKeyToGLFW(SandboxKey key) const;
    static void cursorPosCallbackStatic(GLFWwindow* window, double x, double y);
    GLFWwindow* m_pwindow;
    SandboxKeyCallback m_keyCallback;

    mutable double m_lastX = 0.0;
    mutable double m_lastY = 0.0;
    mutable bool m_firstMouse = true;



    void (*m_cursorCallback)(double, double) = nullptr;

    static void internalKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        auto* userData = static_cast<WindowUserData*>(glfwGetWindowUserPointer(window));
        if (!userData || !userData->input) return;

        auto* input = userData->input;
        auto* self = dynamic_cast<GLFWWindowInput*>(input);  // safe if needed, or static_cast if you're sure
        if (!self || !self->m_keyCallback) return;

        SandboxKey sandboxKey = static_cast<SandboxKey>(key);
        KeyAction sandboxAction =
            action == GLFW_PRESS ? KeyAction::PRESS :
            action == GLFW_RELEASE ? KeyAction::RELEASE :
            KeyAction::REPEAT;

        self->m_keyCallback(sandboxKey, scancode, sandboxAction, mods);
    }




};