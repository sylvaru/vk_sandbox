#include "common/glfw_input.h"

GLFWWindowInput::GLFWWindowInput(GLFWwindow* window)
    : m_pwindow(window) {
    if (glfwRawMouseMotionSupported()) {
        glfwSetInputMode(m_pwindow, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }
    glfwSetKeyCallback(m_pwindow, internalKeyCallback);

    glfwSetCursorPosCallback(
        m_pwindow,
        GLFWWindowInput::cursorPosCallbackStatic);
}

void GLFWWindowInput::lockCursor(bool lock) {
    glfwSetInputMode(m_pwindow, GLFW_CURSOR, lock ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    m_firstMouse = true;
}

// static callback called by GLFW
void GLFWWindowInput::cursorPosCallbackStatic(GLFWwindow* window, double x, double y) {
    auto* userData = static_cast<WindowUserData*>(glfwGetWindowUserPointer(window));
    if (!userData || !userData->input) return;

    auto* self = static_cast<GLFWWindowInput*>(userData->input);

    if (self->m_firstMouse) {
        self->m_lastX = x;
        self->m_lastY = y;
        self->m_firstMouse = false;
        return;
    }

    double dx = x - self->m_lastX;
    double dy = y - self->m_lastY;

    self->m_lastX = x;
    self->m_lastY = y;

    self->m_accumDX += dx;
    self->m_accumDY += dy;
}


void GLFWWindowInput::consumeMouseDelta(double& dx, double& dy) {
    dx = m_accumDX;
    dy = m_accumDY;
    m_accumDX = 0.0f;
    m_accumDY = 0.0f;
}
void GLFWWindowInput::getFramebufferSize(int& width, int& height) const {
    glfwGetFramebufferSize(m_pwindow, &width, &height);
}

bool GLFWWindowInput::isKeyPressed(SandboxKey key) const {
    int glfwKey = mapKeyToGLFW(key);
    if (glfwKey == -1) return false;
    return glfwGetKey(m_pwindow, glfwKey) == GLFW_PRESS;
}

bool GLFWWindowInput::isMouseButtonPressed(int button) const {
    return glfwGetMouseButton(m_pwindow, button) == GLFW_PRESS;
}

int GLFWWindowInput::mapKeyToGLFW(SandboxKey key) const {
    switch (key) {
    case SandboxKey::W: return GLFW_KEY_W;
    case SandboxKey::A: return GLFW_KEY_A;
    case SandboxKey::S: return GLFW_KEY_S;
    case SandboxKey::D: return GLFW_KEY_D;
    case SandboxKey::Q: return GLFW_KEY_Q;
    case SandboxKey::E: return GLFW_KEY_E;
    case SandboxKey::ESCAPE: return GLFW_KEY_ESCAPE;
    case SandboxKey::SPACE: return GLFW_KEY_SPACE;
    case SandboxKey::LEFT_ALT: return GLFW_KEY_LEFT_ALT;
    case SandboxKey::LEFT_SHIFT: return GLFW_KEY_LEFT_SHIFT;
        // Add other keys as needed
    default: return -1; // Unknown/unmapped key
    }
}