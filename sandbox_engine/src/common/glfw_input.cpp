#include "common/glfw_input.h"

GLFWWindowInput::GLFWWindowInput(GLFWwindow* window)
    : m_pwindow(window) {
    if (glfwRawMouseMotionSupported()) {
        glfwSetInputMode(m_pwindow, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }
    glfwSetKeyCallback(m_pwindow, internalKeyCallback);
}

void GLFWWindowInput::lockCursor(bool lock) {
    glfwSetInputMode(m_pwindow, GLFW_CURSOR, lock ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    m_firstMouse = true;
}

void GLFWWindowInput::setCursorCallback(void (*callback)(double, double)) {
    m_cursorCallback = callback;
    glfwSetCursorPosCallback(m_pwindow, GLFWWindowInput::cursorPosCallbackStatic);
}



// static callback called by GLFW
void GLFWWindowInput::cursorPosCallbackStatic(GLFWwindow* window, double x, double y) {
    auto* userData = static_cast<WindowUserData*>(glfwGetWindowUserPointer(window));
    if (!userData || !userData->input) return;

    auto* input = userData->input;
    auto* self = dynamic_cast<GLFWWindowInput*>(input);  // cast only if needed
    if (self && self->m_cursorCallback)
        self->m_cursorCallback(x, y);
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

void GLFWWindowInput::getMouseDelta(double& dx, double& dy) {
    double xpos, ypos;
    glfwGetCursorPos(m_pwindow, &xpos, &ypos);

    if (glfwGetInputMode(m_pwindow, GLFW_CURSOR) != GLFW_CURSOR_DISABLED) {
        m_lastX = xpos;
        m_lastY = ypos;
        dx = 0.0;
        dy = 0.0;
        return;
    }

    if (m_firstMouse) {
        m_lastX = xpos;
        m_lastY = ypos;
        m_firstMouse = false;
    }

    dx = xpos - m_lastX;
    dy = ypos - m_lastY;

    m_lastX = xpos;
    m_lastY = ypos;
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