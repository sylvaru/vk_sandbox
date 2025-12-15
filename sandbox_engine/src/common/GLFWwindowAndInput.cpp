// sandbox_engine/common/glfw_input.cpp
#include "common/engine_pch.h"
#include "common/GLFWwindowAndInput.h"


GLFWwindowAndInput::GLFWwindowAndInput(
    const WindowSpecification& spec,
    const std::string& title)
{
    initGLFW(spec, title);
}

GLFWwindowAndInput::~GLFWwindowAndInput()
{
    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    glfwTerminate();
}

void GLFWwindowAndInput::initGLFW(
    const WindowSpecification& spec,
    const std::string& title)
{
    if (!glfwInit())
        throw std::runtime_error("Failed to initialize GLFW");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    GLFWmonitor* monitor = nullptr;
    int width = static_cast<int>(spec.width);
    int height = static_cast<int>(spec.height);

    switch (spec.mode) {
    case WindowMode::Windowed:
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        monitor = nullptr;
        break;

    case WindowMode::BorderlessFullscreen: {
        glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        width = mode->width;
        height = mode->height;
        break;
    }
    case WindowMode::ExclusiveFullscreen: {
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);

        width = mode->width;
        height = mode->height;

        // Optional but recommended for true exclusive
        glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
        break;
    }
    }


    m_window = glfwCreateWindow(
        width,
        height,
        title.c_str(),
        monitor,
        nullptr
    );

    if (!m_window)
        throw std::runtime_error("Failed to create GLFW window");

    m_width = width;
    m_height = height;

    glfwSetWindowUserPointer(m_window, this);

    glfwSetFramebufferSizeCallback(m_window, framebufferResizeCallback);
    glfwSetCursorPosCallback(m_window, cursorPosCallback);
    glfwSetKeyCallback(m_window, keyCallback);

    if (glfwRawMouseMotionSupported()) {
        glfwSetInputMode(m_window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
    }
}

// Window API
void GLFWwindowAndInput::pollEvents() {
    glfwPollEvents();
}
void GLFWwindowAndInput::waitEvents() {
    glfwWaitEvents();
}

bool GLFWwindowAndInput::isWindowShouldClose() const {
    return glfwWindowShouldClose(m_window);
}

void GLFWwindowAndInput::requestWindowClose() {
    glfwSetWindowShouldClose(m_window, GLFW_TRUE);
    std::exit(0);
}

void GLFWwindowAndInput::getFramebufferSize(int& width, int& height) const {
    glfwGetFramebufferSize(m_window, &width, &height);
}

void GLFWwindowAndInput::setUserPointer(void* ptr) {
    glfwSetWindowUserPointer(m_window, ptr);
}

void* GLFWwindowAndInput::getWindowUserPointer() const {
    return glfwGetWindowUserPointer(m_window);
}


// Input API
void GLFWwindowAndInput::lockCursor(bool lock) {
    m_cursorLocked = lock;
    glfwSetInputMode(
        m_window,
        GLFW_CURSOR,
        lock ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL
    );
    m_firstMouse = true;
}

bool GLFWwindowAndInput::isKeyPressed(SandboxKey key) const {
    int glfwKey = mapKeyToGLFW(key);
    return glfwKey != -1 && glfwGetKey(m_window, glfwKey) == GLFW_PRESS;
}

bool GLFWwindowAndInput::isMouseButtonPressed(int button) const {
    return glfwGetMouseButton(m_window, button) == GLFW_PRESS;
}

void GLFWwindowAndInput::consumeMouseDelta(double& dx, double& dy) {
    dx = m_accumDX;
    dy = m_accumDY;
    m_accumDX = 0.0;
    m_accumDY = 0.0;
}



// Callbacks

void GLFWwindowAndInput::setKeyCallback(SandboxKeyCallback callback) {
    m_keyCallback = std::move(callback);
}

void GLFWwindowAndInput::framebufferResizeCallback(GLFWwindow* window, int w, int h) {
    auto* self = static_cast<GLFWwindowAndInput*>(glfwGetWindowUserPointer(window));
    if (!self) return;

    self->m_framebufferResized = true;
    self->m_width = static_cast<uint32_t>(w);
    self->m_height = static_cast<uint32_t>(h);
}

void GLFWwindowAndInput::cursorPosCallback(GLFWwindow* window, double x, double y) {
    auto* self = static_cast<GLFWwindowAndInput*>(glfwGetWindowUserPointer(window));
    if (!self || !self->m_cursorLocked) return;

    if (self->m_firstMouse) {
        self->m_lastX = x;
        self->m_lastY = y;
        self->m_firstMouse = false;
        return;
    }

    self->m_accumDX += (x - self->m_lastX);
    self->m_accumDY += (y - self->m_lastY);

    self->m_lastX = x;
    self->m_lastY = y;
}

void GLFWwindowAndInput::keyCallback(
    GLFWwindow* window,
    int key,
    int scancode,
    int action,
    int mods)
{
    auto* self = static_cast<GLFWwindowAndInput*>(glfwGetWindowUserPointer(window));
    if (!self || !self->m_keyCallback) return;

    SandboxKey sandboxKey = static_cast<SandboxKey>(key);
    KeyAction sandboxAction =
        action == GLFW_PRESS ? KeyAction::PRESS :
        action == GLFW_RELEASE ? KeyAction::RELEASE :
        KeyAction::REPEAT;

    self->m_keyCallback(sandboxKey, scancode, sandboxAction, mods);
}

int GLFWwindowAndInput::mapKeyToGLFW(SandboxKey key) const {
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
    default: return -1;
    }
}