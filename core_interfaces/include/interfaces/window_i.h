#pragma once
#include "key_codes.h"
#include <functional>


using SandboxKeyCallback = std::function<void(SandboxKey key, int scancode, KeyAction action, int mods)>;

struct IWindow {
	virtual ~IWindow() = default;
    virtual void* getNativeHandle() const = 0;
    virtual void lockCursor(bool lock) = 0;
    virtual void getFramebufferSize(int& width, int& height) const = 0;
    virtual bool isKeyPressed(SandboxKey key) const = 0;
    virtual bool isMouseButtonPressed(int button) const = 0;
    virtual void setUserPointer(void* ptr) = 0;
    virtual void setKeyCallback(SandboxKeyCallback callback) = 0;
    virtual bool isWindowShouldClose() const = 0;
    virtual void requestWindowClose() = 0;
    virtual void pollEvents() = 0;
    virtual void*getWindowUserPointer() const = 0;
    virtual void consumeMouseDelta(double& dx, double& dy) = 0;
    virtual bool wasWindowResized() const  = 0;
    virtual void resetWindowResizedFlag() = 0;
};