#pragma once
#include <string>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <vulkan/vulkan.h>
#include <vector>
#include "vulkan_wrapper/vulkan_instance.h"
#include "interfaces/window_input_i.h"


class SandboxWindow {

public:
	SandboxWindow(int w, int h, std::string name);
	~SandboxWindow();

	bool shouldClose() { return glfwWindowShouldClose(m_pwindow); }
	VkExtent2D getExtent() { return { static_cast<uint32_t>(m_width), static_cast<uint32_t>(m_height) }; }
	bool wasWindowResized() { return m_bFramebufferResized; }
	void resetWindowResizedFlag() { m_bFramebufferResized = false; }
	GLFWwindow* getGLFWwindow() const { return m_pwindow; }

	void createSurface(VkInstance instance, VkSurfaceKHR* surface) const;
private:
	void initWindow();
	static void framebufferResizeCallback(GLFWwindow*, int, int);

	int          m_width, m_height;
	bool         m_bFramebufferResized = false;
	std::string  m_window_name;
	GLFWwindow*  m_pwindow = nullptr;

};

struct WindowUserData {
	SandboxWindow* window = nullptr;
	IWindowInput* input = nullptr;
};