#include "common/renderer_pch.h"
#include "vulkan_wrapper/vulkan_instance.h"
#include <GLFW/glfw3.h>


VkSandboxInstance::VkSandboxInstance() {
    createInstance();
    if constexpr (enableValidationLayers) {
        setupDebugMessenger();
    }
}

VkSandboxInstance::~VkSandboxInstance() {
    if constexpr (enableValidationLayers) {
        destroyDebugMessenger();
    }
    if (m_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }
}

void VkSandboxInstance::createInstance() {
    // 1) app info
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Sandbox App";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "SandboxEngine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    // 2) extensions
    auto exts = getRequiredExtensions();
    // 3) layers
    if constexpr (enableValidationLayers) {
        if (!checkValidationLayerSupport())
            throw std::runtime_error("Validation layers requested, but not available");
    }

    // 4) instance create
    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &appInfo;
    ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    ci.ppEnabledExtensionNames = exts.data();
    if constexpr (enableValidationLayers) {
        ci.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        ci.ppEnabledLayerNames = validationLayers.data();
    }
    else {
        ci.enabledLayerCount = 0;
    }

    if (vkCreateInstance(&ci, nullptr, &m_instance) != VK_SUCCESS)
        throw std::runtime_error("failed to create Vulkan instance");
}

std::vector<const char*> VkSandboxInstance::getRequiredExtensions() {
    uint32_t count = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&count);
    std::vector<const char*> exts(glfwExts, glfwExts + count);
    if constexpr (enableValidationLayers) {
        exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    return exts;
}

bool VkSandboxInstance::checkValidationLayerSupport() {
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> available(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, available.data());

    for (auto layerName : validationLayers) {
        bool found = false;
        for (auto& prop : available) {
            if (std::strcmp(prop.layerName, layerName) == 0) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

//— debug messenger setup/teardown —

void VkSandboxInstance::setupDebugMessenger() {
    VkDebugUtilsMessengerCreateInfoEXT ci{};
    populateDebugMessengerCreateInfo(ci);
    if (CreateDebugUtilsMessengerEXT(&ci, &m_debugMessenger) != VK_SUCCESS)
        throw std::runtime_error("failed to set up debug messenger");
}

void VkSandboxInstance::destroyDebugMessenger() {
    if (m_debugMessenger != VK_NULL_HANDLE)
        DestroyDebugUtilsMessengerEXT(m_debugMessenger);
}

void VkSandboxInstance::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& ci) {
    ci = {};
    ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
        | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
        | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
        | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
        | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = debugCallback;
    ci.pUserData = nullptr;
}

VKAPI_ATTR VkBool32 VKAPI_CALL VkSandboxInstance::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT  sev,
    VkDebugUtilsMessageTypeFlagsEXT         type,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* userData)
{
    std::cerr << "[Validation] " << data->pMessage << "\n";
    return VK_FALSE;
}

VkResult VkSandboxInstance::CreateDebugUtilsMessengerEXT(
    const VkDebugUtilsMessengerCreateInfoEXT* ci,
    VkDebugUtilsMessengerEXT* messenger)
{
    auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT")
        );
    if (func) return func(m_instance, ci, nullptr, messenger);
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void VkSandboxInstance::DestroyDebugUtilsMessengerEXT(VkDebugUtilsMessengerEXT messenger) {
    auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT")
        );
    if (func) func(m_instance, messenger, nullptr);
}