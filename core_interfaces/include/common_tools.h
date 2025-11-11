#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <optional>



// Helper: lower-case extension
static std::string get_file_ext_lower(const std::string& filename) {
    std::string ext = std::filesystem::path(filename).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return ext;
}

static std::optional<std::string> find_existing_path(
    const std::string& relPath,
    const std::vector<std::string>& candidatePrefixes)
{
    namespace fs = std::filesystem;

    // If relPath is absolute and exists, return it
    fs::path p = relPath;
    if (p.is_absolute()) {
        if (fs::exists(p)) return p.string();
    }

    // Helper: check direct candidate (prefix + relPath)
    for (const auto& prefix : candidatePrefixes) {
        fs::path tryPath = fs::path(prefix) / relPath;
        spdlog::debug(" try direct: {}", tryPath.string());
        if (fs::exists(tryPath)) return tryPath.string();
    }

    // Try PROJECT_ROOT_DIR + relPath
    fs::path rootTry = fs::path(PROJECT_ROOT_DIR) / relPath;
    spdlog::debug(" try direct: {}", rootTry.string());
    if (fs::exists(rootTry)) return rootTry.string();

    // If direct attempts failed, attempt recursive search for the basename inside candidate prefixes.
    const std::string basename = fs::path(relPath).filename().string();
    if (basename.empty()) return std::nullopt;

    spdlog::debug(" Starting recursive search for '{}' under candidates", basename);

    for (const auto& prefix : candidatePrefixes) {
        fs::path dir(prefix);
        if (!fs::exists(dir) || !fs::is_directory(dir)) {
            spdlog::debug(" skip recursive search, not a dir: {}", dir.string());
            continue;
        }

        try {
            // Limit recursion by checking depth implicitly via iterator options if needed in future
            for (auto it = fs::recursive_directory_iterator(dir); it != fs::recursive_directory_iterator(); ++it) {
                if (!it->is_regular_file()) continue;
                if (it->path().filename() == basename) {
                    spdlog::info(" Found '{}' at {}", basename, it->path().string());
                    return it->path().string();
                }
            }
        }
        catch (const std::exception& e) {
            spdlog::warn(" recursive search under '{}' failed: {}", prefix, e.what());
            // continue to next candidate
        }
    }

    // Last-resort: search entire project root (expensive)
    try {
        fs::path root(PROJECT_ROOT_DIR);
        if (fs::exists(root) && fs::is_directory(root)) {
            spdlog::debug(" Searching project root '{}' for '{}'", root.string(), basename);
            for (auto it = fs::recursive_directory_iterator(root); it != fs::recursive_directory_iterator(); ++it) {
                if (!it->is_regular_file()) continue;
                if (it->path().filename() == basename) {
                    spdlog::info(" Found '{}' under project root at {}", basename, it->path().string());
                    return it->path().string();
                }
            }
        }
    }
    catch (const std::exception& e) {
        spdlog::warn(" project-root recursive search failed: {}", e.what());
    }

    return std::nullopt;
}
