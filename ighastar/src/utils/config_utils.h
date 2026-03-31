#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <utility>

namespace py = pybind11;

namespace config_utils {

// Get a nested config dict, checking both root level and under experiment_info_default
// Returns (dict, found) pair - if not found, returns empty dict with found=false
inline std::pair<py::dict, bool> get_config_dict(const py::dict& config, const char* key) {
    if (config.contains(key)) {
        return {config[key].cast<py::dict>(), true};
    } else if (config.contains("experiment_info_default")) {
        auto exp_config = config["experiment_info_default"].cast<py::dict>();
        if (exp_config.contains(key)) {
            return {exp_config[key].cast<py::dict>(), true};
        }
    }
    return {py::dict(), false};
}

}  // namespace config_utils
