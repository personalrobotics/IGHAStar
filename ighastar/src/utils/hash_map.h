// Backend-switched unordered map/set for IGHA*.
//
// Default: std::unordered_{map,set}.
// Optional: robin_hood (MIT) when built with -DIGHA_USE_ROBIN_HOOD and
//   robin_hood.h on the include path (common_utils.py / ROBIN_HOOD_DIR).
// Force std even if robin_hood is present: IGHA_NO_ROBIN_HOOD=1 in the env.
//
// Prefer igha::unordered_map / igha::unordered_set in planner code.
#pragma once

#ifdef IGHA_USE_ROBIN_HOOD
#include <robin_hood.h>
namespace igha {
template <class Key, class T>
using unordered_map = robin_hood::unordered_map<Key, T>;
template <class Key>
using unordered_set = robin_hood::unordered_set<Key>;
} // namespace igha
#else
#include <unordered_map>
#include <unordered_set>
namespace igha {
template <class Key, class T>
using unordered_map = std::unordered_map<Key, T>;
template <class Key>
using unordered_set = std::unordered_set<Key>;
} // namespace igha
#endif
