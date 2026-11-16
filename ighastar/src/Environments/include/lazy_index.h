#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

// Multi-resolution cell hashes of one node, computed the first time a level is asked for.
//
// Level l hashes the pose at resolution / division_factor^l. The planner reads index[level] for the level it is
// searching at and for every level below it, so a node only ever pays for the levels the search has reached.
// There is no limit on the number of levels: the search moves up one level each time the frontier at the current
// level is exhausted.
//
// The resolution is divided in float, once per level, in order (the same values an eager precompute would give),
// so the hashes are identical to building every level up front.
//
// NOT THREAD SAFE: operator[] writes to the node (v, res). In bidirectional search one worker also reads the
// other worker's nodes for the near-meet check. That read only asks for a level the node already has (the check does
// not move the search up a level), so nothing is written and no race happens in practice. If a change ever lets
// that read reach a new level, this needs a lock or per-level precompute.
//
// HashDims is the number of pose entries Hash reads; Hash is the environment's calc_hash.
//
// shortcut: a level is refused when its cell index would no longer fit the 32-bit cast calc_hash may use
// (|pose / resolution| >= 1e9). Past that the resolution is far finer than any pose can be told apart at; lift it
// only if every environment hashes with int64.
template <int HashDims, size_t (*Hash)(float *, const float *)>
struct LazyIndex {
  float p[HashDims];
  float res[HashDims];
  float division = 1.f;
  std::vector<size_t> v;

  void init(const float *pose, const float *resolution, float division_factor) {
    for (int i = 0; i < HashDims; ++i) {
      p[i] = pose[i];
      res[i] = resolution[i];
    }
    division = division_factor;
  }

  size_t operator[](size_t l) {
    while (v.size() <= l) {
      for (int j = 0; j < HashDims; ++j) {
        if (std::fabs(p[j]) >= res[j] * 1e9f) {  // false for a NaN pose: those are filtered elsewhere
          throw std::runtime_error(
              "IGHA* level " + std::to_string(v.size()) + ": cell size " + std::to_string(res[j]) +
              " in dimension " + std::to_string(j) +
              " is too small to hash (pose / cell size must stay below 1e9). The search has run out of "
              "useful levels: check resolution, tolerance and division_factor.");
        }
      }
      v.push_back(Hash(p, res));
      for (int j = 0; j < HashDims; ++j)
        res[j] = res[j] / division;
    }
    return v[l];
  }
};
