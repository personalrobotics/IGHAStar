"""LazyIndex (Environments/include/lazy_index.h) must give the hashes an eager per-level precompute would."""
import pathlib
import shutil
import subprocess

import pytest

INCLUDE = pathlib.Path(__file__).resolve().parents[3] / "ighastar" / "src" / "Environments" / "include"

PROGRAM = r"""
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include "lazy_index.h"

// A hash that depends on every dimension and on the cell size, like the environments' calc_hash.
static size_t hash3(float *p, const float *r) {
  size_t h = 1469598103934665603ull;
  for (int i = 0; i < 3; ++i) {
    long long c = (long long)std::floor(p[i] / r[i]);
    h = (h ^ (size_t)c) * 1099511628211ull;
  }
  return h;
}

int main() {
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> u(-500.f, 500.f);
  const float res[3] = {40.f, 40.f, 0.5f};
  const float div = 1.0718f;
  for (int n = 0; n < 200; ++n) {
    float pose[3] = {u(rng), u(rng), u(rng)};
    LazyIndex<3, hash3> lazy;
    lazy.init(pose, res, div);
    float r[3] = {res[0], res[1], res[2]};
    // ask out of order and repeatedly: a later level first, then earlier ones
    size_t order[] = {7, 0, 12, 3, 12, 30, 1};
    for (size_t l : order) {
      float rr[3] = {res[0], res[1], res[2]};
      size_t eager = 0;
      for (size_t k = 0; k <= l; ++k) {
        float q[3] = {pose[0], pose[1], pose[2]};
        eager = hash3(q, rr);
        for (int j = 0; j < 3; ++j) rr[j] = rr[j] / div;
      }
      if (lazy[l] != eager) { std::printf("MISMATCH pose %d level %zu\n", n, l); return 1; }
    }
    (void)r;
  }
  // a cell size too small for the pose must throw, not hash garbage
  float pose[3] = {100.f, 0.f, 0.f};
  LazyIndex<3, hash3> deep;
  deep.init(pose, res, 1e30f);
  deep[0];
  try { deep[1]; std::printf("NO THROW\n"); return 2; } catch (const std::runtime_error &) {}
  std::printf("OK\n");
  return 0;
}
"""


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_lazy_index_matches_eager_hashes(tmp_path):
    src = tmp_path / "t.cpp"
    src.write_text(PROGRAM)
    exe = tmp_path / "t"
    subprocess.run(["g++", "-std=c++17", "-O1", f"-I{INCLUDE}", str(src), "-o", str(exe)], check=True)
    out = subprocess.run([str(exe)], capture_output=True, text=True)
    assert out.stdout.strip() == "OK", out.stdout
