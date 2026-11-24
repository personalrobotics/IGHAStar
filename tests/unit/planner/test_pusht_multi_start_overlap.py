#!/usr/bin/env python3
"""pusht_multi's start and goal check allows contact overlap up to start_overlap_px.

A state the search reaches in contact overlaps by Chipmunk's contact slop or
more, so a later search must accept it as a start. For each kind of contact
(pusher against a T, T against T, T against a board wall) a body is moved
toward another in 0.01 px steps. The last state the exact check
(start_overlap_px 0) accepts is the contact. From there, the default check
must accept 1.5 px of overlap and refuse 2.5 px, and the exact check must
refuse 0.1 px. The pusher-T case also covers the overlap's sign: Chipmunk's
cpShapesCollide returns it flipped for a tee against the pusher.
"""
import copy
import pathlib
import sys

import numpy as np
import pytest
import torch
import yaml

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO))

from ighastar.scripts.common_utils import create_planner  # noqa: E402

CONFIG = REPO / "examples" / "standalone" / "Configs" / "pusht_multi_2T1O_rearrange.yml"
WORLD = torch.zeros(1)  # no interior walls; rearrange closes the box


def _planner(overlap=None):
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    if overlap is not None:
        cfg["experiment_info_default"]["node_info"]["start_overlap_px"] = overlap
    return create_planner(cfg, bidirectional=False)


@pytest.fixture(scope="module")
def planners():
    return {"exact": _planner(0.0), "default": _planner()}


def _state(pusher, t0, t1):
    s = np.zeros(16, dtype=np.float32)
    s[0:2], s[2:5], s[5:8] = pusher, t0, t1
    return s


# Each case: a state as a function of the moving body's coordinate, and the
# direction that deepens the contact.
CASES = {
    "pusher-T": (lambda y: _state((256.0, y), (256.0, 256.0, 0.0), (100.0, 100.0, 0.0)), 150.0, +1),
    "T-T": (lambda x: _state((60.0, 60.0), (256.0, 256.0, 0.0), (x, 256.0, 0.0)), 420.0, -1),
    "T-wall": (lambda x: _state((60.0, 60.0), (x, 256.0, 0.0), (100.0, 100.0, 0.0)), 400.0, +1),
}


def _valid(planner, states):
    batch = torch.from_numpy(np.stack(states))
    return planner.check_validity_batched(batch, WORLD).numpy().astype(bool)


def _contact(planner, make, start, direction):
    """The deepest coordinate the exact check still accepts, approaching from start."""
    coords = start + direction * np.arange(0.0, 200.0, 0.01)
    ok = _valid(planner, [make(c) for c in coords])
    assert ok[0], "the sweep must start clear of contact"
    assert not ok.all(), "the sweep must reach contact"
    return coords[np.argmin(ok) - 1]


@pytest.mark.parametrize("case", sorted(CASES))
def test_contact_overlap(planners, case):
    make, start, direction = CASES[case]
    contact = _contact(planners["exact"], make, start, direction)
    at = lambda depth: make(contact + direction * depth)  # noqa: E731
    exact = _valid(planners["exact"], [at(0.0), at(0.1)])
    default = _valid(planners["default"], [at(0.0), at(1.5), at(2.5)])
    assert exact.tolist() == [True, False]
    assert default.tolist() == [True, True, False]


def test_negative_tolerance_refused():
    with pytest.raises(Exception, match="start_overlap_px"):
        _planner(-1.0)
