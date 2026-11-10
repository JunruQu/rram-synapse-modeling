"""实测协议重放的READ观测定义。"""

import json
from pathlib import Path

import numpy as np
import pytest

from fit_ltp import reconstruct_drive
from simulate_ltp_independent import replay


def test_replay_uses_same_read_window_as_fit():
    source = Path("outputs/ltp_ltd_hires/1ns")
    config = dict(
        parameters=json.loads((source / "parameters.json").read_text()),
        initial_state=json.loads((source / "initial_state.json").read_text()),
        transient_options=json.loads((source / "fit_config.json").read_text())[
            "transient_options"
        ],
    )
    source = np.array([[0, 0, 0], [4e-6, 0, 0]])
    t, v = reconstruct_drive(source, 1, 2)
    result, values, stats = replay(t, v, 2, config, 0)
    for cycle in range(2):
        indices = np.searchsorted(t, cycle * 2e-6 + np.linspace(1.15e-6, 1.45e-6, 7))
        expected = np.mean([result.points[i].current_a for i in indices]) / np.mean(
            v[indices]
        )
        assert values[cycle, 3] == pytest.approx(expected)
        assert np.all(v[indices] == -0.4)
    assert result.states[0].x_m == config["initial_state"]["x_m"]
    assert result.states[0].w_m == config["initial_state"]["w_m"]
    assert stats["accepted_steps"] > 0
