"""本轮完整READ协议的SET→RESET连续状态回归。"""

import numpy as np

from fit_ltp import load_initialization, reconstruct_drive
from rram_model import Parameters, State, TransientOptions, simulate_transient


def test_set_reset_state_continuity():
    warm = load_initialization(100)
    parameters = Parameters(**warm["parameters"])
    initial = State(**warm["initial_state"])
    options = TransientOptions(**warm["transient_options"])
    times, volts = reconstruct_drive(np.array([[0, 0, 0], [4e-6, 0, 0]]), 100, 2)
    full = simulate_transient(
        times, volts, initial, parameters, options, rtol=1e-3, max_step_s=50e-9
    )
    boundary = int(np.searchsorted(times, 2e-6))
    first = simulate_transient(
        times[: boundary + 1],
        volts[: boundary + 1],
        initial,
        parameters,
        options,
        rtol=1e-3,
        max_step_s=50e-9,
    )
    # 阶段终态是下一阶段初态；温度同样继承，不能重置到300 K。
    second = simulate_transient(
        times[boundary:] - times[boundary],
        volts[boundary:],
        first.states[-1],
        parameters,
        options,
        rtol=1e-3,
        max_step_s=50e-9,
        initial_temperature_k=first.points[-1].temperature_k,
    )
    assert second.states[0] == first.states[-1] == full.states[boundary]
    assert second.points[0].temperature_k == first.points[-1].temperature_k
    assert second.states[-1] == full.states[-1]
    np.testing.assert_allclose(
        [point.current_a for point in second.points],
        [point.current_a for point in full.points[boundary:]],
        rtol=1e-10,
        atol=1e-12,
    )
