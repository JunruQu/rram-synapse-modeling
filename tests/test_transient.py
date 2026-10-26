"""瞬态物理极限、单位换算及时间重建的回归测试。"""

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from fit_pulse import (
    INITIAL,
    PulseData,
    measurement_response,
    predict,
    prepare,
    rebuild_time,
)
from rram_model import (
    Parameters,
    State,
    TransientOptions,
    _fixed_temperature_point,
    operating_point,
    simulate_transient,
)


class TransientTests(unittest.TestCase):
    def test_frozen_temperature_circuit_matches_isothermal_dc(self):
        p = replace(Parameters(), thermal_k_w=0, parallel_ohm=1e5)
        for state in (
            State(),
            State(0.1e-9, 2e-9, 1e-9),
            State(0, 5e-9, 1e-9),
            State(5e-9, 5e-9, 5e-9),
        ):
            for voltage in (-1.0, -0.1, 0.1, 1.0):
                fast = _fixed_temperature_point(voltage, state, p, p.ambient_k)
                dc = operating_point(voltage, state, p)
                self.assertAlmostEqual(fast.current_a / dc.current_a, 1, places=7)
                self.assertAlmostEqual(fast.gap_v, dc.gap_v, places=8)

    def test_circuit_solver_across_small_gaps_and_temperature_range(self):
        rng = np.random.default_rng(20261023)
        for _ in range(80):
            temperature = float(rng.uniform(250, 1500))
            p = replace(
                Parameters(),
                thermal_k_w=0,
                ambient_k=temperature,
                conductivity_s_m=float(10 ** rng.uniform(3, 6)),
                conduction_ev=float(rng.uniform(0, 0.15)),
            )
            gap = float(10 ** rng.uniform(-15, math.log10(5e-9)))
            state = State(gap, float(rng.uniform(0.5e-9, 5e-9)), max(1e-9, gap))
            voltage = float(rng.uniform(-1.5, 1.5))
            fast = _fixed_temperature_point(voltage, state, p, temperature)
            dc = operating_point(voltage, state, p)
            self.assertLess(
                abs(fast.current_a - dc.current_a), 1e-11 + 1e-7 * abs(dc.current_a)
            )

    def test_thermal_step_and_cooling_match_analytic_exponential(self):
        p = replace(
            Parameters(),
            conduction_ev=0,
            conductivity_s_m=5e5,
            thermal_k_w=1e4,
            attempt_hz=1e-100,
        )
        state = State(0, 5e-9, 1e-9)
        time = np.linspace(0, 100e-9, 101)
        options = TransientOptions(thermal_time_s=10e-9)
        result = simulate_transient(time, np.full(101, 0.2), state, p, options)
        initial = result.points[0]
        equilibrium = (
            p.ambient_k + initial.switching_a * initial.internal_v * p.thermal_k_w
        )
        expected = equilibrium + (p.ambient_k - equilibrium) * np.exp(
            -time / options.thermal_time_s
        )
        np.testing.assert_allclose(
            [o.temperature_k for o in result.points], expected, atol=1e-8
        )
        cooling = simulate_transient(
            time, np.zeros(101), state, p, options, initial_temperature_k=350
        )
        np.testing.assert_allclose(
            [o.temperature_k for o in cooling.points],
            300 + 50 * np.exp(-time / options.thermal_time_s),
            atol=1e-8,
        )

    def test_capacitive_charge_and_no_false_joule_heating(self):
        p = replace(Parameters(), thermal_k_w=0, attempt_hz=1e-100)
        time = np.linspace(0, 20e-9, 101)
        voltage = np.linspace(-0.1, 0.1, 101)
        state = State()
        plain = simulate_transient(time, voltage, state, p, TransientOptions(5e-9, 0))
        result = simulate_transient(
            time, voltage, state, p, TransientOptions(5e-9, 3e-12)
        )
        np.testing.assert_allclose(result.capacitive_a, 3e-5, rtol=1e-10)
        self.assertAlmostEqual(
            float(np.trapezoid(result.capacitive_a, time)), 3e-12 * 0.2, places=22
        )
        self.assertEqual(result.states, plain.states)
        self.assertEqual(
            [o.temperature_k for o in result.points],
            [o.temperature_k for o in plain.points],
        )

    def test_no_compliance_is_injected_into_pulses(self):
        p = replace(Parameters(), thermal_k_w=0, attempt_hz=1e-100)
        r = simulate_transient([0, 10e-9], [0.5, 0.5], State(0, 5e-9, 1e-9), p)
        self.assertGreater(r.points[-1].current_a, 0.001)
        self.assertFalse(any(o.limited for o in r.points))

    def test_state_and_temperature_continuity_at_split(self):
        p = replace(
            Parameters(), thermal_k_w=1e5, generation_ev=1.65, recombination_ev=0.64
        )
        time = np.linspace(0, 2e-6, 401)
        voltage = np.where(time < 0.2e-6, 0.9, np.where(time > 1e-6, -1.3, 0.0))
        state = State(0.15e-9, 2.5e-9, 1e-9)
        opts = TransientOptions(5e-9, 2e-12)
        full = simulate_transient(time, voltage, state, p, opts)
        first = simulate_transient(time[:201], voltage[:201], state, p, opts)
        tail = simulate_transient(
            time[200:],
            voltage[200:],
            first.states[-1],
            p,
            opts,
            initial_temperature_k=first.points[-1].temperature_k,
        )
        self.assertLess(abs(tail.states[-1].x_m - full.states[-1].x_m), 1e-12)
        self.assertLess(
            abs(tail.points[-1].temperature_k - full.points[-1].temperature_k), 0.1
        )
        for s in full.states:
            s.validate(p)

    def test_resolution_convergence(self):
        p = replace(
            Parameters(), thermal_k_w=1e5, generation_ev=1.65, recombination_ev=0.64
        )
        time = np.linspace(0, 1e-6, 201)
        voltage = np.where(time < 0.2e-6, 0.9, -0.8)
        state = State(0.15e-9, 2.5e-9, 1e-9)
        opts = TransientOptions(5e-9, 2e-12)
        coarse = simulate_transient(time, voltage, state, p, opts, rtol=0.003)
        fine = simulate_transient(
            time, voltage, state, p, opts, rtol=0.0005, atol_m=2e-14, max_step_s=1e-9
        )
        finest = simulate_transient(
            time, voltage, state, p, opts, rtol=0.0001, atol_m=5e-15, max_step_s=0.5e-9
        )
        self.assertLess(abs(coarse.states[-1].x_m - fine.states[-1].x_m), 0.002e-9)
        a = np.array([o.current_a for o in coarse.points])
        b = np.array([o.current_a for o in fine.points])
        c = np.array([o.current_a for o in finest.points])
        coarse_error = np.sqrt(np.mean((a - b) ** 2))
        fine_error = np.sqrt(np.mean((b - c) ** 2))
        self.assertLess(coarse_error, 0.001 * np.max(np.abs(c)))
        self.assertLess(fine_error, 0.4 * coarse_error)

    def test_time_options_and_temperature_validation(self):
        with self.assertRaises(ValueError):
            TransientOptions(-1, 0)
        with self.assertRaises(ValueError):
            simulate_transient([0, 0], [0, 1])
        with self.assertRaises(ValueError):
            simulate_transient([0, 1], [0, 1], initial_temperature_k=-1)


class PulseDataTests(unittest.TestCase):
    def test_rounding_rebuild_retains_every_row_and_unwraps(self):
        t, info = rebuild_time(["2.50E-08", "7.50E-08", "1.25E-07", "1.75E-07"])
        np.testing.assert_allclose(np.diff(t), 50e-9)
        t, info = rebuild_time(
            ["0.000", "0.001", "0.002", "0.000", "0.001", "0.002"], True
        )
        np.testing.assert_allclose(t, np.arange(6) * 0.001)
        self.assertEqual(info["time_reset_rows_zero_based"], [3])
        with self.assertRaises(ValueError):
            rebuild_time(["0.000", "0.001", "0.003", "0.004"])

    def test_polarity_and_scope_shunt_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw"
            source.mkdir()
            (source / "pulse_iv_100ns.csv").write_text(
                "time_s,voltage_v,current_a\n0.000,0.9,-0.001\n0.001,0.2,-0.0003\n0.002,-1.3,0.0005\n"
            )
            (source / "pulse_iv_500ns_HiRes.csv").write_text(
                "time_s,vforce_v,vsense_v\n0.000,0.9,0.05\n0.001,0.4,0.01\n0.002,-1.3,-0.05\n"
            )
            data, _ = prepare(source, Path(directory) / "prepared")
            self.assertTrue(
                (Path(directory) / "prepared" / "100ns_b1530_prepared.csv").is_file()
            )
            self.assertTrue(
                (Path(directory) / "prepared" / "500ns_hires_prepared.csv").is_file()
            )
            b = next(d for d in data if d.instrument == "b1530")
            h = next(d for d in data if d.instrument == "hires")
            np.testing.assert_allclose(b.current_a, [0.001, 0.0003, -0.0005])
            np.testing.assert_allclose(h.current_a, [0.001, 0.0002, -0.001])
            np.testing.assert_allclose(h.voltage_v, [0.85, 0.39, -1.25])
            self.assertEqual(b.integration_s, 10e-9)

    def test_current_channel_step_and_known_window(self):
        t = np.linspace(0, 100e-9, 1001)
        x = np.ones_like(t)
        x[0] = 0
        y = measurement_response(t, x, t, tau_s=10e-9)
        self.assertLess(abs(y[-1] - (1 - np.exp(-10))), 1e-5)
        ramp = 2 * t
        centers = t[100:-100]
        average = measurement_response(t, ramp, centers, integration_s=10e-9)
        np.testing.assert_allclose(average, 2 * centers, atol=1e-18)

    def test_prediction_preserves_drive_knots_and_does_not_use_observed_current(self):
        t = np.array([25e-9, 75e-9, 125e-9])
        v = np.array([0.4, 0.6, 0.0])
        d = PulseData("check", t, v, np.full(3, 0.001), 50, "b1530", 10e-9)
        p = replace(Parameters(), attempt_hz=1e-100)
        s = State(0, 5e-9, 1e-9)
        a, result, offset = predict(d, INITIAL, p, s)
        b, _, _ = predict(replace(d, current_a=np.full(3, 0.002)), INITIAL, p, s)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(offset, 0)
        for sample_time, voltage in zip(t, v, strict=True):
            index = result.time_s.index(sample_time)
            self.assertAlmostEqual(result.points[index].requested_v, voltage)


if __name__ == "__main__":
    unittest.main()
