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
    _capacitive_observation,
    _clean_unaccepted_pulse_outputs,
    _hires_residual,
    _initial_conductance_residual,
    _initial_gap_seed,
    _narrow_peak_residual,
    _observation_prior_residual,
    _priority_voltage_node_mask,
    _pulse_peak_metrics,
    _read_pulse_envelope_errors,
    _read_relative_errors,
    _read_waveform_residual,
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
    def test_observation_anchor_is_a_20_percent_soft_prior(self):
        values = np.r_[np.ones(16), math.log(14.5)]
        values[7] = 5.394015491627448
        values[8] = 7.352863995975265
        prior = {
            "delay_ns": 14.5,
            "response_ns": 7.352863995975265,
            "capacitance_pf": 5.394015491627448,
            "relative_scale": 0.2,
        }
        np.testing.assert_allclose(_observation_prior_residual(values, prior), 0.0)
        values[16] += math.log(1.2)
        self.assertAlmostEqual(_observation_prior_residual(values, prior)[0], 1.0)
        values[8] *= 1.2
        self.assertAlmostEqual(_observation_prior_residual(values, prior)[1], 1.0)
        values[7] *= 1.2
        self.assertAlmostEqual(_observation_prior_residual(values, prior)[2], 1.0)

    def test_output_cleanup_keeps_only_selected_trees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pulse"
            (root / "100ns_refit" / "100ns_hires").mkdir(parents=True)
            (root / "100ns_refit" / "numerical_validation.json").write_text("x")
            (root / "100ns_refit" / "100ns_hires" / "report.json").write_text("x")
            (root / "debug" / "candidate").mkdir(parents=True)
            (root / "debug" / "candidate" / "report.json").write_text("x")
            _clean_unaccepted_pulse_outputs(root, [Path("100ns_refit/100ns_hires")])
            self.assertTrue(
                (root / "100ns_refit" / "100ns_hires" / "report.json").is_file()
            )
            self.assertFalse((root / "debug").exists())
            self.assertFalse(
                (root / "100ns_refit" / "numerical_validation.json").exists()
            )

    def test_initial_gap_seed_uses_requested_conductance_target(self):
        p = Parameters()
        state = State(0.2e-9, 2.5e-9, 1e-9)
        values = np.r_[
            INITIAL,
            p.hopping_ev,
            p.release_ev,
            p.hopping_alpha_m * 1e9,
            0.75,
            0.2,
            p.hopping_voltage_v,
        ]
        before = abs(_initial_conductance_residual(values, p, state, 85e-6, 0.2))
        values[14] = _initial_gap_seed(values, p, state, 85e-6)
        after = abs(_initial_conductance_residual(values, p, state, 85e-6, 0.2))
        self.assertLess(after, before)

    def test_capacitive_observation_is_invariant_to_nonuniform_subdivision(self):
        t = np.array([0.0, 2e-9, 5e-9, 20e-9])
        v = np.array([0.0, 1.0, -0.5, -0.5])
        refined = np.unique(np.r_[t, 1e-9, 2e-9 + 1e-17, 4.9e-9, 19e-9])
        query = np.linspace(-1e-9, 30e-9, 501)
        for tau in (0.0, 3e-9):
            with self.subTest(tau=tau):
                original = _capacitive_observation(
                    t, v, query, 2e-12, tau, delay_s=4e-9
                )
                subdivided = _capacitive_observation(
                    refined, np.interp(refined, t, v), query, 2e-12, tau, delay_s=4e-9
                )
                np.testing.assert_allclose(original, subdivided, atol=1e-16, rtol=1e-12)

    def test_capacitive_ramp_response_and_charge_are_exact(self):
        t = np.array([0.0, 2e-9, 100e-9])
        v = np.array([0.0, 1.0, 1.0])
        cap, tau = 2e-12, 3e-9
        query = np.array([1e-9, 2e-9, 5e-9])
        peak = cap / t[1] * (-math.expm1(-t[1] / tau))
        expected = np.array(
            [
                cap / t[1] * (-math.expm1(-query[0] / tau)),
                peak,
                peak * math.exp(-(query[2] - t[1]) / tau),
            ]
        )
        np.testing.assert_allclose(
            _capacitive_observation(t, v, query, cap, tau), expected, atol=1e-16
        )
        for response in (0.0, tau):
            average = _capacitive_observation(
                t, v, np.array([50e-9]), cap, response, integration_s=100e-9
            )[0]
            self.assertAlmostEqual(average * 100e-9 / cap, 1.0, places=12)

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

    def test_prepare_can_restrict_conversion_to_one_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw"
            output = Path(directory) / "prepared"
            source.mkdir()
            names = ("pulse_iv_100ns_HiRes.csv", "pulse_iv_500ns_HiRes.csv")
            for name in names:
                (source / name).write_text(
                    "time_s,vforce_v,vsense_v\n"
                    "0.00E+00,0.2,0.00001\n"
                    "1.60E-10,0.3,0.00002\n"
                    "3.20E-10,0.4,0.00003\n"
                )
            data, audit = prepare(
                source,
                output,
                source_names=["pulse_iv_500ns_HiRes.csv"],
            )
            self.assertEqual([datum.width_ns for datum in data], [500])
            self.assertEqual(
                [entry["name"] for entry in audit["files"]],
                ["pulse_iv_500ns_HiRes.csv"],
            )
            self.assertTrue((output / "500ns_hires_prepared.csv").is_file())
            self.assertFalse((output / "100ns_hires_prepared.csv").exists())

    def test_read_level_error_uses_both_flat_read_windows(self):
        data = PulseData(
            "pulse_iv_500ns_HiRes",
            np.array([1.2e-6, 1.3e-6, 3.2e-6, 3.3e-6]),
            np.full(4, 0.4),
            np.array([330e-6, 330e-6, 50e-6, 50e-6]),
            500,
            "hires",
            0.0,
        )
        np.testing.assert_allclose(
            _read_relative_errors(data, 1.05 * data.current_a), [0.05, 0.05]
        )

    def test_optimizer_keeps_narrow_program_edges_and_sparsifies_read_nodes(self):
        time = np.linspace(0.0, 4e-6, 4001)
        voltage = np.zeros_like(time)
        voltage[200] = 0.8
        voltage[2200] = -0.9
        data = PulseData(
            "pulse_iv_1ns_HiRes",
            time,
            voltage,
            np.zeros_like(time),
            1,
            "hires",
            0.0,
        )
        selected = _priority_voltage_node_mask(data)
        self.assertTrue(np.all(selected[198:203]))
        self.assertTrue(np.all(selected[2198:2203]))
        self.assertLess(selected.sum(), 200)

    def test_read_waveform_detects_spike_after_flat_mean_window(self):
        time = np.array([1.2e-6, 1.3e-6, 3.2e-6, 3.3e-6, 3.55e-6, 3.8e-6])
        measured = np.array([330e-6, 330e-6, 50e-6, 50e-6, 0.0, 0.0])
        data = PulseData(
            "read_edge", time, np.full(6, 0.4), measured, 200, "hires", 0.0
        )
        prediction = measured.copy()
        prediction[4] = 200e-6
        np.testing.assert_allclose(_read_relative_errors(data, prediction), [0.0, 0.0])
        self.assertGreater(np.linalg.norm(_read_waveform_residual(data, prediction)), 0)
        self.assertGreater(
            float(max(_read_pulse_envelope_errors(data, prediction))), 0.10
        )

    def test_hires_objective_retains_points_outside_priority_windows(self):
        time = np.array([1.2e-6, 1.3e-6, 3.2e-6, 3.3e-6, 3.55e-6, 3.8e-6])
        measured = np.array([330e-6, 330e-6, 50e-6, 50e-6, 0.0, 0.0])
        data = PulseData(
            "global_point", time, np.full(6, 0.4), measured, 200, "hires", 0.0
        )
        reference = Parameters()
        values = np.r_[
            INITIAL,
            reference.hopping_ev,
            reference.release_ev,
            reference.hopping_alpha_m * 1e9,
            0.75,
            0.6,
            reference.hopping_voltage_v,
        ]
        state = State(0.6e-9, 2.5e-9, 1e-9)
        baseline = _hires_residual(data, measured, values, reference, state, "read")
        prediction = measured.copy()
        prediction[-1] += 50e-6
        residual = _hires_residual(data, prediction, values, reference, state, "read")
        self.assertGreater(np.linalg.norm(residual - baseline), 0)

    def test_read_tolerance_does_not_reward_false_edge_spike(self):
        time = np.r_[np.linspace(1e-6, 1.6e-6, 61), np.linspace(3e-6, 3.6e-6, 61)]
        measured = np.where(time < 2e-6, 330e-6, 50e-6)
        measured[(time > 1.5e-6) & (time < 2e-6)] = 0.0
        measured[time > 3.5e-6] = 0.0
        data = PulseData(
            "read_tolerance", time, np.full(len(time), 0.4), measured, 200, "hires", 0.0
        )
        reference = Parameters()
        values = np.r_[
            INITIAL,
            reference.hopping_ev,
            reference.release_ev,
            reference.hopping_alpha_m * 1e9,
            0.75,
            0.6,
            reference.hopping_voltage_v,
        ]
        state = State(0.6e-9, 2.5e-9, 1e-9)
        flat = 0.97 * measured
        spike = measured.copy()
        spike[(time >= 3.52e-6) & (time <= 3.57e-6)] = 200e-6
        self.assertLess(np.max(np.abs(_read_relative_errors(data, flat))), 0.05)
        np.testing.assert_allclose(_read_relative_errors(data, spike), [0.0, 0.0])
        flat_residual = _hires_residual(data, flat, values, reference, state, "read")
        spike_residual = _hires_residual(data, spike, values, reference, state, "read")
        self.assertLess(
            float(flat_residual @ flat_residual), float(spike_residual @ spike_residual)
        )

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

    def test_pure_delay_preserves_peak_and_has_causal_prehistory(self):
        t = np.linspace(0, 100e-9, 1001)
        current = np.where((t >= 10e-9) & (t <= 20e-9), 1e-3, 0.0)
        direct = measurement_response(t, current, t, tau_s=3e-9, initial_current_a=0.0)
        delayed = measurement_response(
            t, current, t, tau_s=3e-9, delay_s=14e-9, initial_current_a=0.0
        )
        np.testing.assert_allclose(delayed, np.interp(t - 14e-9, t, direct, left=0.0))
        self.assertAlmostEqual(float(max(direct)), float(max(delayed)), places=12)
        self.assertTrue(np.all(delayed[t < 14e-9] == 0))

    def test_observation_initial_state_does_not_precharge_filter(self):
        t = np.linspace(0, 20e-9, 201)
        measured = measurement_response(
            t, np.full_like(t, 1e-3), t, tau_s=2e-9, initial_current_a=0.0
        )
        np.testing.assert_allclose(measured, 1e-3 * (1 - np.exp(-t / 2e-9)), atol=1e-15)

    def test_peak_metrics_keep_set_and_reset_polarity(self):
        t = np.array([10e-9, 30e-9, 2.02e-6, 2.04e-6])
        current = np.array([1e-3, -2e-3, -1.5e-3, 3e-3])
        data = PulseData("signed_peaks", t, np.zeros(4), current, 10, "hires", 0.0)
        result = _pulse_peak_metrics(data, 0.8 * current)
        self.assertEqual(result["set"]["measured_a"], 1e-3)
        self.assertEqual(result["reset"]["measured_a"], -1.5e-3)
        self.assertAlmostEqual(result["set"]["relative_error"], -0.2)
        self.assertAlmostEqual(result["reset"]["relative_error"], -0.2)

    def test_narrow_objective_prioritizes_peaks_with_equal_read_levels(self):
        t = np.array([18e-9, 1.2e-6, 1.3e-6, 2.018e-6, 3.2e-6, 3.3e-6])
        current = np.array([1e-3, 300e-6, 300e-6, -1.4e-3, 180e-6, 180e-6])
        data = PulseData("peak_priority", t, np.full(6, 0.4), current, 10, "hires", 0.0)
        p = Parameters()
        values = np.r_[
            INITIAL,
            p.hopping_ev,
            p.release_ev,
            p.hopping_alpha_m * 1e9,
            0.75,
            0.6,
            p.hopping_voltage_v,
        ]
        s = State(0.6e-9, 2.5e-9, 1e-9)
        low = current.copy()
        low[[0, 3]] *= 0.5
        np.testing.assert_allclose(_read_relative_errors(data, low), [0.0, 0.0])
        for stage in ("read", "peaks"):
            with self.subTest(stage=stage):
                good_residual = _narrow_peak_residual(
                    data, current, values, p, s, stage
                )
                low_residual = _narrow_peak_residual(data, low, values, p, s, stage)
                self.assertLess(
                    float(good_residual @ good_residual),
                    float(low_residual @ low_residual),
                )

    def test_observation_delay_does_not_change_device_state_or_heating(self):
        t = np.array([0.0, 1e-9, 2e-9, 5e-9, 20e-9])
        d = PulseData(
            "delay_only",
            t,
            np.array([0.0, 0.8, 0.0, 0.0, 0.0]),
            np.zeros(5),
            10,
            "hires",
            0.0,
        )
        p = replace(Parameters(), attempt_hz=1e-100)
        s = State(0.3e-9, 2.5e-9, 1e-9)
        a, direct, _ = predict(d, INITIAL, p, s, grid_s=0.25e-9)
        b, delayed, _ = predict(d, INITIAL, p, s, grid_s=0.25e-9, current_delay_s=14e-9)
        self.assertEqual(direct.states, delayed.states)
        self.assertEqual(direct.points, delayed.points)
        np.testing.assert_array_equal(direct.capacitive_a, delayed.capacitive_a)
        self.assertFalse(np.array_equal(a, b))

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
