"""物理极限、反馈和数值一致性验证；无第三方依赖也可运行。"""

import math
import unittest
from dataclasses import replace

from rram_model import (
    KB_EV,
    ModelError,
    Parameters,
    State,
    conductivity,
    hopping_density,
    operating_point,
    simulate_dc,
    switching_rates,
)


class ConductionTests(unittest.TestCase):
    def setUp(self):
        self.p = Parameters()

    def test_metallic_limit_matches_ohms_law(self):
        p = replace(self.p, thermal_k_w=0)
        s = State(0, p.filament_max_m, 1e-9)
        area = math.pi * s.w_m**2 / 4
        resistance = p.thickness_m / (area * conductivity(p.ambient_k, p))
        point = operating_point(0.2, s, p)
        self.assertAlmostEqual(
            point.current_a / (0.2 / (resistance + p.series_ohm)), 1, places=9
        )
        self.assertEqual(point.temperature_k, p.ambient_k)

    def test_pure_gap_matches_hopping_equation(self):
        p = replace(self.p, series_ohm=0, thermal_k_w=0)
        s = State(p.thickness_m, p.filament_max_m, p.thickness_m)
        expected = math.pi * s.w_m**2 / 4 * hopping_density(0.4, s.x_m, p)
        self.assertAlmostEqual(
            operating_point(0.4, s, p).current_a / expected, 1, places=8
        )

    def test_activation_sign_and_hopping_oddness(self):
        self.assertGreater(conductivity(300, self.p), conductivity(500, self.p))
        self.assertAlmostEqual(
            hopping_density(0.3, 1e-9, self.p), -hopping_density(-0.3, 1e-9, self.p)
        )
        self.assertEqual(hopping_density(0, 1e-9, self.p), 0)
        self.assertGreater(
            hopping_density(0.3, 0.5e-9, self.p), hopping_density(0.3, 1e-9, self.p)
        )

    def test_electrothermal_kcl_and_energy(self):
        p = replace(self.p, parallel_ohm=1e5)
        point = operating_point(0.6, State(), p)
        self.assertAlmostEqual(
            point.internal_v + p.series_ohm * point.current_a,
            point.terminal_v,
            places=10,
        )
        self.assertAlmostEqual(
            point.switching_a + point.internal_v / p.parallel_ohm,
            point.current_a,
            places=12,
        )
        self.assertAlmostEqual(
            point.temperature_k,
            p.ambient_k + point.switching_a * point.internal_v * p.thermal_k_w,
            places=9,
        )
        self.assertGreaterEqual(point.current_a * point.internal_v, 0)

    def test_electrothermal_symmetry_and_zero(self):
        for s in (State(), State(0, 5e-9, 1e-9)):
            positive, negative = (
                operating_point(0.5, s, self.p),
                operating_point(-0.5, s, self.p),
            )
            self.assertAlmostEqual(positive.current_a, -negative.current_a, places=13)
            self.assertAlmostEqual(positive.temperature_k, negative.temperature_k)
        zero = operating_point(0, State(), self.p)
        self.assertEqual(zero.current_a, 0)
        self.assertEqual(zero.temperature_k, self.p.ambient_k)

    def test_compliance_is_voltage_and_temperature_feedback(self):
        s = State(0, 5e-9, 1e-9)
        p = replace(self.p, parallel_ohm=1e6)
        unlimited = operating_point(0.6, s, p)
        limited = operating_point(0.6, s, p, compliance_a=1e-4)
        self.assertTrue(limited.limited)
        self.assertEqual(limited.current_a, 1e-4)
        self.assertLess(limited.terminal_v, 0.6)
        self.assertLess(limited.internal_v, unlimited.internal_v)
        self.assertLess(limited.temperature_k, unlimited.temperature_k)
        # 再次以实际限流端电压、不指定限流求解，应得到相同电路工作点。
        replay = operating_point(limited.terminal_v, s, p)
        self.assertAlmostEqual(replay.current_a / limited.current_a, 1, places=8)

    def test_subatomic_gap_has_continuous_contact_limit(self):
        s = State(0, 0.5e-9, 1e-9)
        p = replace(self.p, thermal_k_w=0)
        closed = operating_point(0.1, s, p)
        almost = operating_point(0.1, replace(s, x_m=1e-19), p)
        self.assertLess(abs(almost.current_a / closed.current_a - 1), 1e-6)

    def test_set_only_compliance_keeps_reset_unlimited(self):
        s = State(0, 5e-9, 1e-9)
        for polarity in (1, -1):
            p = replace(self.p, set_polarity=polarity)
            set_point = operating_point(
                polarity * 0.6, s, p, 1e-4, compliance_mode="set"
            )
            reset_point = operating_point(
                -polarity * 0.6, s, p, 1e-4, compliance_mode="set"
            )
            unrestricted = operating_point(-polarity * 0.6, s, p)
            self.assertTrue(set_point.limited)
            self.assertFalse(reset_point.limited)
            self.assertEqual(reset_point.current_a, unrestricted.current_a)
            self.assertGreater(abs(reset_point.current_a), 1e-4)
        result = simulate_dc(
            [0, 1e-9], [-0.6, -0.6], s, self.p, compliance_a=1e-4, compliance_mode="set"
        )
        self.assertFalse(any(point.limited for point in result.points))

    def test_invalid_parameters_and_state(self):
        for changes in (
            {"hopping_length_m": 0},
            {"series_ohm": -1},
            {"generation_ev": math.nan},
            {"set_polarity": 0},
            {"seed_diameter_m": 6e-9},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.p, **changes)
        for s in (
            State(-1e-9, 0.5e-9, 1e-9),
            State(2e-9, 0.5e-9, 1e-9),
            State(0, 0, 1e-9),
        ):
            with self.subTest(state=s), self.assertRaises(ValueError):
                operating_point(0.1, s, self.p)
        with self.assertRaises(ValueError):
            operating_point(0.1, State(), self.p, 0)

    def test_excess_temperature_is_reported(self):
        p = replace(self.p, temperature_limit_k=301)
        with self.assertRaisesRegex(ModelError, "温度"):
            operating_point(0.5, State(0, 5e-9, 1e-9), p)


class DynamicsTests(unittest.TestCase):
    def test_rates_follow_equations_and_polarity(self):
        p = Parameters()
        s = State()
        point = operating_point(0.2, s, p)
        rates = switching_rates(0.2, s, p, point)
        field = point.gap_v / s.x_m
        expected = (
            -p.atomic_step_m
            * p.attempt_hz
            * math.exp(
                -(p.generation_ev - p.generation_alpha_m * p.charge_number * field)
                / (KB_EV * point.temperature_k)
            )
        )
        self.assertAlmostEqual(rates.dx_m_s / expected, 1)
        reset = switching_rates(-0.5, s, p)
        self.assertEqual(
            reset.dx_m_s,
            min(
                reset.reset_hopping_m_s,
                reset.reset_release_m_s,
                reset.reset_recombination_m_s,
            ),
        )
        reversed_rates = switching_rates(-0.2, s, replace(p, set_polarity=-1))
        self.assertAlmostEqual(reversed_rates.dx_m_s / rates.dx_m_s, 1)
        self.assertEqual(switching_rates(0, s).dx_m_s, 0)

    def test_set_then_reset_and_bounded_geometry(self):
        r = simulate_dc([0, 0.001, 0.002, 0.003, 0.004], [0, 1, 0, -2, 0])
        self.assertEqual(r.states[1].x_m, 0)
        self.assertGreater(r.states[1].w_m, r.states[0].w_m)
        self.assertGreater(r.states[-1].x_m, 0.5e-9)
        for s in r.states:
            s.validate(Parameters())
        self.assertGreater(r.accepted_steps, 0)

    def test_reset_subtolerance_increment_is_not_erased(self):
        p = replace(Parameters(), thermal_k_w=0)
        initial = State(0, 0.5e-9, 1e-9)
        r = simulate_dc([0, 1e-5], [-0.2, -0.2], initial, p)
        self.assertGreater(r.states[-1].x_m, 0)

    def test_convergence_when_tolerance_and_step_are_tightened(self):
        times = [k * 1e-4 for k in range(41)]
        voltages = [
            t / 0.001
            if t <= 0.001
            else (
                2 - t / 0.001
                if t <= 0.002
                else -2 * (t - 0.002) / 0.001
                if t <= 0.003
                else -2 + 2 * (t - 0.003) / 0.001
            )
            for t in times
        ]
        coarse = simulate_dc(times, voltages, rtol=1e-3, atol_m=1e-13)
        fine = simulate_dc(
            times, voltages, rtol=2.5e-4, atol_m=2.5e-14, max_step_s=1e-5
        )
        self.assertLess(abs(coarse.states[-1].x_m - fine.states[-1].x_m), 0.01e-9)
        # 忽略近零测点的相对误差放大，整体电流变化应远小于峰值。
        maximum_error = max(
            abs(a.current_a - b.current_a)
            for a, b in zip(coarse.points, fine.points, strict=True)
        )
        peak = max(abs(p.current_a) for p in fine.points)
        self.assertLess(maximum_error / peak, 0.02)

    def test_segmented_sweep_preserves_state(self):
        times, voltages = [0, 0.001, 0.002, 0.003], [0, 1, 0, -2]
        whole = simulate_dc(times, voltages)
        first = simulate_dc(times[:3], voltages[:3])
        second = simulate_dc(times[2:], voltages[2:], first.states[-1])
        self.assertLess(abs(whole.states[-1].x_m - second.states[-1].x_m), 1e-12)

    def test_invalid_time_and_iteration_limit(self):
        for times, voltages in [
            ([0, 0], [0, 1]),
            ([1, 0], [0, 1]),
            ([], []),
            ([0], [math.inf]),
        ]:
            with self.subTest(times=times), self.assertRaises(ValueError):
                simulate_dc(times, voltages)
        with self.assertRaisesRegex(ModelError, "max_steps"):
            simulate_dc([0, 0.001], [0, 1], max_steps=1)


if __name__ == "__main__":
    unittest.main()
