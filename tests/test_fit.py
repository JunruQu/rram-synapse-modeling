"""数据与 CLI 无依赖测试；真实最小二乘测试在 NumPy/SciPy 可用时执行。"""

import csv
import importlib.util
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fit_dc import (
    Dataset,
    create_demo,
    export_results,
    fit_parameters,
    load_config,
    load_csv,
    main,
    objective_blocks,
    predict,
    residual_values,
)
from rram_model import Parameters, State, operating_point, simulate_dc

HAS_SCIPY = (
    importlib.util.find_spec("scipy") is not None
    and importlib.util.find_spec("numpy") is not None
)


class DataTests(unittest.TestCase):
    def test_accepted_balanced_dc_fixed_parameter_replay(self):
        root = Path(__file__).resolve().parents[1]
        folder = root / "outputs/dc/cycle1_balanced"
        config = json.loads((folder / "config_used.json").read_text())
        config["datasets"][0]["path"] = str(
            root / "data/dc/prepared/cycle1_prepared.csv"
        )
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps(config))
            loaded, params, datasets = load_config(config_path)
        points, _ = predict(params, datasets[0], loaded["solver"])
        with (folder / "prediction.csv").open(newline="") as stream:
            saved = list(csv.DictReader(stream))
        self.assertEqual(len(points), 404)
        self.assertEqual(len(saved), len(points))
        self.assertLessEqual(
            max(
                abs(point.current_a - float(row["predicted_a"]))
                for point, row in zip(points, saved, strict=True)
            ),
            1e-12,
        )

    def test_csv_preserves_scan_order_and_explicit_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curve.csv"
            path.write_text(
                "voltage_v,current_a\n0,0\n1,1e-4\n0,0\n-1,-1e-4\n", encoding="utf-8"
            )
            voltage, _, time = load_csv(path, dynamic=True, dwell_s=0.01)
            self.assertEqual(voltage, [0, 1, 0, -1])
            self.assertEqual(time, [0, 0.01, 0.02, 0.03])
            with self.assertRaisesRegex(ValueError, "time_s"):
                load_csv(path, dynamic=True)

    def test_bad_csv_is_not_silently_cleaned(self):
        for content in (
            "v,i\n0,0\n",
            "voltage_v,current_a\n",
            "voltage_v,current_a\n0,nan\n",
            "time_s,voltage_v,current_a\n0,0,0\n0,1,1e-4\n",
        ):
            with (
                self.subTest(content=content),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "bad.csv"
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_csv(path, dynamic=True)

    def test_residual_is_signed_finite_and_balanced(self):
        r = residual_values([-1e-3, 0, 1e-3], [0, 0, 0])
        self.assertTrue(all(math.isfinite(v) for v in r))
        self.assertAlmostEqual(r[0], -r[2])
        self.assertEqual(r[1], 0)
        once = sum(v * v for v in residual_values([1e-4], [0]))
        many = sum(v * v for v in residual_values([1e-4] * 10, [0] * 10))
        self.assertAlmostEqual(once, many)
        with self.assertRaises(ValueError):
            residual_values([0], [0], scale_a=0)

    def test_invalid_residual_windows_are_rejected(self):
        dataset = Dataset("s", [0, 1], [0, 0], None, "static", State())
        for change in (
            {"start": -1},
            {"stop": 3},
            {"start": 1, "stop": 1},
            {"start": 0.5},
            {"dataset": "unknown"},
            {"weight": 0},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                objective_blocks(
                    [dataset], {"residual_blocks": [{"dataset": "s", **change}]}
                )

    def test_static_predictions_freeze_state(self):
        dataset = Dataset("fixed", [0.5, 0, -0.5], [0] * 3, None, "static", State())
        points, states = predict(Parameters(), dataset)
        self.assertEqual(states, [State()] * 3)
        self.assertAlmostEqual(points[0].current_a, -points[2].current_a)

    def test_demo_reproducibility_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            static, dynamic = create_demo(directory)
            path = Path(directory) / "SYNTHETIC_dynamic.csv"
            first = path.read_bytes()
            create_demo(directory)
            self.assertEqual(first, path.read_bytes())
            for config_path in (static, dynamic):
                config, params, data = load_config(config_path)
                output = Path(directory) / config_path.stem
                report = export_results(
                    output, params, data, config.get("solver"), plot=False
                )
                prediction_name = (
                    "prediction.csv" if len(data) == 1 else "prediction_0.csv"
                )
                self.assertTrue((output / prediction_name).is_file())
                self.assertTrue((output / "parameters.json").is_file())
                self.assertIn("datasets", report)
            config = json.loads(dynamic.read_text(encoding="utf-8"))
            config["datasets"][0]["mode"] = "typo"
            dynamic.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mode"):
                load_config(dynamic)

    def test_simulation_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config = create_demo(directory)
            code = main(
                [
                    "simulate",
                    str(config),
                    "--output",
                    str(Path(directory) / "result"),
                    "--no-plot",
                ]
            )
            self.assertEqual(code, 0)


@unittest.skipUnless(HAS_SCIPY, "NumPy/SciPy 未安装，无法验证真实优化器")
class CalibrationTests(unittest.TestCase):
    def test_recovers_static_conduction_parameters(self):
        truth = replace(Parameters(), thermal_k_w=0)
        datasets = []
        for state in (
            State(0.8e-9, 0.5e-9, 0.8e-9),
            State(1.3e-9, 0.5e-9, 1.3e-9),
            State(0, 5e-9, 1e-9),
        ):
            voltage = [-0.3, -0.15, 0.05, 0.15, 0.3]
            current = [operating_point(v, state, truth).current_a for v in voltage]
            datasets.append(
                Dataset(str(state.x_m), voltage, current, None, "static", state)
            )
        specs = {
            "hopping_density_a_m2": {"initial": 7e12, "bounds": [2e12, 3e13]},
            "hopping_voltage_v": {"initial": 0.48, "bounds": [0.2, 0.7]},
            "conductivity_s_m": {"initial": 7e5, "bounds": [2e5, 3e6]},
        }
        fitted, report = fit_parameters(
            truth, datasets, {"parameters": specs, "gtol": 1e-11}
        )
        self.assertTrue(report["success"])
        self.assertLess(
            report["final_squared_residual"], report["initial_squared_residual"] * 1e-6
        )
        for name in specs:
            self.assertLess(abs(getattr(fitted, name) / getattr(truth, name) - 1), 0.01)

    def test_dynamic_subset_calibration_improves_loop(self):
        params = Parameters()
        time = [k * 0.000025 for k in range(41)]
        voltage = [k / 40 for k in range(41)]
        solver = {"rtol": 1e-4, "atol_m": 1e-14}
        trace = simulate_dc(time, voltage, params=params, **solver)
        dataset = Dataset(
            "dynamic",
            voltage,
            [p.current_a for p in trace.points],
            time,
            "dynamic",
            State(),
        )
        # 稀疏采样下，错误开关时刻对应的残差可能形成平台。从真值两侧
        # 出发都必须实际改善拟合，不能只检查优化器的 success 标志。
        for initial in (1.15, 1.28):
            with self.subTest(initial=initial):
                fitted, report = fit_parameters(
                    params,
                    [dataset],
                    {
                        "parameters": {
                            "generation_ev": {
                                "initial": initial,
                                "bounds": [1.08, 1.32],
                            }
                        },
                        "diff_step": 2e-4,
                        "set_coordinates": {
                            "reference_field_v_m": 3e8,
                            "barrier_bounds_ev": [0.63, 0.87],
                        },
                        "max_nfev": 60,
                    },
                    solver,
                )
                self.assertTrue(report["success"])
                self.assertLess(
                    report["final_squared_residual"],
                    report["initial_squared_residual"] * 0.05,
                )
                self.assertLess(abs(fitted.generation_ev - params.generation_ev), 0.01)

    def test_window_objective_preserves_dynamic_history(self):
        params = Parameters()
        time = [k * 0.000025 for k in range(41)]
        voltage = [k / 40 for k in range(41)]
        solver = {"rtol": 1e-4, "atol_m": 1e-14}
        trace = simulate_dc(time, voltage, params=params, **solver)
        dataset = Dataset("d", voltage, [0] * 41, time, "dynamic", State())
        # 后段已经 SET，不能把窗口首点重新解释为初始 HRS；窗口末点之后的
        # 因果前缀裁剪则必须不改变本次目标。独立用完整轨迹计算预期平方和。
        window = trace.points[32:38]
        expected = sum(p.current_a**2 for p in window) / (len(window) * 1e-6**2)
        _, report = fit_parameters(
            params,
            [dataset],
            {
                "parameters": {"generation_ev": {"bounds": [1.1, 1.3]}},
                "residual_blocks": [
                    {
                        "dataset": "d",
                        "start": 32,
                        "stop": 38,
                        "residual": "linear",
                        "current_scale_a": 1e-6,
                    }
                ],
                "initialization_points": 0,
                "max_nfev": 1,
            },
            solver,
        )
        self.assertAlmostEqual(
            report["initial_squared_residual"] / expected, 1, places=10
        )
        self.assertEqual(report["objective_blocks"][0]["stop"], 38)

    def test_staged_mixed_calibration_recovers_known_conduction(self):
        truth = replace(Parameters(), thermal_k_w=0)
        voltage = [0.05, 0.1, 0.2, 0.4, 0.6]
        datasets = []
        for name, state in (("HRS", State()), ("LRS", State(0, 5e-9, 1e-9))):
            datasets.append(
                Dataset(
                    name,
                    voltage,
                    [operating_point(v, state, truth).current_a for v in voltage],
                    None,
                    "static",
                    state,
                )
            )
        specs = {
            "hopping_density_a_m2": {"initial": 7e12, "bounds": [2e12, 3e13]},
            "hopping_voltage_v": {"initial": 0.48, "bounds": [0.2, 0.7]},
            "conductivity_s_m": {"initial": 7e5, "bounds": [2e5, 3e6]},
        }
        blocks = [
            {
                "name": "hrs",
                "dataset": "HRS",
                "residual": "asinh",
                "current_scale_a": 1e-6,
            },
            {
                "name": "lrs",
                "dataset": "LRS",
                "residual": "linear",
                "current_scale_a": 1e-3,
            },
        ]
        settings = {
            "parameters": specs,
            "residual_blocks": blocks,
            "gtol": 1e-11,
            "jacobian_step_fraction": 5e-5,
            "joint_parameters": ["hopping_density_a_m2", "conductivity_s_m"],
            "stages": [
                {
                    "name": "hrs",
                    "parameters": list(specs)[:2],
                    "residual_blocks": blocks[:1],
                    "global_search": {"seed": 7, "maxiter": 2, "popsize": 3},
                },
                {
                    "name": "lrs",
                    "parameters": ["conductivity_s_m"],
                    "residual_blocks": blocks[1:],
                },
            ],
        }
        fitted, report = fit_parameters(truth, datasets, settings)
        self.assertEqual([s["name"] for s in report["stages"]], ["hrs", "lrs", "joint"])
        self.assertGreater(report["stages"][0]["global_search"]["nfev"], 0)
        self.assertIn("hopping_voltage_v", report["profiled_parameters"])
        self.assertEqual(report["calibrated_parameters"], list(specs))
        self.assertLess(
            report["final_squared_residual"], report["initial_squared_residual"] * 1e-6
        )
        for name in specs:
            self.assertAlmostEqual(
                getattr(fitted, name) / getattr(truth, name), 1, places=2
            )
        with self.assertRaisesRegex(ValueError, "全部原始测点"):
            fit_parameters(truth, datasets, {**settings, "residual_blocks": blocks[:1]})
        with tempfile.TemporaryDirectory() as directory:
            summary = export_results(
                directory, fitted, datasets, report=report, plot=False
            )
            self.assertEqual(summary["datasets"]["HRS"]["regions"]["hrs"]["n"], 5)
            self.assertLess(summary["datasets"]["LRS"]["rmse_a"], 1e-8)

    def test_invalid_fit_bound_is_rejected(self):
        dataset = Dataset("s", [0], [0], None, "static", State())
        with self.assertRaises(ValueError):
            fit_parameters(
                Parameters(),
                [dataset],
                {"parameters": {"generation_ev": {"initial": 1.2, "bounds": [2, 1]}}},
            )


if __name__ == "__main__":
    unittest.main()
