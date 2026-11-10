"""各组独立参数，按缩短后的实测协议连续仿真。"""

import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fit_ltp import prepare
from rram_model import Parameters, State, TransientOptions, simulate_transient


def replay(t, v, cycles, config, level):
    started = time.perf_counter()
    result = simulate_transient(
        t,
        v,
        State(**config["initial_state"]),
        Parameters(**config["parameters"]),
        TransientOptions(**config["transient_options"]),
        initial_temperature_k=300,
        rtol=1e-4 / 2**level,
        atol_m=1e-13 / 2**level,
        max_step_s=25e-9 / 2**level,
        max_steps=20000000,
    )
    indices = np.searchsorted(
        t, np.arange(cycles)[:, None] * 2e-6 + np.linspace(1.15e-6, 1.45e-6, 7)
    )
    currents = np.array([o.current_a for o in result.points])
    g = currents[indices].mean(axis=1) / v[indices].mean(axis=1)
    sample = indices[:, 3]
    starts = np.searchsorted(t, np.arange(cycles) * 2e-6)
    ends = np.searchsorted(t, (np.arange(cycles) + 1) * 2e-6)
    values = np.array(
        [
            [
                result.states[i].x_m,
                result.states[i].w_m,
                result.states[i].x0_m,
                g[k],
                result.points[i].temperature_k,
                max(result.interval_peak_temperature_k[a:b]),
            ]
            for k, (i, a, b) in enumerate(zip(sample, starts, ends, strict=True))
        ]
    )
    return (
        result,
        values,
        dict(
            level=level,
            rtol=1e-4 / 2**level,
            max_step_s=25e-9 / 2**level,
            elapsed_s=time.perf_counter() - started,
            accepted_steps=result.accepted_steps,
        ),
    )


def main():
    root = Path("outputs/ltp_ltd_simulate")
    root.mkdir(exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(11, 13))
    summaries = []
    for col, width in enumerate([100, 1]):
        out, observed, t, v = prepare(width, root)
        source = Path(f"outputs/ltp_ltd_hires/{width}ns")
        config = dict(
            parameters=json.loads((source / "parameters.json").read_text()),
            initial_state=json.loads((source / "initial_state.json").read_text()),
            transient_options=json.loads((source / "fit_config.json").read_text())[
                "transient_options"
            ],
        )
        original_half = len(observed) // 2
        count = {100: 10, 1: 1000}[width]
        # 保留两个阶段各自前N周期的波形，RESET时间平移，状态不重置。
        boundary = original_half * 2e-6
        first = t <= count * 2e-6
        second = (t > boundary) & (t <= boundary + count * 2e-6)
        t = np.r_[t[first], t[second] - boundary + count * 2e-6]
        v = np.r_[v[first], v[second]]
        observed = np.r_[
            observed[:count], observed[original_half : original_half + count]
        ]
        cycles = 2 * count
        result, values, stats = replay(t, v, cycles, config, 0)
        checks = []
        tolerance = np.array([1e-12, 1e-12, 1e-12, 1e-8, 0.1, 0.1])
        for level in range(1, 7):
            fine, fine_values, fine_stats = replay(t, v, cycles, config, level)
            ratio = float(
                np.max(
                    abs(values - fine_values) / (0.01 * abs(fine_values) + tolerance)
                )
            )
            passed = ratio <= 1
            checks.append(
                dict(
                    level=level,
                    passed=passed,
                    max_tolerance_ratio=ratio,
                    **{f"refined_{k}": x for k, x in fine_stats.items()},
                )
            )
            if passed:
                break
            print(width, "refining", level, ratio, flush=True)
            result, values, stats = fine, fine_values, fine_stats
        summary = dict(
            width_ns=width,
            parameter_source=str(source),
            parameters=config["parameters"],
            initial_state=config["initial_state"],
            transient_options=config["transient_options"],
            source_cycles_per_phase=original_half,
            selected_source_cycles_0based=[
                list(range(count)),
                list(range(original_half, original_half + count)),
            ],
            cycles=cycles,
            writes_per_phase=cycles // 2,
            read_window_s=[1.15e-6, 1.45e-6],
            numerical_checks=checks,
            convergence_pass=passed,
            **stats,
            warning="各组最终独立参数预测，非重新拟合；缩短后无对应实测验证；1 ns WRITE重建仍有不确定性",
        )
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )
        with (out / "prediction.csv").open("w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "pulse",
                    "phase",
                    "x_m",
                    "w_m",
                    "x0_m",
                    "predicted_g_s",
                    "temperature_k",
                    "peak_temperature_k",
                ]
            )
            for k, row in enumerate(values):
                writer.writerow([k + 1, "SET" if k < cycles // 2 else "RESET", *row])
        with (out / "trajectory.csv").open("w") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["time_s", "voltage_v", "x_m", "w_m", "x0_m", "temperature_k"]
            )
            writer.writerows(
                [time, voltage, s.x_m, s.w_m, s.x0_m, o.temperature_k]
                for time, voltage, s, o in zip(
                    t, v, result.states, result.points, strict=True
                )
            )
        for phase, part in [
            ("SET", slice(0, cycles // 2)),
            ("RESET", slice(cycles // 2, cycles)),
        ]:
            pulses = np.arange(1, cycles + 1)[part]
            for row, index, scale in [(0, 0, 1e9), (1, 1, 1e9), (2, 3, 1e6), (3, 4, 1)]:
                axes[row, col].plot(pulses, values[part, index] * scale, label=phase)
            axes[3, col].plot(
                pulses, values[part, 5], linestyle=":", label=f"{phase} peak"
            )

        for row, label in enumerate(
            ["x gap (nm)", "w diameter (nm)", "READ G (µS)", "T (K)"]
        ):
            axes[row, col].set(
                xlabel="Cumulative pulse", ylabel=label, title=f"{width} ns"
            )
            axes[row, col].legend()
        summaries.append(summary)
        print(
            width,
            stats["elapsed_s"],
            stats["accepted_steps"],
            passed,
            flush=True,
        )
        if not passed:
            raise RuntimeError(f"{width} ns convergence failed")
    fig.tight_layout()
    fig.savefig(root / "comparison.png")
    (root / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
