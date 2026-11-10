"""连续 LTP/LTD 定性校准；原始文件不改写。"""

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from rram_model import (
    KB_EV,
    Parameters,
    State,
    TransientOptions,
    _fixed_temperature_point,
    simulate_transient,
)

INITIAL_WIDTH_BOUNDS_NM = (0.5, 2.5)


def prepare(width, root):
    path = Path(f"data/ltp_ltd/ltp_ltd_{width}ns.csv")
    a = np.loadtxt(path, delimiter=",")
    if a.shape[1] != 3 or not np.isfinite(a).all() or np.any(np.diff(a[:, 0]) <= 0):
        raise ValueError(f"Invalid input: {path}")
    out = root / f"{width}ns"
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        out / "prepared.csv",
        np.column_stack((np.arange(len(a)), a, a[:, 2] / 50, a[:, 1] - a[:, 2])),
        delimiter=",",
        header="source_row,time_s,vforce_v,vsense_v,current_a,dut_voltage_v",
        comments="",
    )
    cycles = round(a[-1, 0] / 2e-6)
    local = a[:, 0] % 2e-6
    ids = np.floor(a[:, 0] / 2e-6).astype(int)
    mask = (local >= 1.15e-6) & (local <= 1.45e-6)
    read = []
    for k in range(cycles):
        start, stop = np.searchsorted(ids, [k, k + 1])
        window = a[start:stop][mask[start:stop]]
        g = window[:, 2] / 50 / (window[:, 1] - window[:, 2])
        if not len(g):
            raise ValueError(f"Missing READ {k}")
        read.append(
            (
                np.mean(window[:, 2] / 50) / np.mean(window[:, 1] - window[:, 2]),
                np.median(abs(g - np.median(g))),
                len(g),
            )
        )
    np.savetxt(
        out / "reads.csv",
        read,
        delimiter=",",
        header="conductance_s,mad_s,rows",
        comments="",
    )
    dt = np.diff(a[:, 0])
    nominal = np.median(dt)
    audit = dict(
        source=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        rows=len(a),
        cycles=cycles,
        dt_s=nominal,
        irregular_intervals=np.flatnonzero(abs(dt - nominal) > 0.02 * nominal).tolist(),
        initial_read_available=False,
        read_window_s=[1.15e-6, 1.45e-6],
        conductance_definition="mean(current_a) / mean(dut_voltage_v)",
        write_reconstruction="nominal source trapezoid minus interpolated measured sense voltage; uncertain especially at 1ns",
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    times, volts = reconstruct_drive(a, width, cycles)
    return out, np.array(read)[:, 0], times, volts


def reconstruct_drive(a, width, cycles):
    """复用已确认时序；欠采样写入按标称波形重建。"""
    # 每周期仅保留输入波形节点；READ和WRITE全部保留。
    w = width * 1e-9
    offsets = np.r_[
        0,
        0.1 * w,
        w,
        1.1 * w,
        1e-6,
        1.05e-6,
        np.linspace(1.15e-6, 1.45e-6, 7),
        1.5e-6,
        1.55e-6,
        2e-6,
    ]
    times = []
    volts = []
    for k in range(cycles):
        ts = k * 2e-6 + offsets
        amplitude = 1.2 if k < cycles // 2 else -1.3
        force = np.interp(
            offsets,
            [0, 0.1 * w, w, 1.1 * w, 1e-6, 1.05e-6, 1.5e-6, 1.55e-6, 2e-6],
            [0, amplitude, amplitude, 0, 0, -0.4, -0.4, 0, 0],
        )
        v = force - np.interp(ts, a[:, 0], a[:, 2])
        v[force == 0] = 0
        times.extend(ts if k == 0 else ts[1:])
        volts.extend(v if k == 0 else v[1:])
    return np.array(times), np.array(volts)


def read_trend_residual(pred, observed, groups, trend_scale):
    """趋势窗口之外显式约束首末READ，避免首个窗口中位数掩盖起点误差。"""
    shape = (
        np.array([np.median(pred[group]) for group in groups])
        - np.array([np.median(observed[group]) for group in groups])
    ) / trend_scale
    half = len(groups) // 2
    endpoints = []
    count = len(observed) // 2
    for start, offset in [(0, 0), (half, count)]:
        endpoints.extend(
            [
                shape[start],
                shape[start + half - 1],
                shape[start + half - 1] - shape[start],
            ]
        )
        first = (pred[offset] - observed[offset]) / trend_scale[start]
        last = (pred[offset + count - 1] - observed[offset + count - 1]) / trend_scale[
            start
        ]
        endpoints.extend([first, last, last - first])
        for index in [1, 2, 4, 9, 19]:
            if index < count:
                endpoints.append(
                    (pred[offset + index] - observed[offset + index])
                    / trend_scale[start]
                )
    return np.r_[shape, 3 * np.array(endpoints)]


def decode_ltp(z, reference, state):
    """优化使用nm及室温电导率坐标；初态变化不修改材料几何边界。"""
    par = replace(
        reference,
        generation_ev=z[0],
        release_ev=z[1],
        conductivity_s_m=np.exp(z[2] - reference.conduction_ev / (KB_EV * 300)),
        generation_alpha_m=z[4] * 1e-9,
        recombination_ev=z[5],
    )
    initial = replace(state, x_m=z[3] * 1e-9, w_m=z[6] * 1e-9)
    initial.validate(par)
    return par, initial


def load_initialization(width):
    """独立保存本轮初值；不依赖已删除的历史输出。"""
    path = Path(__file__).resolve().parent / "configs/ltp_ltd_hires.json"
    return json.loads(path.read_text())[str(width)]


def width_seeds(z):
    seeds = [z.copy()]
    for diameter in [0.65, 0.8, 1.0, 1.5, 2.5]:
        seed = z.copy()
        seed[6] = diameter
        seeds.append(seed)
    return seeds


def fit(width, root, max_nfev):
    out, g, t, v = prepare(width, root)
    warm = load_initialization(width)
    p = Parameters(**warm["parameters"])
    state = State(**warm["initial_state"])
    options = TransientOptions(**warm["transient_options"])
    n = len(g) // 2
    # 只比较每阶段20个窗口的READ中位趋势，原始逐脉冲数据仍完整保留。
    groups = [
        part + offset
        for offset in [0, n]
        for part in np.array_split(np.arange(n), min(n, 20))
    ]
    target = np.array([np.median(g[group]) for group in groups])
    trend_scale = np.array(
        [max(np.ptp(target[: len(groups) // 2]), 1e-5)] * (len(groups) // 2)
        + [max(np.ptp(target[len(groups) // 2 :]), 1e-5)] * (len(groups) // 2)
    )
    scale = np.r_[
        np.full(n, max(np.ptp(g[:n]), 1e-5)), np.full(n, max(np.ptp(g[n:]), 1e-5))
    ]

    def predict(z, rtol=1e-3, step=50e-9):
        par, initial = decode_ltp(z, p, state)
        result = simulate_transient(
            t,
            v,
            initial,
            par,
            options,
            rtol=rtol,
            max_step_s=step,
            max_steps=20000000,
        )
        queries = (
            np.arange(len(g))[:, None] * 2e-6
            + np.linspace(1.15e-6, 1.45e-6, 7)[None, :]
        )
        indices = np.searchsorted(t, queries)
        currents = np.array(
            [
                _fixed_temperature_point(
                    v[i], result.states[i], par, result.points[i].temperature_k
                ).current_a
                for i in indices.ravel()
            ]
        ).reshape(indices.shape)
        values = currents.mean(axis=1) / v[indices].mean(axis=1)
        return values, result, par, initial

    z = [
        p.generation_ev,
        p.release_ev,
        np.log(p.conductivity_s_m) + p.conduction_ev / (KB_EV * 300),
        state.x_m * 1e9,
    ]
    low = [1.1, 0.6, np.log(1e5), 0]
    high = [2.2, 1.8, np.log(2.5e6), state.x0_m * 1e9]
    z += [p.generation_alpha_m * 1e9, p.recombination_ev, state.w_m * 1e9]
    low += [0.2, 0.45, INITIAL_WIDTH_BOUNDS_NM[0]]
    high += [1.5, 0.85, INITIAL_WIDTH_BOUNDS_NM[1]]
    z = np.clip(z, low, high)

    def residual(z):
        pred, *_ = predict(z)
        return read_trend_residual(pred, g, groups, trend_scale)

    seeds = width_seeds(z)
    screening = []
    for seed in seeds:
        error = residual(seed)
        score = float(np.sum(0.01 * (np.sqrt(1 + (error / 0.1) ** 2) - 1)))
        screening.append(
            dict(
                initial_width_nm=float(seed[6]),
                coordinates=seed.tolist(),
                full_cost=score,
            )
        )
        print(width, "diameter seed", seed[6], "full cost", score, flush=True)
    (out / "seed_screen.json").write_text(json.dumps(screening, indent=2))
    seed = seeds[int(np.argmin([s["full_cost"] for s in screening]))]
    opt = least_squares(
        residual,
        seed,
        bounds=(low, high),
        loss="soft_l1",
        f_scale=0.1,
        ftol=0.03,
        xtol=1e-4,
        x_scale="jac",
        max_nfev=max_nfev,
        diff_step=0.001,
    )
    print(width, "cost", opt.cost, "nfev", opt.nfev, flush=True)
    pred, result, par, initial = predict(opt.x)
    refined, *_ = predict(opt.x, 3e-4, 25e-9)
    numerical_checks = [
        dict(
            rtol=3e-4,
            max_step_s=25e-9,
            pass_check=bool(np.all(abs(pred - refined) <= 2e-6 + 0.02 * abs(refined))),
            max_difference_us=float(np.max(abs(pred - refined)) * 1e6),
        )
    ]
    if not numerical_checks[0]["pass_check"]:
        previous = refined
        for tolerance, step in [(1e-4, 12.5e-9), (3e-5, 6.25e-9)]:
            tighter, *_ = predict(opt.x, tolerance, step)
            numerical_checks.append(
                dict(
                    rtol=tolerance,
                    max_step_s=step,
                    pass_check=bool(
                        np.all(abs(previous - tighter) <= 2e-6 + 0.02 * abs(tighter))
                    ),
                    max_difference_us=float(np.max(abs(previous - tighter)) * 1e6),
                )
            )
            previous = tighter
    trends = {}
    for phase, section, sign in [
        ("set", slice(0, n), 1),
        ("reset", slice(n, None), -1),
    ]:
        observed, modeled = g[section], pred[section]
        size = max(1, len(observed) // 10)
        delta = float(np.median(modeled[-size:]) - np.median(modeled[:size]))
        measured_delta = float(np.median(observed[-size:]) - np.median(observed[:size]))
        ratio = delta / measured_delta if measured_delta else None
        trends[phase] = dict(
            measured_change_s=measured_delta,
            predicted_change_s=delta,
            change_ratio=ratio,
            largest_step_fraction=float(
                np.max(abs(np.diff(modeled))) / max(np.ptp(modeled), 1e-12)
            ),
            jump_then_plateau=bool(
                (
                    np.max(abs(np.diff(modeled))) > 0.25 * max(np.ptp(modeled), 1e-12)
                    and np.ptp(modeled[-max(2, len(modeled) // 4) :])
                    < 0.05 * max(np.ptp(modeled), 1e-12)
                )
                or (
                    abs(modeled[max(1, len(modeled) // 5)] - modeled[0])
                    > 0.8 * max(np.ptp(modeled), 1e-12)
                    and np.ptp(modeled[len(modeled) // 2 :])
                    < 0.05 * max(np.ptp(modeled), 1e-12)
                )
            ),
            trend_pass=bool(
                sign * delta > 0 and ratio is not None and 0.5 <= ratio <= 2
            ),
        )
    report = dict(
        fit_initial_width=True,
        initial_width_bounds_nm=[low[6], high[6]],
        warm_source="configs/ltp_ltd_hires.json",
        warm_origin=warm["warm_origin"],
        first_read_predicted_us=float(pred[0] * 1e6),
        first_read_measured_us=float(g[0] * 1e6),
        first_write_x_nm=result.states[
            np.argmin(abs(np.asarray(result.time_s) - width * 1.1e-9))
        ].x_m
        * 1e9,
        read_trends=trends,
        objective="mean READ current / mean READ voltage, 1.15–1.45 us; trend bins plus endpoints and update range; no WRITE current residual",
        parameter_names=[
            "generation_ev",
            "release_ev",
            "log_sigma300",
            "initial_gap_nm",
            "generation_alpha_nm",
            "recombination_ev",
            "initial_width_nm",
        ],
        numerical_checks=numerical_checks,
        final_width_nm=result.states[-1].w_m * 1e9,
        first_read_bias_us=float((pred[0] - g[0]) * 1e6),
        optimizer_status=int(opt.status),
        optimizer_cost=float(opt.cost),
        success=bool(opt.success),
        bound_hits=[
            i
            for i, (value, lo, hi) in enumerate(zip(opt.x, low, high, strict=True))
            if min(value - lo, hi - value) < 0.01 * (hi - lo)
        ],
        qualitative_target_pass=bool(
            all(value["trend_pass"] for value in trends.values())
        ),
        message=opt.message,
        nfev=opt.nfev,
        normalized_rmse=float(np.sqrt(np.mean(((pred - g) / scale) ** 2))),
        convergence_pass=bool(
            np.all(abs(pred - refined) <= 2e-6 + 0.02 * abs(refined))
        ),
        parameters=par.__dict__,
        initial_state=initial.__dict__,
        warning="conditional trend fit; reconstructed write waveform; geometry and temperature not independently measured",
    )
    config = dict(
        width_ns=width,
        max_nfev=max_nfev,
        fit_initial_width=True,
        initialization=warm,
        warm_source="configs/ltp_ltd_hires.json",
        warm_coordinates=z.tolist(),
        selected_seed=seed.tolist(),
        final_coordinates=opt.x.tolist(),
        lower_bounds=low,
        upper_bounds=high,
        transient_options=options.__dict__,
        rtol=1e-3,
        max_step_s=50e-9,
        max_steps=20000000,
        read_window_s=[1.15e-6, 1.45e-6],
        read_samples=7,
        trend_bins_per_phase=min(n, 20),
        optimizer=dict(
            loss="soft_l1",
            f_scale=0.1,
            ftol=0.03,
            xtol=1e-4,
            gtol=1e-8,
            x_scale="jac",
            diff_step=0.001,
        ),
    )
    (out / "fit_config.json").write_text(json.dumps(config, indent=2))
    (out / "parameters.json").write_text(json.dumps(par.__dict__, indent=2))
    (out / "initial_state.json").write_text(json.dumps(initial.__dict__, indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    np.savetxt(
        out / "prediction.csv",
        np.column_stack((np.arange(1, len(g) + 1), g, pred)),
        delimiter=",",
        header="cumulative_pulse,measured_s,predicted_s",
        comments="",
    )
    with (out / "trajectory.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "x_m", "w_m", "x0_m", "temperature_k"])
        writer.writerows(
            (time, s.x_m, s.w_m, s.x0_m, o.temperature_k)
            for time, s, o in zip(t, result.states, result.points, strict=True)
        )
    plt.figure()
    plt.plot(np.arange(1, len(g) + 1), g * 1e6, label="measured")
    plt.plot(np.arange(1, len(g) + 1), pred * 1e6, label="model")
    plt.legend()
    plt.xlabel("Cumulative pulse")
    plt.ylabel("READ G (uS)")
    plt.savefig(out / "fit.png")
    plt.close()
    indices = np.searchsorted(t, np.arange(len(g)) * 2e-6 + 1.3e-6)
    fig, axes = plt.subplots(3, 1, sharex=True)
    for axis, values, label in zip(
        axes,
        [
            [result.states[i].x_m * 1e9 for i in indices],
            [result.states[i].w_m * 1e9 for i in indices],
            [result.points[i].temperature_k for i in indices],
        ],
        ["Gap x (nm)", "Diameter w (nm)", "Model sampled READ T (K)"],
        strict=True,
    ):
        axis.plot(np.arange(1, len(g) + 1), values)
        axis.set_ylabel(label)
    axes[-1].set_xlabel("Cumulative pulse")
    fig.tight_layout()
    fig.savefig(out / "states.png")
    plt.close(fig)
    print(width, report["normalized_rmse"], report["convergence_pass"], flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/ltp_ltd_hires"))
    parser.add_argument("--max-nfev", type=int, default=20)
    parser.add_argument("--width", type=int, choices=[100, 1])
    parser.add_argument(
        "--unified", action="store_true", help="统一策略（默认；保留原命令兼容）"
    )
    args = parser.parse_args(argv)
    for width in [args.width] if args.width else [100, 1]:
        fit(width, args.output, min(args.max_nfev, 20))
    summarize(args.output, [args.width] if args.width else [100, 1])


def summarize(root, widths):
    reports = {}
    lines = [
        "# HiRes LTP/LTD 统一策略条件校准",
        "",
        "READ：1.15–1.45 μs 平均电流/平均器件电压；每组独立参数，连续 SET→RESET。",
        "初始有效直径边界 0.5–2.5 nm，最终扩径上限 5 nm。",
        "",
        "| ns | 归一化 RMSE | 首 READ 偏差 μS | 初始/最终直径 nm | SET/LTD 趋势 | 触边参数（1%范围内） | 数值对照 | 突变后平台 SET/LTD |",
        "|---|---|---|---|---|---|---|---|",
    ]
    fig, axes = plt.subplots(
        1, len(widths), figsize=(5 * len(widths), 4), squeeze=False
    )
    for width, axis in zip(widths, axes[0], strict=True):
        out = root / f"{width}ns"
        reports[str(width)] = json.loads((out / "report.json").read_text())
        report = reports[str(width)]
        trends = report["read_trends"]
        hits = [report["parameter_names"][i] for i in report["bound_hits"]]
        lines.append(
            f"| {width} | {report['normalized_rmse']:.2%} | "
            f"{report['first_read_bias_us']:+.2f} | "
            f"{report['initial_state']['w_m'] * 1e9:.3f}/{report['final_width_nm']:.3f} | "
            f"{trends['set']['trend_pass']}/{trends['reset']['trend_pass']} | "
            f"{', '.join(hits) or '无'} | {report['convergence_pass']} | "
            f"{trends['set']['jump_then_plateau']}/{trends['reset']['jump_then_plateau']} |"
        )
        values = np.loadtxt(out / "prediction.csv", delimiter=",", skiprows=1)
        axis.plot(values[:, 0], values[:, 1] * 1e6, label="measured")
        axis.plot(values[:, 0], values[:, 2] * 1e6, label="model")
        axis.set(title=f"{width} ns", xlabel="Cumulative pulse", ylabel="READ G (uS)")
        axis.legend()
    fig.tight_layout()
    fig.savefig(root / "comparison.png")
    plt.close(fig)
    (root / "summary.json").write_text(json.dumps(reports, indent=2))
    lines.extend(
        [
            "",
            "趋势通过不表示逐点吻合或完成物理校准；优化器终止原因、筛选代价和完整参数见各组 JSON。",
            "突变/快速饱和后平台诊断：相邻跃变超过全幅25%且末四分之一范围小于5%；或前20%脉冲完成超过80%全幅且后半程范围小于5%。仅作曲线形状提示。",
            "数值门槛为逐 READ 2%＋2 μS；初次对照失败不会被后续更严格重放覆盖，详见 numerical_checks。",
            "温度为模型在每周期1.3 μs的采样值。参数是条件校准值；拟合不证明真实细丝尺寸或脉宽的因果作用。",
            "1 ns WRITE波形依赖欠采样重建；独立形貌、温度及输入波形验证仍待完成。",
        ]
    )
    (root / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
