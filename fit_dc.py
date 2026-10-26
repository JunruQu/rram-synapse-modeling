"""DC 数据读取、参数校准和绘图；所有器件公式均调用 rram_model.py。

配置为 JSON，路径相对于配置文件。不同数据集共享 Parameters，可以各自
指定阻态、环境温度、限流及采样时间。CLI 帮助：python fit_dc.py --help。
NumPy/SciPy 只在拟合时导入，Matplotlib 只在绘图时导入，故无第三方依赖
也可生成示例数据、运行仿真和检查输入。拟合必须使用真实 SciPy 优化器。
"""

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, fields, replace
from itertools import pairwise
from pathlib import Path

from rram_model import ModelError, Parameters, State, operating_point, simulate_dc


@dataclass
class Dataset:
    name: str
    voltage_v: list[float]
    current_a: list[float]
    time_s: list[float] | None
    mode: str
    state: State
    compliance_a: float | None = None
    ambient_k: float | None = None
    weight: float = 1.0
    compliance_mode: str = "both"


def load_csv(path, *, dynamic=False, dwell_s=None):
    """读取 SI 单位 CSV，不排序、不补测量时间、不丢弃坏行。

    动态扫描允许用明确的每点间隔 dwell_s 生成时间；若有 time_s 列则必须
    二选一，以免配置中的时间和文件时间互相覆盖。相邻电压点线性连接。
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if not {"voltage_v", "current_a"}.issubset(headers):
            raise ValueError(f"{path}: 缺少 voltage_v/current_a 列（单位 V/A）")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: 数据为空")
    has_time = "time_s" in headers
    if dwell_s is not None:
        if not math.isfinite(dwell_s) or dwell_s <= 0:
            raise ValueError("dwell_s 必须为有限正数")
        if has_time:
            raise ValueError("time_s 列和 dwell_s 只能选择其一")
    voltage, current, times = [], [], []
    for line, row in enumerate(rows, 2):
        try:
            v, i = float(row["voltage_v"]), float(row["current_a"])
            t = (
                float(row["time_s"])
                if has_time
                else ((line - 2) * dwell_s if dwell_s is not None else None)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line}: 无法解析数值") from exc
        if not all(math.isfinite(x) for x in (v, i)) or (
            t is not None and not math.isfinite(t)
        ):
            raise ValueError(f"{path}:{line}: 不允许 NaN/Inf")
        voltage.append(v)
        current.append(i)
        if t is not None:
            times.append(t)
    if dynamic and (
        not times or len(times) < 2 or any(b <= a for a, b in pairwise(times))
    ):
        raise ValueError(
            f"{path}: 动态扫描需要至少两个严格递增的 time_s，或显式 dwell_s"
        )
    return voltage, current, times if times else None


def load_config(path):
    """加载共享参数和各曲线的条件；未知参数名直接由数据类拒绝。"""
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    allowed = {"parameters", "datasets", "solver", "fit", "description"}
    if set(config) - allowed:
        raise ValueError(f"未知配置字段: {sorted(set(config) - allowed)}")
    params = Parameters(**config.get("parameters", {}))
    datasets = []
    allowed_dataset = {
        "path",
        "name",
        "mode",
        "state",
        "dwell_s",
        "compliance_a",
        "ambient_k",
        "weight",
        "compliance_mode",
    }
    for index, item in enumerate(config.get("datasets", [])):
        if set(item) - allowed_dataset:
            raise ValueError(f"未知数据集字段: {sorted(set(item) - allowed_dataset)}")
        mode = item.get("mode", "dynamic")
        if mode not in ("static", "dynamic"):
            raise ValueError("mode 只能为 static 或 dynamic")
        v, i, t = load_csv(
            config_path.parent / item["path"],
            dynamic=mode == "dynamic",
            dwell_s=item.get("dwell_s"),
        )
        state = State(**item.get("state", {}))
        state.validate(params)
        weight = float(item.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("数据集 weight 必须为有限正数")
        ambient = item.get("ambient_k")
        if ambient is not None:
            replace(params, ambient_k=ambient)
        cc = item.get("compliance_a")
        if cc is not None and (not math.isfinite(cc) or cc <= 0):
            raise ValueError("compliance_a 必须为有限正数")
        compliance_mode = item.get("compliance_mode", "both")
        if compliance_mode not in ("both", "set"):
            raise ValueError("compliance_mode 必须为 both 或 set")
        datasets.append(
            Dataset(
                item.get("name", f"curve_{index}"),
                v,
                i,
                t,
                mode,
                state,
                cc,
                ambient,
                weight,
                compliance_mode,
            )
        )
    if not datasets:
        raise ValueError("配置至少需要一个数据集")
    if len({d.name for d in datasets}) != len(datasets):
        raise ValueError("数据集 name 不可重复")
    solver = config.get("solver", {})
    if set(solver) - {"rtol", "atol_m", "max_step_s", "max_steps"}:
        raise ValueError("solver 仅支持 rtol、atol_m、max_step_s、max_steps")
    return config, params, datasets


def predict(params, dataset, solver=None):
    """返回电热解和状态序列；静态拟合不会意外更新几何状态。"""
    p = (
        replace(params, ambient_k=dataset.ambient_k)
        if dataset.ambient_k is not None
        else params
    )
    dataset.state.validate(p)
    if dataset.mode == "dynamic":
        result = simulate_dc(
            dataset.time_s,
            dataset.voltage_v,
            dataset.state,
            p,
            compliance_a=dataset.compliance_a,
            compliance_mode=dataset.compliance_mode,
            **(solver or {}),
        )
        return result.points, result.states
    return (
        [
            operating_point(
                v,
                dataset.state,
                p,
                dataset.compliance_a,
                compliance_mode=dataset.compliance_mode,
            )
            for v in dataset.voltage_v
        ],
        [dataset.state] * len(dataset.voltage_v),
    )


def residual_values(predicted, measured, *, scale_a=1e-6, kind="asinh", weight=1.0):
    """有符号残差：asinh 保留负电流与零点，又减弱大电流支配。

    每条曲线除以 sqrt(N)，使曲线的总权重不单由采样点数决定。
    linear 使用相同的电流尺度，以得到无量纲残差。
    """
    if not math.isfinite(scale_a) or scale_a <= 0:
        raise ValueError("current_scale_a 必须为有限正数")
    if len(predicted) != len(measured) or len(predicted) == 0:
        raise ValueError("预测和观测必须等长且非空")
    if kind not in ("asinh", "linear"):
        raise ValueError("residual 必须为 asinh 或 linear")
    transform = math.asinh if kind == "asinh" else lambda x: x
    multiplier = math.sqrt(weight / len(measured))
    return [
        multiplier * (transform(a / scale_a) - transform(b / scale_a))
        for a, b in zip(predicted, measured, strict=True)
    ]


def _dependencies():
    try:
        import numpy as np
        from scipy.optimize import least_squares, minimize_scalar
    except ImportError as exc:
        raise RuntimeError(
            "拟合需要 NumPy/SciPy；请先运行 uv sync --group dev"
        ) from exc
    return np, least_squares, minimize_scalar


def fit_parameters(base, datasets, settings, solver=None):
    """有界最小二乘：只释放配置指定的参数，正值参数默认用对数坐标。

    数值/物理失败会中止并给出原因，不用常量罚值把失败伪装成成功拟合。
    Jacobian 的秩、条件数及相关性仅用于局部诊断，不代表唯一可辨识性。
    """
    np, least_squares, minimize_scalar = _dependencies()
    allowed = {
        "parameters",
        "current_scale_a",
        "residual",
        "loss",
        "max_nfev",
        "diff_step",
        "xtol",
        "ftol",
        "gtol",
        "initialization_points",
    }
    if set(settings) - allowed:
        raise ValueError(f"未知 fit 字段: {sorted(set(settings) - allowed)}")
    specs = settings.get("parameters", {})
    if not specs:
        raise ValueError("fit.parameters 至少指定一个待拟合参数")
    permitted = {f.name for f in fields(base)} - {
        "set_polarity",
        "temperature_limit_k",
        "parallel_ohm",
        "ambient_k",
    }
    if not set(specs).issubset(permitted):
        raise ValueError(f"不可拟合或未知的参数: {sorted(set(specs) - permitted)}")
    names, transforms, lower, upper, initial = [], [], [], [], []
    for name, spec in specs.items():
        if set(spec) - {"initial", "bounds", "log"}:
            raise ValueError(f"{name}: 仅支持 initial、bounds、log")
        lo, hi = map(float, spec["bounds"])
        value = float(spec.get("initial", getattr(base, name)))
        if (
            not all(math.isfinite(x) for x in (lo, hi, value))
            or not lo < hi
            or not lo <= value <= hi
        ):
            raise ValueError(f"{name}: 边界须有限且下限<上限，初值须在边界内")
        logarithmic = spec.get("log", lo > 0)
        if not isinstance(logarithmic, bool) or (logarithmic and lo <= 0):
            raise ValueError(f"{name}: 对数拟合需要正下界及布尔 log")
        for endpoint in (lo, hi):
            trial = replace(base, **{name: endpoint})
            for dataset in datasets:
                dataset.state.validate(trial)
        transform = math.log if logarithmic else lambda x: x
        names.append(name)
        transforms.append(logarithmic)
        lower.append(transform(lo))
        upper.append(transform(hi))
        initial.append(transform(value))

    def decode(values):
        return replace(
            base,
            **{
                name: math.exp(float(v)) if log else float(v)
                for name, v, log in zip(names, values, transforms, strict=True)
            },
        )

    scale = float(settings.get("current_scale_a", 1e-6))
    kind = settings.get("residual", "asinh")

    def residual(values):
        params = decode(values)
        result = []
        for dataset in datasets:
            try:
                points, _ = predict(params, dataset, solver)
            except (ValueError, ModelError) as exc:
                raise ModelError(
                    f"数据集 {dataset.name} 仿真失败: {exc}；请检查参数边界/初态"
                ) from exc
            result.extend(
                residual_values(
                    [p.current_a for p in points],
                    dataset.current_a,
                    scale_a=scale,
                    kind=kind,
                    weight=dataset.weight,
                )
            )
        return np.asarray(result, dtype=float)

    loss = settings.get("loss", "linear")
    loss_functions = {
        "linear": lambda z: z,
        "soft_l1": lambda z: 2 * z / (np.sqrt(1 + z) + 1),
        "huber": lambda z: np.where(z <= 1, z, 2 * np.sqrt(z) - 1),
        "cauchy": np.log1p,
        "arctan": np.arctan,
    }
    if loss not in loss_functions:
        raise ValueError("loss 必须为 linear、soft_l1、huber、cauchy 或 arctan")

    def score(values):
        # 与 least_squares 的默认 f_scale=1 使用同一个目标，避免鲁棒损失的
        # 初值搜索按普通平方和选出受异常点支配的参数。
        return float(0.5 * np.sum(loss_functions[loss](values * values)))

    initial_residual = residual(initial)
    dynamic = any(d.mode == "dynamic" for d in datasets)
    scan_points = settings.get("initialization_points", 17 if dynamic else 0)
    if type(scan_points) is not int or (scan_points != 0 and scan_points < 3):
        raise ValueError("initialization_points 必须为 0（关闭）或至少为 3 的整数")
    start = np.asarray(initial, dtype=float)
    best_score = score(initial_residual)
    best_residual = initial_residual
    scan_nfev = 0
    failed_trials = 0

    # 稀疏采样的突变 SET 回线会出现残差平台：参数变化未将开关时刻移过
    # 观测点时，电流几乎不变。仅靠局部有限差分可能 gtol 成功却未改善拟合。
    # 在声明的边界内逐坐标粗搜，再于最佳网格点邻域无导数细化，给局部
    # 最小二乘选择更好的起点。只接受更低目标值，不覆盖用户给定的边界。
    # 这不是全局最优保证，也不能补足缺失的开关区观测信息。
    if scan_points:
        for coordinate in range(len(names)):
            anchor = start.copy()

            def trial_score(value, anchor=anchor, coordinate=coordinate):
                nonlocal start, best_score, best_residual, scan_nfev, failed_trials
                trial = anchor.copy()
                trial[coordinate] = value
                scan_nfev += 1
                try:
                    trial_residual = residual(trial)
                except (ModelError, ValueError, OverflowError):
                    # 搜索阶段允许绕过无有效电热解的候选，但必须计入报告。
                    # 用户初态和最终局部求解仍然显式失败，不接受伪造残差。
                    failed_trials += 1
                    return math.inf
                value_score = score(trial_residual)
                if value_score < best_score:
                    start = trial
                    best_score, best_residual = value_score, trial_residual
                return value_score

            grid = np.linspace(lower[coordinate], upper[coordinate], scan_points)
            scores = [trial_score(value) for value in grid]
            winner = int(np.argmin(scores))
            if math.isfinite(scores[winner]):
                left = grid[max(0, winner - 1)]
                right = grid[min(scan_points - 1, winner + 1)]
                minimize_scalar(
                    trial_score,
                    bounds=(left, right),
                    method="bounded",
                    options={"xatol": 1e-6, "maxiter": 30},
                )

    # 状态积分的误差可能盖过默认的极小有限差分，因此动态拟合显式使用 1e-3。
    diff_step = float(settings.get("diff_step", 1e-3 if dynamic else 1e-5))
    if not math.isfinite(diff_step) or diff_step <= 0:
        raise ValueError("diff_step 必须为有限正数")
    max_nfev = settings.get("max_nfev", 100)
    if not isinstance(max_nfev, int) or max_nfev <= 0:
        raise ValueError("max_nfev 必须为正整数")
    result = least_squares(
        residual,
        start,
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=diff_step,
        loss=loss,
        max_nfev=max_nfev,
        xtol=settings.get("xtol", 1e-7),
        ftol=settings.get("ftol", 1e-7),
        gtol=settings.get("gtol", 1e-7),
    )
    singular = np.linalg.svd(result.jac, compute_uv=False)
    threshold = (
        (singular[0] * max(result.jac.shape) * np.finfo(float).eps)
        if len(singular)
        else 0
    )
    rank = int(np.sum(singular > threshold))
    condition = (
        float(singular[0] / singular[-1])
        if rank == len(names) and len(singular) and singular[-1] > threshold
        else None
    )
    correlated = []
    if rank == len(names):
        _, s, vh = np.linalg.svd(result.jac, full_matrices=False)
        covariance = (vh.T / s**2) @ vh
        diagonal = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(diagonal, diagonal)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if abs(correlation[i, j]) >= 0.95:
                    correlated.append(
                        {
                            "parameters": [names[i], names[j]],
                            "correlation": float(correlation[i, j]),
                        }
                    )
    boundary = [
        name
        for name, value, lo, hi in zip(names, result.x, lower, upper, strict=True)
        if min(value - lo, hi - value) <= 1e-4 * (hi - lo)
    ]
    report = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "fitted_parameters": names,
        "initial_squared_residual": float(initial_residual @ initial_residual),
        "final_squared_residual": float(result.fun @ result.fun),
        "initialization": {
            "method": "bounded_coordinate_scan" if scan_points else "none",
            "points_per_parameter": scan_points,
            "nfev": scan_nfev,
            "failed_trials": failed_trials,
            "parameters": {name: getattr(decode(start), name) for name in names},
            "squared_residual": float(best_residual @ best_residual),
        },
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "jacobian_rank": rank,
        "jacobian_condition": condition,
        "boundary_parameters": boundary,
        "high_correlations": correlated,
        "parameter_coordinates": {
            n: "log" if log else "linear"
            for n, log in zip(names, transforms, strict=True)
        },
        "note": "局部 Jacobian 诊断不证明全局唯一可辨识；成功标志仅表示优化器终止条件。",
    }
    return decode(result.x), report


def _write_json(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def export_results(
    output_dir, params, datasets, solver=None, report=None, *, plot=True
):
    """导出可复核的电流、几何、温度和限流结果；不将真实数据标签改成合成。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "parameters.json", asdict(params))
    predictions, statistics = [], {}
    for index, dataset in enumerate(datasets):
        points, states = predict(params, dataset, solver)
        predictions.append(points)
        errors = [
            p.current_a - i for p, i in zip(points, dataset.current_a, strict=True)
        ]
        statistics[dataset.name] = {
            "rmse_a": math.sqrt(sum(e * e for e in errors) / len(errors)),
            "max_abs_error_a": max(abs(e) for e in errors),
        }
        prediction_name = (
            "prediction.csv" if len(datasets) == 1 else f"prediction_{index}.csv"
        )
        with (output / prediction_name).open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "time_s",
                    "requested_v",
                    "terminal_v",
                    "internal_v",
                    "measured_a",
                    "predicted_a",
                    "switching_a",
                    "temperature_k",
                    "x_m",
                    "w_m",
                    "x0_m",
                    "gap_v",
                    "limited",
                ]
            )
            for row, (point, state) in enumerate(zip(points, states, strict=True)):
                writer.writerow(
                    [
                        dataset.time_s[row] if dataset.time_s else "",
                        point.requested_v,
                        point.terminal_v,
                        point.internal_v,
                        dataset.current_a[row],
                        point.current_a,
                        point.switching_a,
                        point.temperature_k,
                        state.x_m,
                        state.w_m,
                        state.x0_m,
                        point.gap_v,
                        point.limited,
                    ]
                )
    summary = dict(report or {"mode": "simulation"})
    summary["datasets"] = statistics
    _write_json(output / "report.json", summary)
    if plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "绘图需要 Matplotlib；请 uv sync 或使用 --no-plot。CSV/JSON 已导出。"
            ) from exc
        fig, axes = plt.subplots(
            len(datasets), 2, squeeze=False, figsize=(10, 3.5 * len(datasets))
        )
        for row, (dataset, points) in enumerate(
            zip(datasets, predictions, strict=True)
        ):
            simulated = [p.current_a for p in points]
            for axis in axes[row]:
                axis.plot(dataset.voltage_v, dataset.current_a, ".", ms=3, label="Data")
                axis.plot(dataset.voltage_v, simulated, "-", lw=1, label="Model")
                axis.set(
                    xlabel="Requested voltage (V)",
                    ylabel="Current (A)",
                    title=dataset.name,
                )
                axis.grid(True, alpha=0.25)
                axis.legend()
            axes[row, 1].set_yscale("symlog", linthresh=1e-8)
        fig.tight_layout()
        fig.savefig(output / "comparison.png", dpi=160)
        plt.close(fig)
    return summary


def run_config(config_path, output, *, fit=True, plot=True):
    config, params, datasets = load_config(config_path)
    solver = config.get("solver", {})
    initial_params = params
    report = None
    if fit:
        params, report = fit_parameters(params, datasets, config.get("fit", {}), solver)
        # 初始对比采用 fit.initial，而不只是 parameters 中的参考值。
        initial_params = replace(
            initial_params,
            **{
                name: spec.get("initial", getattr(initial_params, name))
                for name, spec in config["fit"]["parameters"].items()
            },
        )
    output = Path(output)
    if fit:
        export_results(output / "initial", initial_params, datasets, solver, plot=plot)
    export_results(output, params, datasets, solver, report, plot=plot)
    # 归档配置使用绝对输入路径，避免放到输出目录后相对路径指向错误文件。
    for item in config["datasets"]:
        item["path"] = str(
            (Path(config_path).resolve().parent / item["path"]).resolve()
        )
    _write_json(output / "config_used.json", config)
    return params, report


def create_demo(output_dir):
    """生成明确标为 SYNTHETIC 的固定阻态数据和 DC 回线，不冒充论文数字化数据。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261022)
    params = Parameters()
    _write_json(output / "synthetic_truth.json", asdict(params))

    def save(name, voltage, current, time=None):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (["time_s"] if time is not None else []) + ["voltage_v", "current_a"]
            )
            for index, (v, i) in enumerate(zip(voltage, current, strict=True)):
                noisy = i + rng.gauss(0, 0.01 * abs(i) + 1e-9)
                writer.writerow(
                    ([time[index]] if time is not None else []) + [v, noisy]
                )

    static_datasets = []
    for index, state in enumerate(
        [
            State(0.8e-9, 0.5e-9, 0.8e-9),
            State(1.2e-9, 0.5e-9, 1.2e-9),
            State(0, 5e-9, 1e-9),
        ]
    ):
        voltage = [-0.25 + 0.5 * k / 40 for k in range(41)]
        current = [operating_point(v, state, params).current_a for v in voltage]
        name = f"SYNTHETIC_static_{index}.csv"
        save(name, voltage, current)
        static_datasets.append(
            {
                "name": f"SYNTHETIC static {index}",
                "path": name,
                "mode": "static",
                "state": asdict(state),
            }
        )
    static_config = {
        "description": "SYNTHETIC: same-model pipeline validation; not experimental calibration.",
        "parameters": asdict(params),
        "datasets": static_datasets,
        "fit": {
            "parameters": {
                "hopping_density_a_m2": {"initial": 7e12, "bounds": [2e12, 3e13]},
                "hopping_voltage_v": {"initial": 0.48, "bounds": [0.25, 0.6]},
                "conductivity_s_m": {"initial": 7e5, "bounds": [2e5, 3e6]},
            },
            "current_scale_a": 1e-6,
            "max_nfev": 100,
        },
    }
    _write_json(output / "static_config.json", static_config)
    times, voltage = [], []
    knots = [(0, 0), (0.002, 0.6), (0.004, 0), (0.006, -2), (0.008, 0)]
    for segment, ((ta, va), (tb, vb)) in enumerate(pairwise(knots)):
        for k in range(41):
            if segment and k == 0:
                continue
            times.append(ta + (tb - ta) * k / 40)
            voltage.append(va + (vb - va) * k / 40)
    result = simulate_dc(times, voltage, params=params, rtol=2e-4, atol_m=2e-14)
    save("SYNTHETIC_dynamic.csv", voltage, [p.current_a for p in result.points], times)
    dynamic_config = {
        "description": "SYNTHETIC: conduction fixed; fit a small kinetic subset. No compliance in this example.",
        "parameters": asdict(params),
        "datasets": [
            {
                "name": "SYNTHETIC DC loop",
                "path": "SYNTHETIC_dynamic.csv",
                "mode": "dynamic",
                "state": asdict(State()),
            }
        ],
        "solver": {"rtol": 2e-4, "atol_m": 2e-14},
        "fit": {
            "parameters": {
                "generation_ev": {"initial": 1.15, "bounds": [1.05, 1.35]},
                "hopping_ev": {"initial": 0.95, "bounds": [0.85, 1.15]},
            },
            "current_scale_a": 1e-6,
            "diff_step": 1e-3,
            "max_nfev": 80,
        },
    }
    _write_json(output / "dynamic_config.json", dynamic_config)
    return output / "static_config.json", output / "dynamic_config.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("fit", "simulate"):
        child = sub.add_parser(command)
        child.add_argument("config", type=Path)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--no-plot", action="store_true")
    demo = sub.add_parser("demo", help="生成合成数据；--fit 顺序进行传导和动力学校准")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--fit", action="store_true")
    demo.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            static, dynamic = create_demo(args.output)
            print(f"合成数据/配置已生成：{args.output}")
            if args.fit:
                calibrated, static_report = run_config(
                    static, args.output / "static_fit", plot=not args.no_plot
                )
                if not static_report["success"]:
                    print(
                        "传导优化尚未成功终止；请查看报告，未开始动力学校准",
                        file=sys.stderr,
                    )
                    return 2
                config = json.loads(dynamic.read_text(encoding="utf-8"))
                for name in static_report["fitted_parameters"]:
                    config["parameters"][name] = getattr(calibrated, name)
                _write_json(dynamic, config)
                _, report = run_config(
                    dynamic, args.output / "dynamic_fit", plot=not args.no_plot
                )
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return 0 if report["success"] else 2
            return 0
        _, report = run_config(
            args.config, args.output, fit=args.command == "fit", plot=not args.no_plot
        )
        print(f"结果已导出：{args.output}")
        if report:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report is None or report["success"] else 2
    except (
        ValueError,
        TypeError,
        KeyError,
        OSError,
        RuntimeError,
        OverflowError,
    ) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
