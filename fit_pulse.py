"""SET–READ–RESET–READ 数据整理、观测模型与校准，默认使用 HiRes。

器件方程均在 rram_model.py。此处仅负责单位/极性转换、导出时间精度修复、
有限积分窗口、通道响应及优化。原始数据永不覆盖；插值不用于恢复
未测的脉冲前沿或亚采样开关时刻。
"""

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares

from rram_model import (
    KB_EV,
    ModelError,
    Parameters,
    State,
    TransientOptions,
    operating_point,
    simulate_transient,
)


@dataclass
class PulseData:
    name: str
    time_s: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    width_ns: int
    instrument: str
    integration_s: float
    source_sha256: str | None = None


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def rebuild_time(values, allow_reset=False):
    """只修复有原始前段采样节拍证据的舍入时间，不对波形按时间排序。

    重复时间保留所有行。独立采集的 HiRes 两个半周期按用户确认的无额外
    间隔连接；每段的原始相位/舍入误差单独核对，连接处保持一个采样间隔。
    """
    raw = np.asarray(values, float)
    if len(raw) < 3 or not np.all(np.isfinite(raw)):
        raise ValueError("时间数据不完整")
    cuts = np.flatnonzero(np.diff(raw) < 0) + 1
    if len(cuts) and not allow_reset:
        raise ValueError("出现未经允许的时间归零")
    limits = [0, *cuts.tolist(), len(raw)]
    intervals = []
    for lo, hi in pairwise(limits):
        d = np.diff(raw[lo : min(hi, lo + 30)])
        if np.any(d <= 0):
            raise ValueError("前段时间不能提供明确的采样节拍")
        intervals.append(float(np.median(d)))
    dt = float(np.median(intervals))
    if not np.allclose(intervals, dt, rtol=1e-5, atol=1e-18):
        raise ValueError("不同采集段采样间隔不同，不能自动连接")
    # 用原 CSV 数字表示的小数位检查每一行是否与均匀采样舍入相容。
    from decimal import Decimal

    for lo, hi in pairwise(limits):
        reconstructed = raw[lo] + np.arange(hi - lo) * dt
        tolerances = np.array(
            [
                0.51001 * float(Decimal(1).scaleb(Decimal(s).as_tuple().exponent))
                for s in values[lo:hi]
            ]
        )
        if np.any(
            np.abs(reconstructed - raw[lo:hi]) > tolerances + max(dt * 1e-5, 1e-18)
        ):
            raise ValueError("时间偏差超过导出精度，不能假定均匀采样")
    return raw[0] + np.arange(len(raw)) * dt, {
        "sampling_interval_s": dt,
        "duplicate_raw_timestamps": int(np.sum(np.diff(raw) == 0)),
        "time_reset_rows_zero_based": cuts.tolist(),
        "method": "uniform cadence inferred from unrounded leading samples, validated against CSV decimal precision; retain row order",
        "segment_join": "one sample interval, no extra delay" if len(cuts) else None,
    }


def prepare(
    input_dir="data/pulse", output_dir="data/pulse/prepared", source_names=None
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit = {
        "conditions": {
            "b1530_current": "I_DUT = -source current_a",
            "hires_current": "I_DUT = source vsense_v / 50 ohm",
            "hires_voltage": "V_DUT = source vforce_v - source vsense_v",
            "compliance": "none, both devices",
            "b1530_integration_s": 10e-9,
            "devices": "device identity is not inferred from the instrument name",
            "hires_join": "SET-READ then RESET-READ, user confirmed no extra interval",
            "voltage_input": "measured DUT differential voltage, linearly interpolated; no sub-sampling waveform deconvolution",
        },
        "files": [],
    }
    data = []
    sources = sorted(Path(input_dir).glob("pulse_iv_*.csv"))
    if source_names is not None:
        requested = set(source_names)
        sources = [source for source in sources if source.name in requested]
        missing = requested - {source.name for source in sources}
        if missing:
            raise FileNotFoundError(f"未找到指定脉冲数据: {sorted(missing)}")
    for source in sources:
        hires = "HiRes" in source.name
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            expected_fields = (
                ["time_s", "vforce_v", "vsense_v"]
                if hires
                else ["time_s", "voltage_v", "current_a"]
            )
            if reader.fieldnames != expected_fields:
                raise ValueError(f"{source}: 列名不符")
            rows = list(reader)
        array = np.array([[float(r[k]) for k in expected_fields] for r in rows])
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{source}: NaN/Inf")
        ts, info = rebuild_time([r["time_s"] for r in rows], allow_reset=hires)
        v = array[:, 1] - array[:, 2] if hires else array[:, 1]
        current = array[:, 2] / 50 if hires else -array[:, 2]
        width = int(source.stem.split("_")[2].removesuffix("ns"))
        instrument = "hires" if hires else "b1530"
        prepared_name = f"{width}ns_{instrument}_prepared.csv"
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        datum = PulseData(
            source.stem,
            ts,
            v,
            current,
            width,
            instrument,
            0 if hires else 10e-9,
            source_hash,
        )
        data.append(datum)
        with (output / prepared_name).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "time_s",
                    "voltage_v",
                    "current_a",
                    "source_time_s",
                    "source_vforce_v" if hires else "source_voltage_v",
                    "source_vsense_v" if hires else "source_current_a",
                ]
            )
            writer.writerows(
                zip(ts, v, current, array[:, 0], array[:, 1], array[:, 2], strict=True)
            )
        info.update(
            name=source.name,
            prepared_file=prepared_name,
            source_sha256=source_hash,
            rows=len(rows),
            instrument=instrument,
            programming_width_ns=width,
            voltage_range_v=[float(v.min()), float(v.max())],
            current_range_a=[float(current.min()), float(current.max())],
            resolution_warning="50 ns sampling cannot resolve 20/50 ns pulse shape or precise switching delay"
            if not hires and width <= 50
            else None,
        )
        audit["files"].append(info)
    if not data:
        raise ValueError("未找到脉冲 CSV")
    if not any(d.instrument == "b1530" for d in data):
        audit["conditions"].pop("b1530_current")
        audit["conditions"].pop("b1530_integration_s")
        audit["conditions"]["devices"] = (
            "HiRes oscilloscope data; device identity is not inferred from filename"
        )
        audit["conditions"]["compliance"] = "none"
    if source_names is not None and (output / "audit.json").exists():
        previous = json.loads((output / "audit.json").read_text())
        selected_names = {item["name"] for item in audit["files"]}
        retained = [
            item
            for item in previous.get("files", [])
            if item["name"] not in selected_names
        ]
        audit["files"] = sorted(
            [*retained, *audit["files"]], key=lambda item: item["name"]
        )
    save_json(output / "audit.json", audit)
    return data, audit


def measurement_response(
    time_s,
    current_a,
    sample_times,
    tau_s=0.0,
    integration_s=0.0,
    offset_a=0.0,
    delay_s=0.0,
    initial_current_a=None,
):
    """电流通道一阶有效响应，再按已知积分窗口平均；不更改状态/温度。

    tau 包含电流通道与未辨识链路的有效带宽，不能当作热时间常数。
    B1530A 的积分窗口固定 10 ns，绝不将采样间隔 50 ns 当作积分时间。
    首个滤波状态取输入首值；缺失预触发段的影响需在结果说明中保留。
    delay_s 是电流观测相对输入电压的纯时移，与 tau_s 的带宽衰减分开。
    可显式用首个实测电流初始化观测状态；这不是电流零点扣除，也不改变
    器件初态或强制电压。延迟后落在记录前的观测保持该初值，不补造前沿。
    """
    t, u = np.asarray(time_s), np.asarray(current_a)
    if any(not math.isfinite(x) or x < 0 for x in (tau_s, integration_s, delay_s)):
        raise ValueError("响应、积分和延迟时间须为非负有限数")
    if initial_current_a is not None and not math.isfinite(initial_current_a):
        raise ValueError("观测初始电流必须有限")
    filtered = u.copy()
    if tau_s:
        if initial_current_a is not None:
            filtered[0] = initial_current_a
        for k, h in enumerate(np.diff(t), 1):
            one_minus = -math.expm1(-h / tau_s)
            # 线性变化输入的精确一阶滤波，避免不均匀网格上的显式不稳定。
            filtered[k] = (
                (1 - one_minus) * filtered[k - 1]
                + one_minus * u[k - 1]
                + (u[k] - u[k - 1]) * (1 - tau_s / h * one_minus)
            )
    observation_times = np.asarray(sample_times) - delay_s
    if integration_s:
        integral = cumulative_trapezoid(filtered, t, initial=0)
        low = np.maximum(t[0], observation_times - integration_s / 2)
        high = np.minimum(t[-1], observation_times + integration_s / 2)
        if np.any(high <= low):
            raise ValueError("积分窗口在仿真区间之外")
        values = (np.interp(high, t, integral) - np.interp(low, t, integral)) / (
            high - low
        )
    else:
        values = np.interp(
            observation_times,
            t,
            filtered,
            left=filtered[0] if initial_current_a is None else initial_current_a,
        )
    return values + offset_a


def _capacitive_observation(
    time_s,
    voltage_v,
    sample_times,
    capacitance_f,
    tau_s=0.0,
    integration_s=0.0,
    delay_s=0.0,
):
    """分段线性 V 的 C*dV/dt 观测：按区间电荷解析推进，不插值节点梯度。

    每段 Icap=C*ΔV/Δt 为常数，一阶响应 z 的精确解为
    z(t+h)=z(t)*exp(-h/tau)+Icap*(1-exp(-h/tau))。
    这只是既有电容电流与通道响应的数值闭合；电容不参与细丝焦耳热。
    观测电容状态初值为零，首个实测电流初值交给导电观测支路承担。
    """
    t, voltage = np.asarray(time_s), np.asarray(voltage_v)
    if len(t) < 2 or len(t) != len(voltage) or np.any(np.diff(t) <= 0):
        raise ValueError("电容观测需要等长、严格递增的电压时序")
    if any(
        not math.isfinite(x) or x < 0
        for x in (capacitance_f, tau_s, integration_s, delay_s)
    ):
        raise ValueError("电容及观测时间参数须为非负有限数")
    query = np.asarray(sample_times) - delay_s
    if capacitance_f == 0:
        return np.zeros_like(query)
    intervals = np.diff(t)
    rates = capacitance_f * np.diff(voltage) / intervals
    states = np.zeros(len(t))
    integral = capacitance_f * (voltage - voltage[0])
    if tau_s:
        integral = np.zeros(len(t))
        for k, h in enumerate(intervals, 1):
            one_minus = -math.expm1(-h / tau_s)
            states[k] = (1 - one_minus) * states[k - 1] + one_minus * rates[k - 1]
            integral[k] = (
                integral[k - 1]
                + rates[k - 1] * h
                + (states[k - 1] - rates[k - 1]) * tau_s * one_minus
            )

    def evaluate(points, integrated=False):
        index = np.clip(np.searchsorted(t, points, side="right") - 1, 0, len(t) - 2)
        elapsed = np.maximum(points - t[index], 0.0)
        if tau_s:
            one_minus = -np.expm1(-elapsed / tau_s)
            if integrated:
                result = (
                    integral[index]
                    + rates[index] * elapsed
                    + (states[index] - rates[index]) * tau_s * one_minus
                )
            else:
                result = (1 - one_minus) * states[index] + one_minus * rates[index]
            after = points > t[-1]
            tail = np.maximum(points - t[-1], 0.0)
            tail_value = (
                integral[-1] + states[-1] * tau_s * (-np.expm1(-tail / tau_s))
                if integrated
                else states[-1] * np.exp(-tail / tau_s)
            )
            result = np.where(after, tail_value, result)
        elif integrated:
            result = capacitance_f * (np.interp(points, t, voltage) - voltage[0])
        else:
            result = rates[index]
            result = np.where(points > t[-1], 0.0, result)
        return np.where(points < t[0], 0.0, result)

    if integration_s:
        low = np.maximum(t[0], query - integration_s / 2)
        high = np.minimum(t[-1], query + integration_s / 2)
        if np.any(high <= low):
            raise ValueError("电容观测积分窗口在仿真区间之外")
        return (evaluate(high, True) - evaluate(low, True)) / (high - low)
    return evaluate(query)


def decode(values, reference):
    """优化坐标采用易读尺度，对数只用于跨数量级的正值量。"""
    sigma, density, ea, er, gamma, rth, tau, cap, response, eac = values
    # 用室温有效电导率与温度系数分开参数化，避免 sigma0 与 EAC 在室温
    # 指数项中直接互相补偿。它们仍不是独立温度测量得到的材料常数。
    p = replace(
        reference,
        conductivity_s_m=5e5 * math.exp(sigma - eac / (KB_EV * reference.ambient_k)),
        conduction_ev=eac,
        hopping_density_a_m2=1e13 * math.exp(density),
        generation_ev=ea,
        recombination_ev=er,
        release_gamma=gamma,
        thermal_k_w=1e5 * math.exp(rth),
    )
    options = TransientOptions(
        thermal_time_s=1e-9 * math.exp(tau), pad_capacitance_f=cap * 1e-12
    )
    return p, options, response * 1e-9


def _decode_candidate(values, reference):
    values = np.asarray(values, dtype=float)
    parameters, options, response = decode(values[:10], reference)
    if len(values) >= 15:
        parameters = replace(
            parameters,
            hopping_ev=values[10],
            release_ev=values[11],
            hopping_alpha_m=values[12] * 1e-9,
            generation_alpha_m=values[13] * 1e-9,
        )
    if len(values) >= 16:
        parameters = replace(parameters, hopping_voltage_v=values[15])
    return parameters, options, response


INITIAL = np.array(
    [0.0, 0.0, 1.65, 0.64, 0.31, math.log(3.0), math.log(5.0), 2.0, 15.0, 0.02]
)
# HiRes 热/通道及传导的旧已接受尺度仅作新数据重拟合的数值起点；
# 旧预测和旧目标函数不复用，所有阶段均对当前源文件重新计算。
HIRES_SEED = np.array(
    [
        -0.5585319139794074,
        1.208111824487309,
        1.935814832245289,
        0.626903328663627,
        0.6267876025524851,
        0.4584057488451928,
        -1.601990360503187,
        0.00010422580521014478,
        14.129407460894825,
        0.008234175734808119,
    ]
)
LOWER = np.array(
    [
        math.log(0.2),
        math.log(0.1),
        1.1,
        0.45,
        0.10,
        math.log(0.01),
        math.log(0.2),
        0.0,
        0.0,
        0.0,
    ]
)
UPPER = np.array(
    [
        math.log(5.0),
        math.log(5.0),
        2.2,
        0.85,
        0.8,
        math.log(10.0),
        math.log(100.0),
        100.0,
        100.0,
        0.15,
    ]
)
NAMES = [
    "log_sigma300_over_5e5",
    "log_J0_over_1e13",
    "generation_ev",
    "recombination_ev",
    "release_gamma",
    "log_Rth_over_1e5",
    "log_thermal_tau_ns",
    "pad_capacitance_pf",
    "current_response_ns",
    "conduction_ev",
]

READ_WINDOWS = ((1.10e-6, 1.50e-6), (3.10e-6, 3.50e-6))
READ_PULSE_WINDOWS = ((1.00e-6, 1.60e-6), (3.00e-6, 3.60e-6))
RESET_WINDOW = (2.00e-6, 2.50e-6)
SET_WINDOW = (0.0, 0.50e-6)
INITIAL_CONDUCTANCE = 85e-6
INITIAL_CONDUCTANCE_V = 0.2
INITIAL_CONDUCTANCE_RELATIVE_SCALE = 0.2


def predict(
    data,
    values,
    reference,
    initial_state,
    *,
    grid_s=None,
    rtol=0.003,
    include_sample_nodes=True,
    include_program_nodes=False,
    include_priority_nodes=False,
    current_delay_s=0.0,
    initial_observed_a=None,
    charge_conserving_capacitance=False,
):
    values = np.asarray(values, dtype=float)
    p, options, response = _decode_candidate(values, reference)
    if len(values) >= 16:
        initial_state = replace(initial_state, x_m=values[14] * 1e-9)
    step = grid_s or (10e-9 if data.instrument == "b1530" else 2e-9)
    # B1530A 首点在 25 ns，缺少前沿零点。显式补一个 t=0,V=0 的边界假设，
    # 不把它写回原始测量。HiRes 已处于前沿，避免人为在 0.32 ns 内从零跳变。
    start = 0.0 if data.instrument == "b1530" else data.time_s[0]
    end = data.time_s[-1] + data.integration_s / 2
    # 保留全部输入电压节点，尤其不能用均匀网格略过欠采样脉冲唯一的峰点。
    # 细化网格只增加求解点，不改变实测节点间的线性驱动定义。
    grid_parts = [np.linspace(start, end, int(np.ceil((end - start) / step)) + 1)]
    if include_sample_nodes:
        grid_parts.append(data.time_s)
    if include_program_nodes and data.instrument == "hires":
        program = _phase_mask(data, SET_WINDOW) | _phase_mask(data, RESET_WINDOW)
        grid_parts.append(data.time_s[program])
    if include_priority_nodes and data.instrument == "hires":
        priority = _priority_voltage_node_mask(data)
        grid_parts.append(data.time_s[priority])
    grid = np.unique(np.concatenate(grid_parts))
    tx, vx = data.time_s, data.voltage_v
    if data.instrument == "b1530":
        tx, vx = np.r_[0.0, tx], np.r_[0.0, vx]
    voltage = np.interp(grid, tx, vx)
    result = simulate_transient(
        grid,
        voltage,
        initial_state,
        p,
        options,
        rtol=rtol,
        atol_m=1e-13,
        max_step_s=10e-9 if grid_s is None else grid_s,
    )
    # 没有独立零点标定时不擅自减去电流。特别是 HiRes 的 V_DUT 已减 Vread，
    # 按 |V_DUT| 很小筛选会反过来选择某些 Iread，造成内生的偏置估计。
    offset = 0.0
    observed = measurement_response(
        grid,
        [point.current_a for point in result.points]
        if charge_conserving_capacitance
        else result.total_a,
        data.time_s,
        response,
        data.integration_s,
        offset,
        delay_s=current_delay_s,
        initial_current_a=initial_observed_a,
    )
    if charge_conserving_capacitance:
        observed += _capacitive_observation(
            grid,
            voltage,
            data.time_s,
            options.pad_capacitance_f,
            response,
            data.integration_s,
            current_delay_s,
        )
    return observed, result, offset


def residual_for(data, predicted):
    phase = np.mod(data.time_s, 4e-6)
    read = ((phase > 1.1e-6) & (phase < 1.6e-6)) | ((phase > 3.1e-6) & (phase < 3.6e-6))
    program = (phase < (data.width_ns + 150) * 1e-9) | (
        (phase > 2e-6) & (phase < 2e-6 + (data.width_ns + 150) * 1e-9)
    )
    # 零偏压噪声不主导器件参数；全部测点仍导出并计算无权 RMSE。
    weight = np.where(read | program, 1.0, 0.1)
    return np.sqrt(weight / weight.sum()) * (
        np.arcsinh(predicted / 1e-4) - np.arcsinh(data.current_a / 1e-4)
    )


def _phase_mask(data, window):
    phase = np.mod(data.time_s, 4e-6)
    return (phase >= window[0]) & (phase <= window[1])


def _optimizer_grid_s(data):
    """HiRes 短脉宽使用 0.5 ns 主网格，其余使用 1 ns 并加关键节点。"""
    if data.instrument != "hires":
        return None
    return 0.5e-9 if data.width_ns in (5, 10, 20) else 1e-9


def _priority_voltage_node_mask(data):
    """保留编程脉冲边沿及 READ 平台稀疏节点，避免无效密集积分。"""
    program = _phase_mask(data, SET_WINDOW) | _phase_mask(data, RESET_WINDOW)
    active = program & (np.abs(data.voltage_v) >= 0.05)
    edge = active.copy()
    for shift in (1, 2):
        edge |= np.roll(active, shift) | np.roll(active, -shift)
    read = np.zeros(len(data.time_s), dtype=bool)
    for window in READ_PULSE_WINDOWS:
        read |= _phase_mask(data, window)
    sample_step = float(np.median(np.diff(data.time_s)))
    stride = max(1, round(10e-9 / sample_step))
    read_indices = np.flatnonzero(read)
    sparse_read = np.zeros_like(read)
    sparse_read[read_indices[::stride]] = True
    sparse_read[read_indices[-1:]] = True
    return edge | sparse_read


def _asinh_residual(measured, predicted):
    return np.arcsinh(predicted / 1e-4) - np.arcsinh(measured / 1e-4)


def _read_relative_errors(data, predicted):
    errors = []
    for window in READ_WINDOWS:
        mask = _phase_mask(data, window)
        measured_mean = float(np.mean(data.current_a[mask]))
        predicted_mean = float(np.mean(predicted[mask]))
        errors.append((predicted_mean - measured_mean) / abs(measured_mean))
    return np.asarray(errors)


def _initial_conductance_residual(
    values,
    reference,
    state,
    target_s=INITIAL_CONDUCTANCE,
    relative_scale=INITIAL_CONDUCTANCE_RELATIVE_SCALE,
):
    parameters, _, _ = _decode_candidate(values, reference)
    initial = replace(state, x_m=values[14] * 1e-9)
    current = operating_point(INITIAL_CONDUCTANCE_V, initial, parameters).current_a
    conductance = current / INITIAL_CONDUCTANCE_V
    if conductance <= 0:
        return 1e6
    return math.log(conductance / target_s) / math.log(1 + relative_scale)


def _observation_prior_residual(values, prior):
    """以正值对数坐标软约束通道延迟、响应时间和有效电容。"""
    if prior is None:
        return np.empty(0)
    delay_ns = math.exp(float(values[16]))
    targets = np.asarray(
        [prior["delay_ns"], prior["response_ns"], prior["capacitance_pf"]],
        dtype=float,
    )
    actual = np.asarray([delay_ns, values[8], values[7]], dtype=float)
    scale = float(prior["relative_scale"])
    if (
        not np.all(np.isfinite(targets))
        or np.any(targets <= 0)
        or not np.all(np.isfinite(actual))
        or np.any(actual <= 0)
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("观测软先验参数须为有限正数")
    return np.log(actual / targets) / math.log1p(scale)


def _initial_gap_seed(
    values,
    reference,
    state,
    target_s=INITIAL_CONDUCTANCE,
):
    """在固定 w、x0 下扫描初始间隙，为独立电导先验提供可行起点。"""
    parameters, _, _ = _decode_candidate(values, reference)
    candidates = np.linspace(0.0, state.x0_m * 1e9, 101)
    errors = []
    for gap_nm in candidates:
        initial = replace(state, x_m=gap_nm * 1e-9)
        conductance = (
            operating_point(INITIAL_CONDUCTANCE_V, initial, parameters).current_a
            / INITIAL_CONDUCTANCE_V
        )
        errors.append(
            abs(math.log(conductance / target_s)) if conductance > 0 else math.inf
        )
    return float(candidates[int(np.argmin(errors))])


def _read_waveform_residual(data, predicted):
    """完整 READ 波形含上升、下降沿，防止均值正确却出现额外 SET 尖峰。"""
    read_blocks = []
    for window in READ_PULSE_WINDOWS:
        mask = _phase_mask(data, window)
        read_blocks.append(
            _asinh_residual(data.current_a[mask], predicted[mask])
            / math.sqrt(mask.sum())
        )
    return np.concatenate(read_blocks)


def _read_pulse_envelope_errors(data, predicted):
    """READ 完整脉冲的幅值包络用于排除均值合格但额外 SET 的伪解。"""
    errors = []
    for window in READ_PULSE_WINDOWS:
        mask = _phase_mask(data, window)
        measured_peak = float(np.max(np.abs(data.current_a[mask])))
        if measured_peak <= 0:
            raise ValueError("READ 包络没有非零实测电流")
        errors.append(float(np.max(np.abs(predicted[mask]))) / measured_peak - 1)
    return np.asarray(errors)


def _hires_residual(
    data,
    predicted,
    values,
    reference,
    state,
    stage,
    initial_conductance_target_s=INITIAL_CONDUCTANCE,
    initial_conductance_relative_scale=INITIAL_CONDUCTANCE_RELATIVE_SCALE,
    observation_prior=None,
):
    read_errors = _read_relative_errors(data, predicted)
    # 5% 是平台均值验收门槛。容差外用 10 倍斜率强约束；容差内保留
    # 温和误差，避免优化器为把均值强行拉到零而制造局部 READ/SET 尖峰。
    normalized_level = read_errors / 0.05
    level_residual = normalized_level + 9 * np.sign(normalized_level) * np.maximum(
        np.abs(normalized_level) - 1, 0
    )
    initial_residual = np.array(
        [
            _initial_conductance_residual(
                values,
                reference,
                state,
                initial_conductance_target_s,
                initial_conductance_relative_scale,
            )
        ]
    )
    read_shape_residual = _read_waveform_residual(data, predicted)
    # 全部原始电流点保留低权重约束，尤其不能漏掉 READ 返回边沿与零偏段。
    global_residual = (
        0.1 * _asinh_residual(data.current_a, predicted) / math.sqrt(len(predicted))
    )
    common = np.r_[
        level_residual, initial_residual, 10 * read_shape_residual, global_residual
    ]
    common = np.r_[common, _observation_prior_residual(values, observation_prior)]
    if stage == "read":
        return common
    reset_mask = _phase_mask(data, RESET_WINDOW)
    reset_residual = _asinh_residual(
        data.current_a[reset_mask], predicted[reset_mask]
    ) / math.sqrt(reset_mask.sum())
    if stage == "reset":
        return np.r_[common, reset_residual]
    set_mask = _phase_mask(data, SET_WINDOW)
    set_residual = _asinh_residual(
        data.current_a[set_mask], predicted[set_mask]
    ) / math.sqrt(set_mask.sum())
    return np.r_[common, 2 * reset_residual, set_residual]


def _pulse_peak_metrics(data, predicted):
    """保留原始符号，分别评价正 SET 峰和负 RESET 峰，不取全波形绝对值。"""
    duration = min(0.5e-6, (data.width_ns + 80) * 1e-9)
    result = {}
    for name, start, sign in (("set", 0.0, 1), ("reset", 2e-6, -1)):
        rows = np.flatnonzero(_phase_mask(data, (start, start + duration)))
        if not len(rows):
            raise ValueError(f"{name} 峰值窗口没有测点")
        measured_row = int(rows[np.argmax(sign * data.current_a[rows])])
        predicted_row = int(rows[np.argmax(sign * predicted[rows])])
        magnitude = sign * data.current_a[measured_row]
        if magnitude <= 0:
            raise ValueError(f"{name} 窗口未包含正确极性的实测峰值")
        result[name] = {
            "measured_a": float(data.current_a[measured_row]),
            "predicted_a": float(predicted[predicted_row]),
            "relative_error": float(
                (sign * predicted[predicted_row] - magnitude) / magnitude
            ),
            "prediction_at_measured_peak_a": float(predicted[measured_row]),
            "measured_time_s": float(data.time_s[measured_row]),
            "predicted_time_s": float(data.time_s[predicted_row]),
            "timing_error_s": float(
                data.time_s[predicted_row] - data.time_s[measured_row]
            ),
            "window_s": [start, start + duration],
        }
    return result


def _narrow_peak_residual(
    data,
    predicted,
    values,
    reference,
    state,
    stage,
    initial_conductance_target_s=INITIAL_CONDUCTANCE,
    initial_conductance_relative_scale=INITIAL_CONDUCTANCE_RELATIVE_SCALE,
    observation_prior=None,
):
    common = _hires_residual(
        data,
        predicted,
        values,
        reference,
        state,
        "read",
        initial_conductance_target_s,
        initial_conductance_relative_scale,
        observation_prior,
    )
    read_errors = _read_relative_errors(data, predicted)
    # READ 5% 是先决门槛；峰值优化在 4.5% 处开始加罚，保留数值余量。
    read_guard = (
        50 * np.sign(read_errors) * np.maximum(np.abs(read_errors) - 0.045, 0) / 0.005
    )
    peaks = _pulse_peak_metrics(data, predicted)
    peak_residual = []
    pulse_shapes = []
    for name, sign in (("set", 1), ("reset", -1)):
        peak = peaks[name]
        magnitude = abs(peak["measured_a"])
        peak_residual.extend(
            (
                peak["relative_error"] / 0.05,
                (sign * peak["prediction_at_measured_peak_a"] / magnitude - 1) / 0.05,
            )
        )
        mask = _phase_mask(data, peak["window_s"])
        pulse_shapes.append(
            (predicted[mask] - data.current_a[mask]) / magnitude / math.sqrt(mask.sum())
        )
    if stage == "read":
        # 很弱的双峰约束排除 READ 正确却写脉冲数倍过冲的初始化分支；
        # 权重仅为后续峰值阶段的 0.5%，READ 仍是先决门槛。
        return np.r_[common, read_guard, 0.01 * np.asarray(peak_residual)]
    # 同时约束峰顶幅值与实测峰时处幅值，避免错峰或单点假峰满足标量目标。
    return np.r_[
        common, read_guard, 2 * np.asarray(peak_residual), np.concatenate(pulse_shapes)
    ]


def fit_group(
    data,
    reference,
    output,
    *,
    max_nfev=20,
    start_coordinates=None,
    optimizer_grid_s=None,
    calibration_mode="waveform",
    current_delay_s=0.0,
    initial_observed_a=None,
    response_bounds_ns=None,
    read_max_nfev=None,
    initial_conductance_target_s=INITIAL_CONDUCTANCE,
    initial_conductance_relative_scale=INITIAL_CONDUCTANCE_RELATIVE_SCALE,
    observation_prior=None,
    stage_max_nfev=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    group = data[0].instrument
    if read_max_nfev is not None and read_max_nfev < 1:
        raise ValueError("READ 阶段评估次数须为正整数")
    if group == "hires" and len(data) != 1:
        raise ValueError("HiRes 校准需按脉宽逐组独立执行")
    if calibration_mode not in ("waveform", "narrow_peaks"):
        raise ValueError("未知脉冲校准模式")
    if calibration_mode == "narrow_peaks" and group != "hires":
        raise ValueError("窄脉冲峰值优先模式仅用于 HiRes")
    if not math.isfinite(current_delay_s) or current_delay_s < 0:
        raise ValueError("电流观测延迟须为非负有限数")
    if calibration_mode == "narrow_peaks" and initial_observed_a is None:
        initial_observed_a = float(data[0].current_a[0])
    observation_options = {
        "current_delay_s": current_delay_s,
        "initial_observed_a": initial_observed_a,
        "charge_conserving_capacitance": calibration_mode == "narrow_peaks",
    }
    if optimizer_grid_s is not None and (
        group != "hires" or not math.isfinite(optimizer_grid_s) or optimizer_grid_s <= 0
    ):
        raise ValueError("optimizer_grid_s 必须是 HiRes 的有限正积分间隔（s）")
    fit_grid_s = (
        optimizer_grid_s if optimizer_grid_s is not None else _optimizer_grid_s(data[0])
    )
    # 几何不可由单端脉冲唯一提取；采用与 DC 一致的尺度，仅作为条件校准。
    state = State(0.15e-9, 2.5e-9, 1e-9)
    initial = HIRES_SEED.copy() if group == "hires" else INITIAL.copy()
    if group == "hires":
        initial[3], initial[4] = 0.45, 0.8
        initial = np.r_[
            initial,
            reference.hopping_ev,
            reference.release_ev,
            reference.hopping_alpha_m * 1e9,
            0.75,
            state.x_m * 1e9,
            reference.hopping_voltage_v,
        ]
        initial[13] = 0.75
        initial[14] = _initial_gap_seed(initial, reference, state)
    lower = np.r_[LOWER, 0.5, 0.6, 0.2, 0.2, 0.0, 0.1] if group == "hires" else LOWER
    # x0 仍固定为 1 nm；初始间隙拟合边界必须满足 x <= x0。
    upper = np.r_[UPPER, 1.5, 1.8, 1.5, 1.5, 1.0, 1.0] if group == "hires" else UPPER
    if observation_prior is not None:
        if group != "hires":
            raise ValueError("观测软先验仅支持 HiRes 校准")
        for key in ("delay_ns", "response_ns", "capacitance_pf", "relative_scale"):
            value = float(observation_prior[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"观测软先验 {key} 必须为有限正数")
        lower = np.r_[lower, -10.0]
        upper = np.r_[upper, 6.0]
        initial = np.r_[initial, math.log(observation_prior["delay_ns"])]
    if start_coordinates is not None:
        initial = np.asarray(start_coordinates, dtype=float)
        if group == "hires" and initial.shape == LOWER.shape:
            initial = np.r_[
                initial,
                reference.hopping_ev,
                reference.release_ev,
                reference.hopping_alpha_m * 1e9,
                0.75,
                state.x_m * 1e9,
                reference.hopping_voltage_v,
            ]
        elif group == "hires" and initial.shape == (15,):
            initial = np.r_[initial, reference.hopping_voltage_v]
        if (
            group == "hires"
            and observation_prior is not None
            and initial.shape == (16,)
        ):
            initial = np.r_[initial, math.log(observation_prior["delay_ns"])]
        if (
            initial.shape != lower.shape
            or np.any(initial < lower)
            or np.any(initial > upper)
        ):
            raise ValueError("续拟合坐标不在边界内")
    if observation_prior is not None and start_coordinates is None:
        initial[7] = observation_prior["capacitance_pf"]
        initial[8] = observation_prior["response_ns"]
    response_seed_before_bounds = float(initial[8])
    if response_bounds_ns is not None:
        if (
            group != "hires"
            or len(response_bounds_ns) != 2
            or any(not math.isfinite(x) for x in response_bounds_ns)
            or not 0 <= response_bounds_ns[0] < response_bounds_ns[1] <= UPPER[8]
        ):
            raise ValueError("通道响应约束须为 HiRes 的有效 [最小, 最大] ns 范围")
        lower[8], upper[8] = response_bounds_ns
        initial[8] = np.clip(initial[8], lower[8], upper[8])
    # 对命令行实际提供的数据逐条拟合；不根据脉宽名称隐式排除训练条件。
    training = list(data)
    # HiRes 全部 25,000 个电流测点都参与残差；为降低有限差分成本，优化时
    # 在配置的主网格及关键驱动节点上积分；最终预测仍纳入全部原始节点。
    fit_data = training
    values = initial.copy()
    history = []
    calls = 0
    selected_objective = "read"

    def observation_options_for(candidate):
        options = observation_options.copy()
        if observation_prior is not None:
            options["current_delay_s"] = math.exp(float(candidate[16])) * 1e-9
        return options

    def full_residual(candidate, objective=None):
        nonlocal calls
        calls += 1
        result = []
        for datum in fit_data:
            if group == "hires":
                pred, _, _ = predict(
                    datum,
                    candidate,
                    reference,
                    state,
                    grid_s=fit_grid_s,
                    include_sample_nodes=False,
                    include_priority_nodes=True,
                    **observation_options_for(candidate),
                )
            else:
                pred, _, _ = predict(datum, candidate, reference, state)
            if group == "hires":
                objective_function = (
                    _narrow_peak_residual
                    if calibration_mode == "narrow_peaks"
                    else _hires_residual
                )
                residual_options = {
                    "initial_conductance_target_s": initial_conductance_target_s,
                    "initial_conductance_relative_scale": initial_conductance_relative_scale,
                    "observation_prior": observation_prior,
                }
                result.extend(
                    objective_function(
                        datum,
                        pred,
                        candidate,
                        reference,
                        state,
                        objective,
                        **residual_options,
                    )
                )
            else:
                result.extend(residual_for(datum, pred))
        return np.asarray(result)

    if group == "hires":
        stages = [
            (
                "read_priority",
                "read",
                [0, 1, 2, 3, 4, 7, 8, 10, 11, 12, 14, 15, 16]
                if observation_prior is not None
                else [0, 1, 2, 3, 4, 10, 11, 12, 14, 15],
            ),
            (
                "reset_priority",
                "reset",
                [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]
                if observation_prior is not None
                else [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
            ),
            (
                "set_priority",
                "set",
                [2, 5, 6, 13, 16] if observation_prior is not None else [2, 5, 6, 13],
            ),
        ]
        if calibration_mode == "narrow_peaks":
            stages = [
                (
                    "read_priority",
                    "read",
                    list(range(len(initial))),
                ),
                ("peak_priority", "peaks", list(range(len(initial)))),
            ]
        peak_score_at_read = None
        read_candidate = values.copy()
        reset_candidate = None
        reset_score = None
        reset_score_at_read = None
        set_score_at_reset = None
        read_shape_at_read = None
        read_shape_at_reset = None
        reset_accepted = False
    else:
        stages = [
            ("transport_and_channel", None, [0, 1, 7, 8, 9]),
            ("switching_and_heat", None, [2, 3, 4, 5, 6]),
            ("joint", None, list(range(10))),
        ]
        if start_coordinates is not None:
            stages = [("joint_refinement", None, list(range(10)))]
    if group != "hires":
        full_residual(initial)
    for label, objective, indices in stages:
        if group == "hires" and label == "set_priority" and not reset_accepted:
            break
        anchor = values.copy()

        def residual(z, anchor=anchor, indices=indices, objective=objective):
            candidate = anchor.copy()
            candidate[indices] = z
            return full_residual(candidate, objective)

        def jacobian(z, indices=indices, residual=residual):
            # 使用按边界跨度定义的绝对差分，避免对数坐标=0 时退回极小
            # 相对步长，导致状态积分误差压过数值导数。
            base = residual(z)
            jac = np.empty((len(base), len(z)))
            for j, index in enumerate(indices):
                h = 0.002 * (upper[index] - lower[index])
                step = h if z[j] + h <= upper[index] else -h
                shifted = z.copy()
                shifted[j] += step
                jac[:, j] = (residual(shifted) - base) / step
            return jac

        start = time.perf_counter()
        stage_limit = (stage_max_nfev or {}).get(label, max_nfev)
        try:
            result = least_squares(
                residual,
                values[indices],
                bounds=(lower[indices], upper[indices]),
                x_scale="jac",
                jac=jacobian,
                max_nfev=read_max_nfev
                if label == "read_priority" and read_max_nfev is not None
                else stage_limit,
                ftol=1e-5,
                xtol=1e-5,
                gtol=1e-6,
            )
        except (ModelError, ValueError, OverflowError) as exc:
            # 不将物理求解失败当作巨大常量残差；保留已成功阶段以便诊断。
            save_json(
                output / "failure.json",
                {
                    "stage": label,
                    "error": str(exc),
                    "last_valid_coordinates": values.tolist(),
                },
            )
            raise
        values[indices] = result.x
        info = {
            "stage": label,
            "success": bool(result.success),
            "message": str(result.message),
            "cost": float(result.cost),
            "nfev": int(result.nfev),
            "seconds": time.perf_counter() - start,
            "optimality": float(result.optimality),
            "coordinates": values.tolist(),
        }
        history.append(info)
        save_json(output / "progress.json", history)
        print(
            group,
            label,
            "cost",
            info["cost"],
            "seconds",
            round(info["seconds"], 1),
            flush=True,
        )
        if group == "hires":
            datum = data[0]
            prediction = predict(
                datum,
                values,
                reference,
                state,
                grid_s=fit_grid_s,
                include_sample_nodes=False,
                include_priority_nodes=True,
                **observation_options_for(values),
            )[0]
            relative_error = _read_relative_errors(datum, prediction)
            read_ok = np.max(np.abs(relative_error)) <= 0.05
            history[-1]["read_relative_errors"] = relative_error.tolist()
            history[-1]["read_within_5_percent"] = bool(read_ok)
            read_shape_score = float(
                np.linalg.norm(_read_waveform_residual(datum, prediction))
            )
            history[-1]["read_waveform_score"] = read_shape_score
            if calibration_mode == "narrow_peaks":
                peaks = _pulse_peak_metrics(datum, prediction)
                peak_errors = np.array(
                    [
                        value
                        for name, sign in (("set", 1), ("reset", -1))
                        for value in (
                            peaks[name]["relative_error"],
                            sign
                            * peaks[name]["prediction_at_measured_peak_a"]
                            / abs(peaks[name]["measured_a"])
                            - 1,
                        )
                    ]
                )
                score = float(np.linalg.norm(peak_errors))
                set_score = float(np.linalg.norm(peak_errors[:2]))
                reset_score = float(np.linalg.norm(peak_errors[2:]))
                history[-1]["pulse_peaks"] = peaks
                history[-1]["peak_score"] = score
                history[-1]["set_peak_score"] = set_score
                history[-1]["reset_peak_score"] = reset_score
                if label == "read_priority":
                    read_candidate = values.copy()
                    read_shape_at_read = read_shape_score
                    peak_score_at_read = score
                else:
                    accepted = (
                        read_ok
                        and peak_score_at_read is not None
                        and score <= peak_score_at_read + 1e-8
                        and np.max(_read_pulse_envelope_errors(datum, prediction))
                        <= 0.10
                    )
                    # 新的窄脉冲优先级：READ 均值正常且无额外大尖峰后，
                    # 双峰优先于 READ 边沿的细小 RMS 改善；不沿用宽脉冲的
                    # READ 形状仅允许恶化 10% 的相对门控。
                    history[-1]["read_pulse_envelope_errors"] = (
                        _read_pulse_envelope_errors(datum, prediction).tolist()
                    )
                    history[-1]["accepted"] = bool(accepted)
                    if accepted:
                        selected_objective = "peaks"
                    else:
                        values = read_candidate.copy()
                save_json(output / "progress.json", history)
                continue
            if label == "read_priority":
                read_candidate = values.copy()
                read_shape_at_read = read_shape_score
                reset_mask = _phase_mask(datum, RESET_WINDOW)
                reset_score_at_read = float(
                    np.sqrt(
                        np.mean(
                            _asinh_residual(
                                datum.current_a[reset_mask], prediction[reset_mask]
                            )
                            ** 2
                        )
                    )
                )
                history[-1]["reset_score_at_read"] = reset_score_at_read
            elif label == "reset_priority":
                reset_mask = _phase_mask(datum, RESET_WINDOW)
                candidate_reset_score = float(
                    np.sqrt(
                        np.mean(
                            _asinh_residual(
                                datum.current_a[reset_mask], prediction[reset_mask]
                            )
                            ** 2
                        )
                    )
                )
                reset_ok = (
                    reset_score_at_read is not None
                    and candidate_reset_score <= reset_score_at_read * 1.10 + 1e-8
                )
                history[-1]["reset_score_at_read"] = reset_score_at_read
                history[-1]["reset_score"] = candidate_reset_score
                read_shape_ok = (
                    read_shape_at_read is not None
                    and read_shape_score <= read_shape_at_read * 1.10 + 1e-8
                )
                if read_ok and reset_ok and read_shape_ok:
                    reset_candidate = values.copy()
                    reset_score = candidate_reset_score
                    read_shape_at_reset = read_shape_score
                    set_mask = _phase_mask(datum, SET_WINDOW)
                    set_score_at_reset = float(
                        np.sqrt(
                            np.mean(
                                _asinh_residual(
                                    datum.current_a[set_mask], prediction[set_mask]
                                )
                                ** 2
                            )
                        )
                    )
                    reset_accepted = True
                    selected_objective = "reset"
                else:
                    values = read_candidate.copy()
                    history[-1]["accepted"] = False
            else:
                reset_mask = _phase_mask(datum, RESET_WINDOW)
                candidate_reset_score = float(
                    np.sqrt(
                        np.mean(
                            _asinh_residual(
                                datum.current_a[reset_mask], prediction[reset_mask]
                            )
                            ** 2
                        )
                    )
                )
                accepted = (
                    read_ok
                    and reset_score is not None
                    and candidate_reset_score <= reset_score * 1.01 + 1e-8
                    and set_score_at_reset is not None
                    and read_shape_at_reset is not None
                    and read_shape_score <= read_shape_at_reset * 1.01 + 1e-8
                )
                set_mask = _phase_mask(datum, SET_WINDOW)
                candidate_set_score = float(
                    np.sqrt(
                        np.mean(
                            _asinh_residual(
                                datum.current_a[set_mask], prediction[set_mask]
                            )
                            ** 2
                        )
                    )
                )
                accepted = accepted and (
                    candidate_set_score <= set_score_at_reset * 1.01 + 1e-8
                )
                history[-1]["reset_score"] = candidate_reset_score
                history[-1]["set_score"] = candidate_set_score
                history[-1]["set_score_at_reset"] = set_score_at_reset
                history[-1]["accepted"] = bool(accepted)
                if accepted:
                    selected_objective = "set"
                elif reset_candidate is not None:
                    values = reset_candidate.copy()
            save_json(output / "progress.json", history)
    p, options, response = _decode_candidate(values, reference)
    final_delay_s = (
        math.exp(float(values[16])) * 1e-9
        if observation_prior is not None
        else current_delay_s
    )
    save_json(output / "parameters.json", asdict(p))
    save_json(
        output / "transient_options.json",
        {
            **asdict(options),
            "current_response_s": response,
            "current_delay_s": final_delay_s,
            "observation_initial_current_a": initial_observed_a,
            "charge_conserving_capacitance": calibration_mode == "narrow_peaks",
        },
    )
    save_json(
        output / "fit_config.json",
        {
            "instrument": group,
            "source_names": [d.name for d in data],
            "training_names": [d.name for d in training],
            "initial_state": asdict(replace(state, x_m=values[14] * 1e-9))
            if group == "hires"
            else asdict(state),
            "reference_parameters": asdict(reference),
            "initial_coordinates": initial.tolist(),
            "final_coordinates": values.tolist(),
            "coordinate_names": (
                [
                    *NAMES,
                    "hopping_ev",
                    "release_ev",
                    "hopping_alpha_nm",
                    "generation_alpha_nm",
                    "initial_gap_nm",
                    "hopping_voltage_v",
                    *(
                        ["log_current_delay_ns"]
                        if observation_prior is not None
                        else []
                    ),
                ]
                if group == "hires"
                else NAMES
            ),
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "max_nfev_per_stage": max_nfev,
            "stage_max_nfev": stage_max_nfev or {},
            "read_max_nfev": read_max_nfev,
            "calibration_mode": calibration_mode,
            "charge_conserving_capacitance": calibration_mode == "narrow_peaks",
            "current_delay_s": final_delay_s,
            "observation_initial_current_a": initial_observed_a,
            "observation_initialization": "fixed first measured current sample"
            if calibration_mode == "narrow_peaks"
            else "input first value"
            if initial_observed_a is None
            else "explicit current state",
            "peak_relative_tolerance": 0.05
            if calibration_mode == "narrow_peaks"
            else None,
            "response_bounds_ns": list(response_bounds_ns)
            if response_bounds_ns is not None
            else None,
            "response_seed_before_bounds_ns": response_seed_before_bounds,
            "observation_prior": observation_prior,
            "initial_conductance_target_s": initial_conductance_target_s,
            "initial_conductance_relative_scale": initial_conductance_relative_scale,
            "narrow_peak_objective": {
                "read_stage_peak_weight": 0.01,
                "peak_stage_peak_weight": 2.0,
                "peak_relative_scale": 0.05,
                "read_pulse_envelope_relative_tolerance": 0.10,
                "read_guard_start_relative_error": 0.045,
            }
            if calibration_mode == "narrow_peaks"
            else None,
            "scope_current_conversion": "vsense_v / 50 ohm",
            "scope_voltage_conversion": "vforce_v - vsense_v",
            "compliance": None,
            "rtol": 0.003,
            "atol_m": 1e-13,
            "optimizer_grid_s": fit_grid_s,
            "optimizer_includes_program_voltage_nodes": False,
            "optimizer_includes_priority_voltage_nodes": group == "hires",
            "optimizer_uses_all_current_samples": group == "hires",
            "final_prediction_includes_all_input_samples": group == "hires",
            "source_sha256": data[0].source_sha256,
            "initial_conductance_constraint": {
                "voltage_v": INITIAL_CONDUCTANCE_V,
                "conductance_s": initial_conductance_target_s,
                "relative_scale": initial_conductance_relative_scale,
            }
            if group == "hires"
            else None,
            "priority_windows_s": {
                "read_lrs": list(READ_WINDOWS[0]),
                "reset": list(RESET_WINDOW),
                "read_hrs": list(READ_WINDOWS[1]),
                "set": list(SET_WINDOW),
            }
            if group == "hires"
            else None,
            "read_mean_relative_tolerance": 0.05 if group == "hires" else None,
            "read_waveform_windows_s": [list(w) for w in READ_PULSE_WINDOWS]
            if group == "hires"
            else None,
            "hires_residual_weights": {
                "read_mean_inside_tolerance": 1.0,
                "read_mean_outside_tolerance_slope": 10.0,
                "read_waveform": 10.0,
                "global_waveform": 0.1,
            }
            if group == "hires"
            else None,
            "selected_objective": selected_objective if group == "hires" else None,
        },
    )
    statistics = []
    for datum in data:
        fitted, sim, offset = predict(
            datum,
            values,
            reference,
            state,
            grid_s=fit_grid_s,
            **observation_options_for(values),
        )
        phase = np.mod(datum.time_s, 4e-6)
        masks = {
            "all": np.ones(len(fitted), dtype=bool),
            "read_lrs": _phase_mask(datum, READ_WINDOWS[0])
            if group == "hires"
            else (phase >= 1.15e-6) & (phase <= 1.45e-6),
            "read_hrs": _phase_mask(datum, READ_WINDOWS[1])
            if group == "hires"
            else (phase >= 3.15e-6) & (phase <= 3.45e-6),
            "reset": _phase_mask(datum, RESET_WINDOW)
            if group == "hires"
            else np.zeros(len(fitted), dtype=bool),
            "set": _phase_mask(datum, SET_WINDOW)
            if group == "hires"
            else np.zeros(len(fitted), dtype=bool),
        }
        metric = {
            "name": datum.name,
            "training": datum.name in [d.name for d in training],
            "rows": len(fitted),
            "offset_a": offset,
            "rmse_a": {},
            "max_temperature_k": max(op.temperature_k for op in sim.points),
        }
        for name, mask in masks.items():
            metric["rmse_a"][name] = (
                float(np.sqrt(np.mean((fitted[mask] - datum.current_a[mask]) ** 2)))
                if np.any(mask)
                else None
            )
        if group == "hires":
            metric["pulse_peaks"] = _pulse_peak_metrics(datum, fitted)
            metric["read_pulse_rmse_a"] = {
                name: float(
                    np.sqrt(np.mean((fitted[mask] - datum.current_a[mask]) ** 2))
                )
                for name, mask in (
                    ("lrs", _phase_mask(datum, READ_PULSE_WINDOWS[0])),
                    ("hrs", _phase_mask(datum, READ_PULSE_WINDOWS[1])),
                )
            }
            metric["segment_mean_a"] = {
                name: {
                    "measured": float(np.mean(datum.current_a[mask])),
                    "predicted": float(np.mean(fitted[mask])),
                }
                for name, mask in masks.items()
                if name != "all"
            }
            metric["read_mean_relative_error"] = _read_relative_errors(
                datum, fitted
            ).tolist()
        statistics.append(metric)
        single_datum = len(data) == 1
        prediction_name = (
            "prediction.csv" if single_datum else datum.name + "_prediction.csv"
        )
        comparison_name = "comparison.png" if single_datum else datum.name + ".png"
        observed_columns = []
        if calibration_mode == "narrow_peaks":
            # 导出精确观测分量；原 capacitive_a 仍是器件节点梯度诊断值。
            observed_cap = _capacitive_observation(
                sim.time_s,
                [op.requested_v for op in sim.points],
                datum.time_s,
                options.pad_capacitance_f,
                response,
                datum.integration_s,
                final_delay_s,
            )
            observed_columns = [fitted - observed_cap, observed_cap]
        with (output / prediction_name).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "time_s",
                    "voltage_v",
                    "measured_a",
                    "predicted_a",
                    "conductive_a",
                    "capacitive_a",
                    "temperature_k",
                    "x_m",
                    "w_m",
                    *(
                        ["observed_conductive_a", "observed_capacitive_a"]
                        if observed_columns
                        else []
                    ),
                ]
            )
            grid = np.asarray(sim.time_s)
            writer.writerows(
                zip(
                    datum.time_s,
                    datum.voltage_v,
                    datum.current_a,
                    fitted,
                    np.interp(datum.time_s, grid, [op.current_a for op in sim.points]),
                    np.interp(datum.time_s, grid, sim.capacitive_a),
                    np.interp(
                        datum.time_s, grid, [op.temperature_k for op in sim.points]
                    ),
                    np.interp(datum.time_s, grid, [s.x_m for s in sim.states]),
                    np.interp(datum.time_s, grid, [s.w_m for s in sim.states]),
                    *observed_columns,
                    strict=True,
                )
            )
        plot_result(
            datum,
            fitted,
            output,
            filename=comparison_name,
            zoom_peaks=calibration_mode == "narrow_peaks",
        )
    final = full_residual(values, selected_objective if group == "hires" else None)
    notes = [
        "No current compliance",
        "No independent zero-current calibration supplied: no data-dependent current offset subtraction",
    ]
    if group == "hires":
        notes.append("I_DUT = vsense_v / 50 ohm; V_DUT = vforce_v - vsense_v")
    else:
        notes.append(
            "50 ns sampling cannot resolve the 20/50 ns drive shape or precise switching delay"
        )
    report = {
        "instrument": group,
        "optimizer": {
            k: history[-1][k] for k in ("success", "message", "nfev", "optimality")
        },
        "final_squared_residual": float(final @ final),
        "statistics": statistics,
        "parameters_identifiability": "effective conditional calibration; no uniqueness or cross-device transfer claim",
        "notes": notes,
    }
    if group == "hires":
        datum = data[0]
        fitted = predict(
            datum,
            values,
            reference,
            state,
            grid_s=fit_grid_s,
            **observation_options_for(values),
        )[0]
        relative_errors = _read_relative_errors(datum, fitted)
        initial_residual = _initial_conductance_residual(
            values,
            reference,
            state,
            initial_conductance_target_s,
            initial_conductance_relative_scale,
        )
        initial_conductance = initial_conductance_target_s * math.exp(
            initial_residual * math.log(1 + initial_conductance_relative_scale)
        )
        report["priority_result"] = {
            "selected_objective": selected_objective,
            "read_mean_relative_error": relative_errors.tolist(),
            "read_pass_5_percent": bool(np.max(np.abs(relative_errors)) <= 0.05),
            "initial_conductance_at_0p2v_s": initial_conductance,
            "initial_conductance_target_s": initial_conductance_target_s,
            "final_current_delay_s": final_delay_s,
            "observation_prior": observation_prior,
            "stages": history,
        }
        if calibration_mode == "narrow_peaks":
            peaks = _pulse_peak_metrics(datum, fitted)
            report["priority_result"]["peak_pass_5_percent"] = all(
                abs(p["relative_error"]) <= 0.05 for p in peaks.values()
            )
            report["priority_result"]["pulse_peaks"] = peaks
            report["priority_result"]["read_pulse_envelope_pass_10_percent"] = bool(
                np.max(_read_pulse_envelope_errors(datum, fitted)) <= 0.10
            )
    save_json(output / "report.json", report)
    (output / "failure.json").unlink(missing_ok=True)
    (output / "progress.json").unlink(missing_ok=True)
    return values


def _read_prediction(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return (
        np.asarray([float(row["time_s"]) for row in rows]),
        np.asarray([float(row["predicted_a"]) for row in rows]),
    )


def _clean_unaccepted_pulse_outputs(output_root, accepted_outputs):
    root = Path(output_root).resolve()
    accepted = {Path(path).as_posix() for path in accepted_outputs}
    prefixes = set()
    for path in accepted:
        parts = Path(path).parts
        prefixes.update(
            Path(*parts[:index]).as_posix() for index in range(1, len(parts))
        )

    def prune(directory, relative):
        for child in directory.iterdir():
            child_relative = (relative / child.name).as_posix()
            if child_relative in accepted:
                continue
            if child.is_dir() and child_relative in prefixes:
                prune(child, relative / child.name)
            elif child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    if root.exists():
        prune(root, Path("."))


def fit_batch(manifest_path, input_dir, output_root, report_path):
    """以统一模型和优化器逐曲线拟合，并采纳每条成功完成的结果。"""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("cleanup_unlisted_outputs") is not True:
        raise ValueError("批量清理 outputs/pulse 需在 manifest 显式启用")
    project_root = Path.cwd()
    output_root = Path(output_root)
    report_path = Path(report_path)
    results = []
    accepted_output_paths = []

    with TemporaryDirectory(prefix="rram-pulse-prepare-") as prepare_dir:
        source_names = [entry["source"] for entry in manifest["fits"]]
        prepared, _ = prepare(input_dir, prepare_dir, source_names)
        by_name = {datum.name + ".csv": datum for datum in prepared}

        for entry in manifest["fits"]:
            source = entry["source"]
            datum = by_name[source]
            output_rel = Path(entry["output"])
            if output_rel.is_absolute() or ".." in output_rel.parts:
                raise ValueError(
                    f"manifest 路径必须位于 outputs/pulse 内: {output_rel}"
                )
            candidate = output_root / output_rel
            target = Path("outputs/pulse") / output_rel
            accepted_output_paths.append(output_rel)
            seed_path = project_root / entry["seed_config"]
            seed = json.loads(seed_path.read_text(encoding="utf-8"))
            reference = Parameters(**seed["reference_parameters"])
            coordinates = np.asarray(seed["final_coordinates"], dtype=float).copy()
            prior = manifest["observation_prior"]
            if len(coordinates) not in (16, 17):
                raise ValueError(f"{seed_path}: HiRes 坐标维数错误")
            coordinates = coordinates[:16]
            coordinates[7] = prior["capacitance_pf"]
            coordinates[8] = prior["response_ns"]
            coordinates = np.r_[coordinates, math.log(prior["delay_ns"])]
            print(f"统一拟合开始: {source}", flush=True)
            try:
                fit_group(
                    [datum],
                    reference,
                    candidate,
                    max_nfev=entry["max_nfev"],
                    start_coordinates=coordinates,
                    optimizer_grid_s=entry["optimizer_grid_ns"] * 1e-9,
                    calibration_mode=entry["calibration_mode"],
                    current_delay_s=prior["delay_ns"] * 1e-9,
                    initial_observed_a=float(datum.current_a[0]),
                    initial_conductance_target_s=85e-6,
                    initial_conductance_relative_scale=0.2,
                    observation_prior=prior,
                    stage_max_nfev=entry.get("stage_max_nfev"),
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                staged = target.with_name(target.name + ".unified-stage")
                backup = target.with_name(target.name + ".unified-backup")
                if staged.exists():
                    shutil.rmtree(staged)
                if backup.exists():
                    shutil.rmtree(backup)
                shutil.copytree(candidate, staged)
                if target.exists():
                    target.rename(backup)
                staged.rename(target)
                if backup.exists():
                    shutil.rmtree(backup)
                candidate_report = json.loads((candidate / "report.json").read_text())
                results.append(
                    {
                        "source": source,
                        "output": output_rel.as_posix(),
                        "candidate_directory": candidate.as_posix(),
                        "optimizer_success": candidate_report["optimizer"]["success"],
                        "adopted": True,
                    }
                )
                print(f"统一拟合完成并采纳: {source}", flush=True)
            except Exception as exc:  # 保留该曲线已有结果并继续处理其他条件
                results.append(
                    {
                        "source": source,
                        "output": output_rel.as_posix(),
                        "candidate_directory": candidate.as_posix(),
                        "adopted": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"统一拟合失败: {source}; {type(exc).__name__}: {exc}", flush=True
                )
            save_json(
                report_path.with_suffix(report_path.suffix + ".partial"),
                {
                    "manifest": manifest_path.as_posix(),
                    "observation_prior": manifest["observation_prior"],
                    "results": results,
                },
            )

    failed_sources = {result["source"] for result in results if not result["adopted"]}
    _clean_unaccepted_pulse_outputs(Path("outputs/pulse"), accepted_output_paths)
    report = {
        "manifest": manifest_path.as_posix(),
        "initial_conductance_target_s": 85e-6,
        "observation_prior": manifest["observation_prior"],
        "results": results,
        "all_adopted": not failed_sources,
    }
    save_json(report_path, report)
    report_path.with_suffix(report_path.suffix + ".partial").unlink(missing_ok=True)
    return report


def plot_result(data, fitted, output, *, filename=None, zoom_peaks=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = 4 if zoom_peaks else 2 if data.instrument == "hires" else 3
    fig, axes = plt.subplots(panels, 1, figsize=(11, 3 * panels))
    t = data.time_s * 1e6
    axes[0].plot(t, data.voltage_v, lw=0.7)
    axes[0].set_ylabel("DUT voltage (V)")
    axes[1].plot(t, data.current_a * 1e3, ".", ms=1.8, label="Measured")
    axes[1].plot(t, fitted * 1e3, lw=0.8, label="Transient model")
    axes[1].set_ylabel("Current (mA)")
    axes[1].legend()
    axes[1].set_xlabel("Time (us)")
    if panels == 3:
        mask = data.time_s < data.time_s[0] + 4e-6
        axes[2].plot(t[mask], data.current_a[mask] * 1e3, ".", ms=2.5, label="Measured")
        axes[2].plot(t[mask], fitted[mask] * 1e3, lw=1, label="Model")
        axes[2].set(xlabel="First cycle time (us)", ylabel="First cycle (mA)")
    if zoom_peaks:
        for ax, name, start in zip(
            axes[2:], ("SET", "RESET"), (0.0, 2e-6), strict=True
        ):
            mask = _phase_mask(data, (start, start + (data.width_ns + 80) * 1e-9))
            x = (data.time_s[mask] - start) * 1e9
            ax.plot(x, data.current_a[mask] * 1e3, ".", ms=2, label="Measured")
            ax.plot(x, fitted[mask] * 1e3, lw=1, label="Aligned transient model")
            ax.set(
                title=f"{name} pulse",
                xlabel="Time from pulse start (ns)",
                ylabel="Current (mA)",
            )
            ax.legend()
    for a in axes:
        a.grid(alpha=0.2)
    fig.suptitle(data.name)
    fig.tight_layout()
    fig.savefig(output / (filename or data.name + ".png"), dpi=160)
    plt.close(fig)


def validate_results(
    output_dir="outputs/pulse",
    input_dir="data/pulse",
    source_names=("pulse_iv_500ns_HiRes.csv",),
):
    """在已拟合参数不变时细化网格/积分容差，区分数据偏差与数值误差。"""
    data, _ = prepare(input_dir, source_names=source_names)
    output = Path(output_dir)
    checks = []
    for datum in data:
        group = datum.instrument
        folder = f"{datum.width_ns}ns_hires" if group == "hires" else "b1530"
        name = datum.name
        if not (output / folder / "fit_config.json").exists() or not any(
            d.name == name for d in data
        ):
            continue
        config = json.loads((output / folder / "fit_config.json").read_text())
        reference = Parameters(**config["reference_parameters"])
        state = State(**config["initial_state"])
        values = np.asarray(config["final_coordinates"])
        observation_options = {
            "current_delay_s": config.get("current_delay_s", 0.0),
            "initial_observed_a": config.get("observation_initial_current_a"),
            "charge_conserving_capacitance": config.get(
                "charge_conserving_capacitance", False
            ),
        }
        coarse, _, _ = predict(
            datum,
            values,
            reference,
            state,
            grid_s=config.get("optimizer_grid_s"),
            include_program_nodes=config.get(
                "optimizer_includes_program_voltage_nodes", False
            ),
            include_priority_nodes=config.get(
                "optimizer_includes_priority_voltage_nodes", False
            ),
            **observation_options,
        )
        fine_grid_s = config["optimizer_grid_s"] / 2 if group == "hires" else 5e-9
        fine, _, _ = predict(
            datum,
            values,
            reference,
            state,
            grid_s=fine_grid_s,
            rtol=0.001,
            include_program_nodes=group == "hires",
            **observation_options,
        )
        delta = float(np.sqrt(np.mean((coarse - fine) ** 2)))
        original_error = float(np.sqrt(np.mean((coarse - datum.current_a) ** 2)))
        check = {
            "name": name,
            "prediction_rms_change_a": delta,
            "prediction_max_change_a": float(np.max(np.abs(coarse - fine))),
            "coarse_rmse_a": original_error,
            "fine_rmse_a": float(np.sqrt(np.mean((fine - datum.current_a) ** 2))),
            "numerical_change_over_data_rmse": delta / original_error,
            "original_parameters_unchanged": True,
        }
        if group == "hires":
            coarse_peaks = _pulse_peak_metrics(datum, coarse)
            fine_peaks = _pulse_peak_metrics(datum, fine)
            check["pulse_peak_relative_change"] = {
                name: float(
                    (
                        abs(fine_peaks[name]["predicted_a"])
                        - abs(coarse_peaks[name]["predicted_a"])
                    )
                    / abs(coarse_peaks[name]["predicted_a"])
                )
                for name in ("set", "reset")
            }
            check["pulse_peak_timing_change_s"] = {
                name: float(
                    fine_peaks[name]["predicted_time_s"]
                    - coarse_peaks[name]["predicted_time_s"]
                )
                for name in ("set", "reset")
            }
            if observation_options["charge_conserving_capacitance"]:
                finest, _, _ = predict(
                    datum,
                    values,
                    reference,
                    state,
                    grid_s=fine_grid_s / 2,
                    rtol=0.0003,
                    include_program_nodes=True,
                    **observation_options,
                )
                finest_peaks = _pulse_peak_metrics(datum, finest)
                check["second_refinement_rms_change_a"] = float(
                    np.sqrt(np.mean((finest - fine) ** 2))
                )
                check["second_refinement_peak_relative_change"] = {
                    name: float(
                        abs(finest_peaks[name]["predicted_a"])
                        / abs(fine_peaks[name]["predicted_a"])
                        - 1
                    )
                    for name in ("set", "reset")
                }
        if name == "pulse_iv_100ns":
            finest, _, _ = predict(
                datum, values, reference, state, grid_s=2.5e-9, rtol=0.0003
            )
            check["second_refinement_rms_change_a"] = float(
                np.sqrt(np.mean((finest - fine) ** 2))
            )
            check["second_refinement_max_change_a"] = float(
                np.max(np.abs(finest - fine))
            )
            check["finest_rmse_a"] = float(
                np.sqrt(np.mean((finest - datum.current_a) ** 2))
            )
        checks.append(check)
    save_json(output / "numerical_validation.json", checks)
    print(json.dumps(checks, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["prepare", "fit", "fit-batch", "validate"],
    )
    parser.add_argument("--device", choices=["b1530", "hires", "all"], default="hires")
    parser.add_argument("--max-nfev", type=int, default=20)
    parser.add_argument(
        "--read-max-nfev",
        type=int,
        help="单独指定 READ 阶段评估次数，便于复用已完成的 READ 校准",
    )
    parser.add_argument(
        "--seed-config", help="以自含 fit_config.json 的参数和最终坐标为起点"
    )
    parser.add_argument(
        "--calibration-mode",
        choices=["waveform", "narrow-peaks"],
        help="窄脉冲模式先校准 READ，再联合优先约束正 SET 与负 RESET 峰值",
    )
    parser.add_argument(
        "--current-delay-ns", type=float, help="由独立对齐确定的固定电流观测时移（ns）"
    )
    parser.add_argument(
        "--response-bounds-ns",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="由独立 READ 边沿约束的响应时间范围（ns）",
    )
    parser.add_argument(
        "--optimizer-grid-ns",
        type=float,
        help="显式指定 HiRes 拟合与最终导出的主网格间隔（ns）；validate 再细化一半",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从现有 fit_config.json 最终坐标继续联合细化",
    )
    parser.add_argument("--input", default="data/pulse")
    parser.add_argument("--output", default="outputs/pulse")
    parser.add_argument("--batch-config", help="fit-batch 使用的逐曲线拟合与采纳清单")
    parser.add_argument("--report", default="configs/pulse_reproduction_report.json")
    parser.add_argument(
        "--dataset",
        action="append",
        help="只处理指定原始 CSV 文件名；例如 pulse_iv_500ns_HiRes.csv",
    )
    parser.add_argument(
        "--initial-conductance-us",
        type=float,
        default=85.0,
        help="初态 HRS 电导软约束目标（µS），默认85",
    )
    args = parser.parse_args(argv)
    if args.resume and args.seed_config:
        parser.error("--resume 与 --seed-config 不能同时使用")
    if args.command == "fit-batch":
        if not args.batch_config:
            parser.error("fit-batch 必须提供 --batch-config")
        if Path(args.output).resolve().is_relative_to(Path("outputs/pulse").resolve()):
            parser.error("批量候选输出必须位于 outputs/pulse 之外")
        result = fit_batch(args.batch_config, args.input, args.output, args.report)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "validate":
        source_names = args.dataset or ["pulse_iv_500ns_HiRes.csv"]
        validate_results(args.output, args.input, source_names)
        return
    source_names = args.dataset
    if args.command == "fit" and args.device == "hires" and source_names is None:
        source_names = ["pulse_iv_500ns_HiRes.csv"]
    data, _ = prepare(args.input, source_names=source_names)
    if args.command == "prepare":
        print("Pulse data prepared:", [d.name for d in data])
        return
    reference = Parameters(
        **json.loads(Path("outputs/dc/cycle1/parameters.json").read_text())
    )
    for instrument in ("b1530", "hires"):
        if args.device in (instrument, "all"):
            selected = [d for d in data if d.instrument == instrument]
            if not selected:
                if args.device == "all":
                    continue
                raise ValueError(f"未找到 {instrument} 输入数据")
            if instrument == "hires" and len(selected) != 1:
                raise ValueError(
                    "HiRes 校准需逐脉宽独立拟合，请通过 --dataset 指定单个文件"
                )
            folder_name = (
                f"{selected[0].width_ns}ns_hires"
                if instrument == "hires"
                else instrument
            )
            folder = Path(args.output) / folder_name
            local_reference = reference
            local_grid = (
                args.optimizer_grid_ns * 1e-9
                if args.optimizer_grid_ns is not None
                else None
            )
            local_delay = (
                args.current_delay_ns * 1e-9
                if args.current_delay_ns is not None
                else 0.0
            )
            local_mode = (
                args.calibration_mode.replace("-", "_")
                if args.calibration_mode
                else "waveform"
            )
            local_initial = None
            local_response_bounds = args.response_bounds_ns
            if args.resume:
                saved_config = json.loads((folder / "fit_config.json").read_text())
                progress = folder / "progress.json"
                saved = (
                    json.loads(progress.read_text())[-1]
                    if instrument == "hires" and progress.exists()
                    else saved_config
                )
                coordinates = saved.get("coordinates", saved.get("final_coordinates"))
                local_reference = Parameters(**saved_config["reference_parameters"])
                if args.optimizer_grid_ns is None:
                    local_grid = saved_config.get("optimizer_grid_s")
                if args.current_delay_ns is None:
                    local_delay = saved_config.get("current_delay_s", 0.0)
                if args.calibration_mode is None:
                    local_mode = saved_config.get("calibration_mode", "waveform")
                local_initial = saved_config.get("observation_initial_current_a")
                if args.response_bounds_ns is None:
                    local_response_bounds = saved_config.get("response_bounds_ns")
            elif args.seed_config:
                seed = json.loads(Path(args.seed_config).read_text())
                coordinates = seed["final_coordinates"]
                local_reference = Parameters(**seed["reference_parameters"])
            elif instrument == "hires" and selected[0].width_ns != 500:
                seed_config = Path(args.output) / "500ns_hires" / "fit_config.json"
                coordinates = json.loads(seed_config.read_text())["final_coordinates"]
            else:
                coordinates = None
            fit_group(
                selected,
                local_reference,
                folder,
                max_nfev=args.max_nfev,
                start_coordinates=coordinates,
                optimizer_grid_s=local_grid,
                calibration_mode=local_mode,
                current_delay_s=local_delay,
                initial_observed_a=local_initial,
                response_bounds_ns=local_response_bounds,
                read_max_nfev=args.read_max_nfev,
                initial_conductance_target_s=args.initial_conductance_us * 1e-6,
            )


if __name__ == "__main__":
    main()
