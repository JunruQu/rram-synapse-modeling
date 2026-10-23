"""SET–READ–RESET–READ 数据整理、观测模型与校准，默认使用 HiRes。

器件方程均在 rram_model.py。此处仅负责单位/极性转换、导出时间精度修复、
有限积分窗口、通道响应及优化。原始数据永不覆盖；插值不用于恢复
未测的脉冲前沿或亚采样开关时刻。
"""
import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares

from rram_model import Parameters, State, TransientOptions, simulate_transient, ModelError, KB_EV


@dataclass
class PulseData:
    name: str
    time_s: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    width_ns: int
    instrument: str
    integration_s: float


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def rebuild_time(values, allow_reset=False):
    """只修复有原始前段采样节拍证据的舍入时间，不对波形按时间排序。

    重复时间保留所有行。独立采集的 HiRes 两个半周期按用户确认的无额外
    间隔连接；每段的原始相位/舍入误差单独核对，连接处保持一个采样间隔。
    """
    raw = np.asarray(values, float)
    if len(raw) < 3 or not np.all(np.isfinite(raw)):
        raise ValueError('时间数据不完整')
    cuts = np.flatnonzero(np.diff(raw) < 0)+1
    if len(cuts) and not allow_reset:
        raise ValueError('出现未经允许的时间归零')
    limits = [0, *cuts.tolist(), len(raw)]
    intervals = []
    for lo, hi in zip(limits, limits[1:]):
        d = np.diff(raw[lo:min(hi, lo+30)])
        if np.any(d <= 0):
            raise ValueError('前段时间不能提供明确的采样节拍')
        intervals.append(float(np.median(d)))
    dt = float(np.median(intervals))
    if not np.allclose(intervals, dt, rtol=1e-5, atol=1e-18):
        raise ValueError('不同采集段采样间隔不同，不能自动连接')
    # 用原 CSV 数字表示的小数位检查每一行是否与均匀采样舍入相容。
    from decimal import Decimal
    for lo, hi in zip(limits, limits[1:]):
        reconstructed = raw[lo]+np.arange(hi-lo)*dt
        tolerances = np.array([.51001*float(Decimal(1).scaleb(Decimal(s).as_tuple().exponent)) for s in values[lo:hi]])
        if np.any(np.abs(reconstructed-raw[lo:hi]) > tolerances+max(dt*1e-5, 1e-18)):
            raise ValueError('时间偏差超过导出精度，不能假定均匀采样')
    return raw[0]+np.arange(len(raw))*dt, {
        'sampling_interval_s': dt, 'duplicate_raw_timestamps': int(np.sum(np.diff(raw) == 0)),
        'time_reset_rows_zero_based': cuts.tolist(),
        'method': 'uniform cadence inferred from unrounded leading samples, validated against CSV decimal precision; retain row order',
        'segment_join': 'one sample interval, no extra delay' if len(cuts) else None}


def prepare(input_dir='data/pulse', output_dir='data/pulse/prepared'):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit = {'conditions': {
        'b1530_current': 'I_DUT = -source current_a',
        'hires_current': 'I_DUT = source current_a / 50 ohm (source column is Vread)',
        'hires_voltage': 'V_DUT = source voltage_v - source current_a (Vforce - Vread)',
        'compliance': 'none, both devices', 'b1530_integration_s': 10e-9,
        'devices': 'all B1530A files are one device; HiRes is another device',
        'hires_join': 'SET-READ then RESET-READ, user confirmed no extra interval',
        'voltage_input': 'measured DUT differential voltage, linearly interpolated; no sub-sampling waveform deconvolution'}, 'files': []}
    data = []
    for source in sorted(Path(input_dir).glob('pulse_iv_*.csv')):
        with source.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ['time_s', 'voltage_v', 'current_a']:
                raise ValueError(f'{source}: 列名不符')
            rows = list(reader)
        array = np.array([[float(r[k]) for k in ('time_s', 'voltage_v', 'current_a')] for r in rows])
        if not np.all(np.isfinite(array)):
            raise ValueError(f'{source}: NaN/Inf')
        hires = 'HiRes' in source.name
        ts, info = rebuild_time([r['time_s'] for r in rows], allow_reset=hires)
        v = array[:, 1]-array[:, 2] if hires else array[:, 1]
        current = array[:, 2]/50 if hires else -array[:, 2]
        width = int(source.stem.split('_')[2].removesuffix('ns'))
        instrument = 'hires' if hires else 'b1530'
        datum = PulseData(source.stem, ts, v, current, width, instrument, 0 if hires else 10e-9)
        data.append(datum)
        with (output/source.name).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time_s', 'voltage_v', 'current_a', 'source_time_s', 'source_voltage_v', 'source_current_column'])
            writer.writerows(zip(ts, v, current, array[:, 0], array[:, 1], array[:, 2]))
        info.update(name=source.name, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), rows=len(rows),
                    instrument=instrument, programming_width_ns=width,
                    voltage_range_v=[float(v.min()), float(v.max())], current_range_a=[float(current.min()), float(current.max())],
                    resolution_warning='50 ns sampling cannot resolve 20/50 ns pulse shape or precise switching delay' if not hires and width <= 50 else None)
        audit['files'].append(info)
    if not data:
        raise ValueError('未找到脉冲 CSV')
    if not any(d.instrument == 'b1530' for d in data):
        audit['conditions'].pop('b1530_current')
        audit['conditions'].pop('b1530_integration_s')
        audit['conditions']['devices'] = 'HiRes oscilloscope device'
        audit['conditions']['compliance'] = 'none'
    save_json(output/'audit.json', audit)
    return data, audit


def measurement_response(time_s, current_a, sample_times, tau_s=0.0, integration_s=0.0, offset_a=0.0):
    """电流通道一阶有效响应，再按已知积分窗口平均；不更改状态/温度。

    tau 包含电流通道与未辨识链路的有效带宽，不能当作热时间常数。
    B1530A 的积分窗口固定 10 ns，绝不将采样间隔 50 ns 当作积分时间。
    首个滤波状态取输入首值；缺失预触发段的影响需在结果说明中保留。
    """
    t, u = np.asarray(time_s), np.asarray(current_a)
    if tau_s < 0 or integration_s < 0:
        raise ValueError('响应时间和积分时间不可为负')
    filtered = u.copy()
    if tau_s:
        for k, h in enumerate(np.diff(t), 1):
            one_minus = -math.expm1(-h/tau_s)
            # 线性变化输入的精确一阶滤波，避免不均匀网格上的显式不稳定。
            filtered[k] = ((1-one_minus)*filtered[k-1]+one_minus*u[k-1]
                           +(u[k]-u[k-1])*(1-tau_s/h*one_minus))
    if integration_s:
        integral = cumulative_trapezoid(filtered, t, initial=0)
        low = np.maximum(t[0], sample_times-integration_s/2)
        high = np.minimum(t[-1], sample_times+integration_s/2)
        if np.any(high <= low):
            raise ValueError('积分窗口在仿真区间之外')
        values = (np.interp(high, t, integral)-np.interp(low, t, integral))/(high-low)
    else:
        values = np.interp(sample_times, t, filtered)
    return values+offset_a


def decode(values, reference):
    """优化坐标采用易读尺度，对数只用于跨数量级的正值量。"""
    sigma, density, ea, er, gamma, rth, tau, cap, response, eac = values
    # 用室温有效电导率与温度系数分开参数化，避免 sigma0 与 EAC 在室温
    # 指数项中直接互相补偿。它们仍不是独立温度测量得到的材料常数。
    p = replace(reference, conductivity_s_m=5e5*math.exp(sigma-eac/(KB_EV*reference.ambient_k)),
                conduction_ev=eac, hopping_density_a_m2=1e13*math.exp(density),
                generation_ev=ea, recombination_ev=er, release_gamma=gamma, thermal_k_w=1e5*math.exp(rth))
    options = TransientOptions(thermal_time_s=1e-9*math.exp(tau), pad_capacitance_f=cap*1e-12)
    return p, options, response*1e-9


INITIAL = np.array([0., 0., 1.65, .64, .31, math.log(3.), math.log(5.), 2., 15., .02])
LOWER = np.array([math.log(.2), math.log(.1), 1.1, .45, .10, math.log(.01), math.log(.2), 0., 0., 0.])
UPPER = np.array([math.log(5.), math.log(5.), 2.2, .85, .8, math.log(10.), math.log(100.), 100., 100., .15])
NAMES = ['log_sigma300_over_5e5', 'log_J0_over_1e13', 'generation_ev', 'recombination_ev', 'release_gamma',
         'log_Rth_over_1e5', 'log_thermal_tau_ns', 'pad_capacitance_pf', 'current_response_ns', 'conduction_ev']


def predict(data, values, reference, initial_state, *, grid_s=None, rtol=.003):
    p, options, response = decode(values, reference)
    step = grid_s or (10e-9 if data.instrument == 'b1530' else 2e-9)
    # B1530A 首点在 25 ns，缺少前沿零点。显式补一个 t=0,V=0 的边界假设，
    # 不把它写回原始测量。HiRes 已处于前沿，避免人为在 0.32 ns 内从零跳变。
    start = 0.0 if data.instrument == 'b1530' else data.time_s[0]
    end = data.time_s[-1]+data.integration_s/2
    # 保留全部输入电压节点，尤其不能用均匀网格略过欠采样脉冲唯一的峰点。
    # 细化网格只增加求解点，不改变实测节点间的线性驱动定义。
    grid = np.unique(np.r_[np.linspace(start, end, int(np.ceil((end-start)/step))+1), data.time_s])
    tx, vx = data.time_s, data.voltage_v
    if data.instrument == 'b1530':
        tx, vx = np.r_[0., tx], np.r_[0., vx]
    voltage = np.interp(grid, tx, vx)
    result = simulate_transient(grid, voltage, initial_state, p, options, rtol=rtol, atol_m=1e-13,
                                max_step_s=10e-9 if grid_s is None else grid_s)
    # 没有独立零点标定时不擅自减去电流。特别是 HiRes 的 V_DUT 已减 Vread，
    # 按 |V_DUT| 很小筛选会反过来选择某些 Iread，造成内生的偏置估计。
    offset = 0.0
    observed = measurement_response(grid, result.total_a, data.time_s, response, data.integration_s, offset)
    return observed, result, offset


def residual_for(data, predicted):
    phase = np.mod(data.time_s, 4e-6)
    read = ((phase > 1.1e-6) & (phase < 1.6e-6)) | ((phase > 3.1e-6) & (phase < 3.6e-6))
    program = (phase < (data.width_ns+150)*1e-9) | ((phase > 2e-6) & (phase < 2e-6+(data.width_ns+150)*1e-9))
    # 零偏压噪声不主导器件参数；全部测点仍导出并计算无权 RMSE。
    weight = np.where(read | program, 1.0, .1)
    return np.sqrt(weight/weight.sum())*(np.arcsinh(predicted/1e-4)-np.arcsinh(data.current_a/1e-4))


def fit_group(data, reference, output, *, max_nfev=20, start_coordinates=None, include_all_widths=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    group = data[0].instrument
    # 几何不可由单端脉冲唯一提取；采用与 DC 一致的尺度，仅作为条件校准。
    state = State(.15e-9, 2.5e-9, 1e-9)
    initial = INITIAL.copy()
    if group == 'hires':
        initial[0], initial[8] = math.log(.6), 2.
    if start_coordinates is not None:
        initial = np.asarray(start_coordinates, dtype=float)
        if initial.shape != LOWER.shape or np.any(initial < LOWER) or np.any(initial > UPPER):
            raise ValueError('续拟合坐标不在边界内')
    # B1530A 用 50/200/500 ns 训练，100 ns 留出，20 ns 分辨率不足仅作预测。
    training = [d for d in data if group == 'hires' or include_all_widths or d.width_ns in (50, 200, 500)]
    # 高分辨率训练保留每 8 点及末点；完整 25000 点在最终导出时评估。
    fit_data = [replace(d, time_s=d.time_s[np.r_[np.arange(0, len(d.time_s)-1, 8), len(d.time_s)-1]],
                        voltage_v=d.voltage_v[np.r_[np.arange(0, len(d.time_s)-1, 8), len(d.time_s)-1]],
                        current_a=d.current_a[np.r_[np.arange(0, len(d.time_s)-1, 8), len(d.time_s)-1]])
                if group == 'hires' else d for d in training]
    values = initial.copy()
    history = []
    calls = 0

    def full_residual(candidate):
        nonlocal calls
        calls += 1
        result = []
        for datum in fit_data:
            pred, _, _ = predict(datum, candidate, reference, state)
            result.extend(residual_for(datum, pred))
        return np.asarray(result)

    first = full_residual(initial)
    stages = [('transport_and_channel', [0, 1, 7, 8, 9]),
              ('switching_and_heat', [2, 3, 4, 5, 6]),
              ('joint', list(range(10)))]
    if start_coordinates is not None:
        stages = [('joint_refinement', list(range(10)))]
    for label, indices in stages:
        anchor = values.copy()
        def residual(z):
            candidate = anchor.copy(); candidate[indices] = z
            return full_residual(candidate)
        def jacobian(z):
            # 使用按边界跨度定义的绝对差分，避免对数坐标=0 时退回极小
            # 相对步长，导致状态积分误差压过数值导数。
            base = residual(z)
            jac = np.empty((len(base), len(z)))
            for j, index in enumerate(indices):
                h = .002*(UPPER[index]-LOWER[index])
                step = h if z[j]+h <= UPPER[index] else -h
                shifted = z.copy(); shifted[j] += step
                jac[:, j] = (residual(shifted)-base)/step
            return jac
        start = time.perf_counter()
        try:
            result = least_squares(residual, values[indices], bounds=(LOWER[indices], UPPER[indices]),
                                   x_scale='jac', jac=jacobian, max_nfev=max_nfev,
                                   ftol=2e-4, xtol=2e-4, gtol=1e-5)
        except (ModelError, ValueError, OverflowError) as exc:
            # 不将物理求解失败当作巨大常量残差；保留已成功阶段以便诊断。
            save_json(output/'failure.json', {'stage': label, 'error': str(exc), 'last_valid_coordinates': values.tolist()})
            raise
        values[indices] = result.x
        info = {'stage': label, 'success': bool(result.success), 'message': str(result.message),
                'cost': float(result.cost), 'nfev': int(result.nfev), 'seconds': time.perf_counter()-start,
                'optimality': float(result.optimality), 'coordinates': values.tolist()}
        history.append(info)
        save_json(output/'progress.json', history)
        print(group, label, 'cost', info['cost'], 'seconds', round(info['seconds'], 1), flush=True)
    p, options, response = decode(values, reference)
    save_json(output/'parameters.json', asdict(p))
    save_json(output/'transient_options.json', {**asdict(options), 'current_response_s': response})
    save_json(output/'fit_config.json', {'instrument': group, 'source_names': [d.name for d in data],
              'training_names': [d.name for d in training], 'initial_state': asdict(state),
              'reference_parameters': asdict(reference), 'initial_coordinates': initial.tolist(),
              'final_coordinates': values.tolist(), 'coordinate_names': NAMES,
              'include_all_widths': include_all_widths,
              'lower': LOWER.tolist(), 'upper': UPPER.tolist(), 'max_nfev_per_stage': max_nfev,
              'scope_current_conversion': 'Vread / 50', 'scope_voltage_conversion': 'Vforce - Vread',
              'compliance': None, 'rtol': .003, 'atol_m': 1e-13})
    statistics = []
    for datum in data:
        fitted, sim, offset = predict(datum, values, reference, state)
        phase = np.mod(datum.time_s, 4e-6)
        masks = {'all': np.ones(len(fitted), dtype=bool),
                 'read_lrs': (phase >= 1.15e-6) & (phase <= 1.45e-6),
                 'read_hrs': (phase >= 3.15e-6) & (phase <= 3.45e-6)}
        metric = {'name': datum.name, 'training': datum.name in [d.name for d in training],
                  'rows': len(fitted), 'offset_a': offset, 'rmse_a': {},
                  'max_temperature_k': max(op.temperature_k for op in sim.points)}
        for name, mask in masks.items():
            metric['rmse_a'][name] = float(np.sqrt(np.mean((fitted[mask]-datum.current_a[mask])**2)))
        statistics.append(metric)
        with (output/(datum.name+'_prediction.csv')).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time_s', 'voltage_v', 'measured_a', 'predicted_a', 'conductive_a', 'capacitive_a', 'temperature_k', 'x_m', 'w_m'])
            grid = np.asarray(sim.time_s)
            writer.writerows(zip(datum.time_s, datum.voltage_v, datum.current_a, fitted,
                np.interp(datum.time_s, grid, [op.current_a for op in sim.points]),
                np.interp(datum.time_s, grid, sim.capacitive_a),
                np.interp(datum.time_s, grid, [op.temperature_k for op in sim.points]),
                np.interp(datum.time_s, grid, [s.x_m for s in sim.states]),
                np.interp(datum.time_s, grid, [s.w_m for s in sim.states])))
        plot_result(datum, fitted, output)
    final = full_residual(values)
    notes = ['No current compliance', 'No independent zero-current calibration supplied: no data-dependent current offset subtraction']
    if group == 'hires':
        notes.append('I_DUT = Vread / 50 ohm; V_DUT = Vforce - Vread')
    else:
        notes.append('50 ns sampling cannot resolve the 20/50 ns drive shape or precise switching delay')
    save_json(output/'report.json', {'instrument': group,
              'optimizer': {k: history[-1][k] for k in ('success', 'message', 'nfev', 'optimality')},
              'final_squared_residual': float(final@final), 'statistics': statistics,
              'parameters_identifiability': 'effective conditional calibration; no uniqueness or cross-device transfer claim',
              'notes': notes})
    (output/'failure.json').unlink(missing_ok=True)
    (output/'progress.json').unlink(missing_ok=True)
    return values


def plot_result(data, fitted, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = 2 if data.instrument == 'hires' else 3
    fig, axes = plt.subplots(panels, 1, figsize=(11, 3*panels))
    t = data.time_s*1e6
    axes[0].plot(t, data.voltage_v, lw=.7); axes[0].set_ylabel('DUT voltage (V)')
    axes[1].plot(t, data.current_a*1e3, '.', ms=1.8, label='Measured')
    axes[1].plot(t, fitted*1e3, lw=.8, label='Transient model')
    axes[1].set_ylabel('Current (mA)'); axes[1].legend()
    axes[1].set_xlabel('Time (us)')
    if panels == 3:
        mask = (data.time_s < data.time_s[0]+4e-6)
        axes[2].plot(t[mask], data.current_a[mask]*1e3, '.', ms=2.5, label='Measured')
        axes[2].plot(t[mask], fitted[mask]*1e3, lw=1, label='Model')
        axes[2].set(xlabel='First cycle time (us)', ylabel='First cycle (mA)')
    for a in axes:a.grid(alpha=.2)
    fig.suptitle(data.name); fig.tight_layout(); fig.savefig(output/(data.name+'.png'), dpi=160);plt.close(fig)


def validate_results(output_dir='outputs/pulse'):
    """在已拟合参数不变时细化网格/积分容差，区分数据偏差与数值误差。"""
    data, _ = prepare()
    output = Path(output_dir)
    checks = []
    for group, name in (('b1530', 'pulse_iv_100ns'), ('b1530', 'pulse_iv_500ns'), ('hires', 'pulse_iv_500ns_HiRes')):
        if not (output/group/'fit_config.json').exists() or not any(d.name == name for d in data):
            continue
        config = json.loads((output/group/'fit_config.json').read_text())
        reference = Parameters(**config['reference_parameters'])
        state = State(**config['initial_state'])
        datum = next(d for d in data if d.name == name)
        values = np.asarray(config['final_coordinates'])
        coarse, _, _ = predict(datum, values, reference, state)
        fine, _, _ = predict(datum, values, reference, state,
                             grid_s=5e-9 if group == 'b1530' else 1e-9, rtol=.001)
        delta = float(np.sqrt(np.mean((coarse-fine)**2)))
        original_error = float(np.sqrt(np.mean((coarse-datum.current_a)**2)))
        check = {'name': name, 'prediction_rms_change_a': delta,
                       'prediction_max_change_a': float(np.max(np.abs(coarse-fine))),
                       'coarse_rmse_a': original_error,
                       'fine_rmse_a': float(np.sqrt(np.mean((fine-datum.current_a)**2))),
                       'numerical_change_over_data_rmse': delta/original_error,
                       'original_parameters_unchanged': True}
        if name == 'pulse_iv_100ns':
            finest, _, _ = predict(datum, values, reference, state, grid_s=2.5e-9, rtol=.0003)
            check['second_refinement_rms_change_a'] = float(np.sqrt(np.mean((finest-fine)**2)))
            check['second_refinement_max_change_a'] = float(np.max(np.abs(finest-fine)))
            check['finest_rmse_a'] = float(np.sqrt(np.mean((finest-datum.current_a)**2)))
        checks.append(check)
    save_json(output/'numerical_validation.json', checks)
    print(json.dumps(checks, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'fit', 'validate'])
    parser.add_argument('--device', choices=['b1530', 'hires', 'all'], default='hires')
    parser.add_argument('--max-nfev', type=int, default=20)
    parser.add_argument('--resume', action='store_true', help='从现有 fit_config.json 最终坐标继续联合细化')
    parser.add_argument('--include-all-widths', action='store_true', help='全部 B1530A 脉宽共同校准；不再将100ns称为留出预测')
    parser.add_argument('--input', default='data/pulse')
    parser.add_argument('--output', default='outputs/pulse')
    args = parser.parse_args(argv)
    if args.command == 'validate':
        validate_results(args.output);return
    data, _ = prepare(args.input)
    if args.command == 'prepare':
        print('Pulse data prepared:', [d.name for d in data]);return
    reference = Parameters(**json.loads(Path('outputs/measured_dc/cycle3/refined/parameters.json').read_text()))
    for instrument in ('b1530', 'hires'):
        if args.device in (instrument, 'all'):
            selected = [d for d in data if d.instrument == instrument]
            if not selected:
                if args.device == 'all':
                    continue
                raise ValueError(f'未找到 {instrument} 输入数据')
            folder = Path(args.output)/instrument
            coordinates = json.loads((folder/'fit_config.json').read_text())['final_coordinates'] if args.resume else None
            fit_group(selected, reference,
                      folder, max_nfev=args.max_nfev, start_coordinates=coordinates,
                      include_all_widths=args.include_all_widths)


if __name__ == '__main__':
    main()
