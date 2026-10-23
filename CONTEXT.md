# 接续上下文

更新：2026-10-23。此文件是压缩后的项目工作记忆，配合 `AGENTS.md` 和 `TODO.md` 使用，不是完整历史报告。

## 当前目标与成果

目标是面向氧化物 RRAM 模拟突触，用物理紧凑模型解释 DC 阻变、脉冲 SET/RESET，进一步验证脉宽对 LTP/LTD 线性度、电导窗口和粒度的影响。

- 基础论文为本地 Huang 等 IEEE TED 2013 PDF。已实现确定性细丝间隙/直径演化、金属+hopping 传导、串联接触和可选漏电、自热。
- `rram_model.py` 实现器件公式及详细中文机理注释；`fit_dc.py` 实现 DC 数据及校准；`fit_pulse.py` 实现脉冲数据转换、有限观测响应和校准。
- 瞬态新增有限热弛豫与断电冷却、有效端子电容电流。端电压是已测驱动，不求完整发生器输出电路；通道滤波与器件温度是不同状态/作用。
- 当前保留两套独立的条件校准，未证明参数可以跨器件或跨脉宽通用。尚无 LTP/LTD、随机开关、保持弛豫或传输线振铃模型。

## 保留的实测基线

| 数据 | 来源与处理 | 最终结果 |
| --- | --- | --- |
| DC cycle 3 | `data/dc/dc_iv_cycle3.csv`；404 点；按约 0.5 ms/点重建时间；设定电压；仅正偏 SET 限流 1 mA | `outputs/measured_dc/cycle3/refined/`；全周期 RMSE 57.19 µA |
| HiRes 500 ns | `data/pulse/pulse_iv_500ns_HiRes.csv`；25000 点；单周期 SET–READ–RESET–READ；不限流 | `outputs/pulse/hires/`；全量 RMSE 50.09 µA，READ-LRS/HRS 4.00/19.42 µA |

HiRes 用户确认的关键信息：

- 原 `voltage_v` 是对地 Vforce，原 `current_a` 实际为 Vread。必须换算 `I_DUT=Vread/50 Ω`、`V_DUT=Vforce−Vread`；不再次加入已扣除的读出电阻压降。
- 时间有效精度不足，约 0.16 ns 节拍可从前段识别；文件第 12501 个数据点归零，前半为 SET–READ，后半为 RESET–READ，用户确认可按无额外等待的同周期连接。
- 原始文件未改动。转换数据和原文件 SHA256 在 `data/pulse/prepared/`，转换 CSV 保留源时间/电压/读取列。
- 固定条件为 L=5 nm、最大直径 5 nm、种子直径 0.5 nm，初态 x=0.15 nm、w=2.5 nm、x0=1 nm、T=300 K，均不是独立实测几何。

## 参数解释与重要限制

- DC 接受结果是 cycle 3 的独立传导/动力学校准，完整初值与边界在 `data/dc/prepared/cycle3_refined_config.json`。初态直接记录在配置中，不依赖已删除的周期数据。
- HiRes 的 `parameters.json`、`transient_options.json` 和 `fit_config.json` 保存最终器件参数、通道/瞬态参数、固定状态、优化边界及坐标。配置自含参考参数和最终坐标；现有 CLI 新拟合入口仍把 DC 文件当初值尺度，后续解耦见 TODO。
- HiRes 有效电容约 1.04e-16 F，接近零边界；热时间约 0.2015 ns，接近 0.2 ns 搜索下限；电流通道响应约 14.13 ns。不能将这些数值直接认定为本征电容或独立测得的热时间。
- HiRes 优化按 `xtol` 停止。几何、温升、能垒、传导温度系数和通道响应仍相关；能描述主波形，但残余振铃未完全解释。
- 同测点收紧网格/容差后，HiRes 预测电流 RMS 变化约 0.003 µA，远小于实测拟合误差；数值稳定不等于物理参数唯一或跨条件预测通过。

## 已完成的清理决定

用户接受 DC cycle 3 和 HiRes 拟合。其他 DC 周期及全部 B1530A 原始/整理/拟合数据已删除。旧跨脉宽对比、历史报告、阶段日志、初始曲线已清理。通用仪器转换代码与测试保留，但不要恢复被删除的数据或报告。

最终图/预测只含测量与接受的最终模型，未保留初始曲线列。`report.json` 是精简的最终机器校准指标，保留以复核结果；它不是旧历史对比报告。

## 验证与工作习惯

当前完整基线：37 tests passed，18 subtests passed。HiRes-only 校准/导出已做临时目录烟雾检查；使用已存配置重放的预测保持不变。新增结果写入独立目录，避免覆盖接受的基线。

环境采用 uv，Python >=3.11；`.venv` 已安装。沙箱有时不能写默认 uv 缓存，见 AGENTS.md 中的 `/tmp` 缓存命令。
Python 文件现按 `pyproject.toml` 中的 Ruff 规则检查与格式化；提交前命令见 `AGENTS.md`。
