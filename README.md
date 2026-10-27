# RRAM Synapse 高速脉冲建模

本项目使用 Python 建立氧化物 RRAM 的物理紧凑模型，研究 DC I–V 与 SET–READ–RESET–READ 瞬态。当前接受的基线为 DC cycle 1 与八组 HiRes 脉宽条件拟合。当前状态见 [CONTEXT.md](CONTEXT.md)，未完成工作见 [TODO.md](TODO.md)，历史调试过程见 [DEBUG_HISTORY.log](DEBUG_HISTORY.log)。

## 模型

器件方程集中在 [rram_model.py](rram_model.py)，基于 Huang 等人的 *A Physics-Based Compact Model of Metal-Oxide-Based RRAM DC and AC Operations*（IEEE TED, 2013），并保留中文方程、单位和数值保护说明。模型包含间隙与细丝几何状态、场辅助空位生成、离子迁移/释放/复合、金属与 hopping 传导、串联接触电阻以及电热耦合。瞬态求解保留状态历史和断电冷却；测量观测中的传导电流、电容电流和电流通道响应分别处理，电容充电功率不作为细丝焦耳热。

[fit_dc.py](fit_dc.py) 负责 DC 数据整理、拟合与验证；[fit_pulse.py](fit_pulse.py) 负责脉冲数据转换、观测响应和瞬态拟合。HiRes 原始列按 `I_DUT = vsense_v / 50 Ω`、`V_DUT = vforce_v - vsense_v` 转换。原始数据保留字节、行序和测点；舍入时间按已确认的采样节拍重建，两个半周期按实验确认连接。

## 拟合算法

DC cycle 1 使用分区混合目标和分阶段参数校准，最终细化覆盖全部原始测点。脉冲拟合使用统一器件模型与 `fit_group` 分阶段 least-squares 优化内核，每条脉宽曲线独立拟合，不施加跨脉宽参数一致性约束。八组均以 85 μS 为初始 HRS 电导软目标，相对尺度 20%；观测延迟、响应和电容以 1/2/5 ns 锚点初始化，并使用 20% 相对软先验。宽脉冲以 READ、RESET、SET 波形为优先目标；窄脉冲先校准 READ，再优先拟合 SET/RESET 峰值。曲线指标由用户整体验收，旧结果间的差异不作为自动回退条件。

## 当前接受基线

| 数据 | 结果目录 | 全记录电流 RMSE |
| --- | --- | ---: |
| DC cycle 1 | `outputs/dc/cycle1/` | 46.48 μA |
| HiRes 1 ns | `outputs/pulse/1ns_hires/` | 32.18 μA |
| HiRes 2 ns | `outputs/pulse/2ns_hires/` | 34.23 μA |
| HiRes 5 ns | `outputs/pulse/5ns_hires/` | 49.04 μA |
| HiRes 10 ns | `outputs/pulse/10ns_hires/` | 48.20 μA |
| HiRes 20 ns | `outputs/pulse/20ns_hires/` | 73.75 μA |
| HiRes 100 ns | `outputs/pulse/100ns_hires/` | 67.32 μA |
| HiRes 200 ns | `outputs/pulse/200ns_hires/` | 37.36 μA |
| HiRes 500 ns | `outputs/pulse/500ns_hires/` | 41.59 μA |

各 HiRes 目录保存参数、拟合配置、全采样预测、对比图、指标报告和瞬态选项。各脉宽的 READ 与峰值指标见对应 `report.json`。拟合参数为单条件校准值，不代表唯一可辨识的材料常数，也未验证跨脉宽预测能力。

DC 与 HiRes 参数分开保存。DC cycle 1 使用模型初态假设；脉冲初始 HRS 目标为软约束，因此最终拟合的初始电导可以偏离 85 μS。未完成的测量与物理可辨识性事项列于 TODO。

## 复现与检查

依赖由 `pyproject.toml` 和 `uv.lock` 管理，Python 3.11+。

```bash
uv sync --group dev
uv run pytest
uv run python fit_pulse.py prepare
```

统一拟合清单为 [configs/accepted_pulse_fits.json](configs/accepted_pulse_fits.json)。批量运行会将成功拟合的曲线写入标准基线目录，并清理未列入清单的脉冲结果；应先确认清单和输出目录设置：

```bash
uv run python fit_pulse.py fit-batch \
  --batch-config configs/accepted_pulse_fits.json \
  --output /tmp/rram-pulse-candidates
```

单曲线重拟合应写入独立目录，避免覆盖接受基线。`fit_pulse.py validate` 和 `fit_dc.py validate` 用于固定参数下的数值检查，不等同于重新拟合或实验验证。
