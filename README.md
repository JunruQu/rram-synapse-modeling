# RRAM Synapse 高速脉冲建模

本项目面向氧化物 RRAM 模拟突触，以 Huang 等的 *A Physics-Based Compact Model of Metal-Oxide-Based RRAM DC and AC Operations*（IEEE TED, 2013）为基础，建立 Python 物理紧凑模型。长期目标是用共享参数解释不同编程脉宽下的 SET/RESET、LTP/LTD 电导更新及更新线性度；目前完成了 DC 和单周期脉冲的初步条件校准。

接续工作先读 [CONTEXT.md](CONTEXT.md)；项目工作约定见 [AGENTS.md](AGENTS.md)，后续事项见 [TODO.md](TODO.md)。已接受的原始数据和最终基线纳入 Git，临时生成结果与环境缓存默认忽略。

## 已有建模成果

| 内容 | 已实现能力 |
| --- | --- |
| 几何状态 | 间隙长度 `x`、细丝直径 `w`、断裂区域记忆 `x0`；连续传递状态 |
| 开关机制 | 场辅助空位生成；RESET 采用氧离子释放、迁移、复合中最慢步骤；SET 先闭合间隙再扩径 |
| 电子传导 | 类金属传导与 hopping 路径，考虑串联接触电阻和可选漏电 |
| DC 求解 | 电流、电压、准静态自热联立求解，支持仅 SET 限流 |
| 瞬态求解 | 有限热响应及断电冷却、有效端子电容电流；READ 期间也按实际电压演化状态 |
| 测量观测 | 一阶电流通道响应及可选积分窗口，与器件热响应分开处理 |
| 参数提取 | 有界校准、初值搜索、分阶段拟合及数值收敛检查 |

器件数学公式集中在 [rram_model.py](rram_model.py)，含详细中文参数和物理机理注释。数据读取、优化、观测处理分别在 [fit_dc.py](fit_dc.py) 和 [fit_pulse.py](fit_pulse.py)。原始论文的方程对应及 DC 简化见 [MODEL_NOTES.md](MODEL_NOTES.md)。

## 当前保留的实验与校准

| 数据 | 评价范围 | 电流 RMSE |
| --- | --- | --- |
| DC cycle 3 | 全部 404 点 | 57.19 µA |
| HiRes 500 ns SET–READ–RESET–READ | 全部 25000 点 | 50.09 µA |
| 同一 HiRes 的 READ-LRS | 周期内 1.15–1.45 µs | 4.00 µA |
| 同一 HiRes 的 READ-HRS | 周期内 3.15–3.45 µs | 19.42 µA |

DC 与 HiRes 参数分别保存，未假定可以直接通用。以上是单周期、给定几何和初态的条件校准，不是跨器件、跨脉宽预测验证。全周期误差与固定 READ 窗口误差均按原始采样点计算，没有通过平均或删除测量点降低误差。

### 保留文件

- DC 原始数据：`data/dc/dc_iv_cycle3.csv`；整理输入及配置：`data/dc/prepared/`。
- DC 最终参数、预测和曲线：`outputs/measured_dc/cycle3/refined/`。
- HiRes 原始数据：`data/pulse/pulse_iv_500ns_HiRes.csv`；转换输入和追溯审计：`data/pulse/prepared/`。
- HiRes 最终结果：`outputs/pulse/hires/`，包含 `parameters.json`、`transient_options.json`、`fit_config.json`、`pulse_iv_500ns_HiRes_prediction.csv`、`pulse_iv_500ns_HiRes.png` 及精简的 `report.json` 校准指标。

仅保留上述实验数据和最终结果。中间参数轨迹、初始预测曲线、多条件对比报告及旧汇总已清理。

## HiRes 数据与瞬态模型

HiRes 原 CSV 的列为 `time_s,vforce_v,vsense_v`：`vforce_v` 是对地施加电压 Vforce，`vsense_v` 是 50 Ω 取样电阻上的读取电压 Vsense。转换为：

- 器件电流：`I_DUT = vsense_v / 50 Ω`。
- 器件差分电压：`V_DUT = vforce_v − vsense_v`。
- 测试不限流；50 Ω 读出电阻压降已扣除，不重复加入器件接触电阻。

原时间列存在舍入重复值，并在 SET–READ 与 RESET–READ 两半之间归零。程序依据前段约 0.16 ns 的采样节拍重建时间，逐行核对原始十进制精度，按已确认的无额外间隔连接两个半周期；保持行顺序和全部测量点。转换 CSV 保留原时间、电压及读取列用于追溯，原始文件不改写。

瞬态入口由实测器件差分电压驱动，点间线性插值。温度按 `dT/dt = (T0 + Pcond*Rth − T) / tau_th` 演化；`tau_th=0` 为准静态热近似。有效端子电容电流为 `Cpad*dV_DUT/dt`，不计入细丝焦耳热。电流通道响应时间与热时间常数是不同参数；没有独立零点标定，不擅自减去电流偏置。

本轮固定几何尺度为氧化层 5 nm、最大细丝直径 5 nm、种子直径 0.5 nm；初态为 `x=0.15 nm, w=2.5 nm, x0=1 nm, T=300 K`。这些是条件假设，不是实测细丝尺寸。首点已处于脉冲前沿，不凭空补出未测预触发波形；整个记录内不重置状态。

### 参数与适用边界

- DC 有效校准了传导尺度、hopping 电压尺度及部分 SET/RESET 动力学；参数仍与几何和估计扫描时间相关。
- HiRes 联合调整了有效传导、开关、热与测量响应参数，能够描述主要编程和读出波形；残余振铃尚未完全描述。
- 有效端子电容接近零边界，热时间约 0.20 ns 接近搜索下限；不能将它们当作已独立测得的本征电容或热时间。
- 按步长条件终止不证明全局最优。单条波形不能唯一分离能垒、温升、几何及通道带宽。
- 尚未实现或验证 LTP/LTD、随机开关、保持弛豫、分布参数测试线及电感反射。脉宽改善更新线性度的机制仍需多条件实验验证。

## 环境与复现

依赖由 `pyproject.toml` 和 `uv.lock` 管理，Python 3.11+，使用 NumPy、SciPy、Matplotlib 和 pytest。

```bash
uv sync --group dev
uv run pytest
uv run python fit_pulse.py prepare
```

重新校准时写入独立目录，保留已接受的结果：

```bash
uv run python fit_pulse.py fit --device hires --max-nfev 30 --output outputs/pulse/refit
uv run python fit_dc.py fit data/dc/prepared/cycle3_refined_config.json --output outputs/measured_dc/cycle3/refit
```

HiRes 的完整初始参数、优化边界、最终坐标及固定状态均保存在 `outputs/pulse/hires/fit_config.json`。单独检验数值细化可运行 `uv run python fit_pulse.py validate`；该命令只处理当前存在的数据与配置，不重拟合参数，并会生成新的数值检查文件。

今后新增实验请保留实际器件端 V–t/I–t、编程和读出条件、初始阻态、脉冲间隔及重复统计。优先以共享参数预测未参与校准的条件，再依据独立证据扩展物理状态。
