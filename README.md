# RRAM Synapse Modeling

## 1. 仓库简介

本仓库以Huang等人2013年IEEE TED论文 *A Physics-Based Compact Model of Metal-Oxide-Based RRAM DC and AC Operations* 为基础，建立氧化物RRAM的确定性物理紧凑模型，处理DC I–V、单周期Pulse和多脉冲LTP/LTD。

三类实验共用器件方程，但各自具有加压协议、初态、观测方式和校准参数。目标是解释与复现实验中的电流、READ电导及模型状态演化；校准结果不等于真实细丝形貌、参数唯一性或跨条件预测能力的证明。

本文第5节明确列出的结果为当前已采纳结果。`outputs/`也可以容纳后续待选结果，**目录存在不代表已采纳**。DC/Pulse目前采用用户恢复的原结果，后续被撤回的重新拟合批次不再使用。LTP/LTD仅保留100/1 ns，10 ns实测数据因趋势问题已撤回；单周期Pulse的10 ns是另一项实验，继续保留。

Python 3.11+，依赖由 `pyproject.toml` 和 `uv.lock` 管理，使用 `uv sync --group dev` 安装。

## 2. RRAM模型参数介绍及物理假设

### 2.1 状态、观测量与单位

| 量 | 代码表示 | 单位 | 含义 |
|---|---|---|---|
| x | `State.x_m` | m | 细丝轴向绝缘间隙；x=0表示接通 |
| w | `State.w_m` | m | 新生/有效导电细丝的**直径**，不是半径 |
| x0 | `State.x0_m` | m | 断裂区域历史尺度；RESET扩大间隙时可增长，SET时保留 |
| T | `OperatingPoint.temperature_k` | K | 模型局部温度；瞬态中连续演化 |
| G | READ观测计算 | S | 指定READ窗口的平均电流/平均器件电压，不是独立状态变量 |

状态满足 `0 ≤ x ≤ x0 ≤ thickness_m`，直径限制在 `seed_diameter_m` 和 `filament_max_m` 之间。内部采用m/A/V/s/K；能垒以eV输入，与 `KB_EV` 配合使用。绘图中的nm、μA、μS、ns仅为单位换算。

### 2.2 器件与观测参数

以下为 `rram_model.Parameters` 和 `TransientOptions` 的参数分组。**类默认值不是实验校准值**；各条件的实际数值、初态、拟合边界和固定项以对应输出的JSON配置为准。

| 分组 | 参数与单位 | 物理作用 |
|---|---|---|
| 几何 | `thickness_m`、`filament_max_m`、`seed_diameter_m`（m） | 氧化层厚度、残余细丝直径/有效直径上界、合法直径下界 |
| 原子尺度 | `atomic_step_m`（m）、`attempt_hz`（Hz）、`charge_number`（无量纲） | 跳跃步长、速率前因子、氧离子电荷数绝对值 |
| 反应能垒 | `generation_ev`、`hopping_ev`、`release_ev`、`recombination_ev`（eV） | 空位生成、离子迁移、电极释放及复合弛豫；影响SET/RESET动力学 |
| 场增强与扩径 | `generation_alpha_m`、`hopping_alpha_m`、`radial_shell_m`（m）；`release_gamma`（无量纲） | 生成/迁移的场致能垒降低、径向生成尺度、电压增强的离子释放 |
| 金属传导 | `conductivity_s_m`（S/m）、`conduction_ev`（eV） | 电导率前因子及温度依赖；公式为 `σ=σ0 exp(E_AC/(kBT))`，指数为正号 |
| 间隙hopping传导 | `hopping_density_a_m2`（A/m²）、`hopping_length_m`（m）、`hopping_voltage_v`（V） | `J=J0 exp(−x/x_T) sinh(V_gap/V_T)`；电流还需乘有效面积 |
| 热 | `ambient_k`（K）、`thermal_k_w`（K/W）、`thermal_time_s`（s） | 环境温度、有效热阻及热弛豫时间；时间常数为零时采用准静态热近似 |
| 电路 | `series_ohm`、`parallel_ohm`（Ω）、`pad_capacitance_f`（F） | 外部总串联电阻、可选并联漏电阻、端子有效电容；并联电阻为None表示开路 |
| 方向与保护 | `set_polarity`、`temperature_limit_k`（K） | SET极性约定及数值/适用性温度保护；后者不是物理饱和机制 |
| 测量响应 | Pulse保存配置中的响应时间、观测延迟、积分时间等 | 描述测量通道，对应观测模型；不改变器件内部状态的真实时间轴 |

面积与电导率/电流密度、频率与能垒、初始间隙与hopping衰减尺度、热参数与反应速率存在相关性。单条曲线不足以独立识别所有参数，应区分固定假设和被拟合参数。

### 2.3 物理假设与模型边界

- 以近似圆柱形细丝及连续反应速率描述开关，不模拟单离子随机事件。SET先缩小间隙，闭合后可扩径；RESET主要通过间隙演化降导，没有额外经验缩径窗口或逐周期参数。
- 主支路包含残余金属段及新生细丝/间隙，另有背景hopping路径。背景路径、历史尺度x0及亚原子间隙的处理属于工程闭合假设，不能当作论文逐项原式。
- 间隙未闭合时使用实际间隙压降计算局部场；闭合后改用内部电压与历史/原子尺度计算，避免除以零。x小于原子步长时采用连续接触闭合，详见代码注释。
- DC准静态温升由开关元件导电耗散决定；瞬态采用有限热弛豫，断电后连续冷却。外部接触电阻、可选并联漏电阻的耗散及电容充电功率不计入局部细丝焦耳热。
- 导电、电容电流与通道响应分开；有效端子电容概括电极/夹具，不直接解释为氧化层本征电容。HiRes使用器件端差分电压驱动，不额外求解发生器输出电路。

详细公式、论文原式与扩展的区别见 [MODEL_NOTES.md](MODEL_NOTES.md) 和 `rram_model.py` 中文注释。

## 3. RRAM仿真优化算法

### 3.1 前向求解

给定输入电压、初态和参数，求解电流、内部压降及温度，并积分x、w、x0。DC限流通过降低实际器件端电压实现；瞬态保留热历史，连续执行WRITE、READ、冷却和SET→RESET连接。

状态积分采用带误差控制和边界投影的Heun方法，自动调整步长。瞬态热弛豫在积分中处理，零偏段解析冷却；场辅助速率限制有效能垒不低于零。求解失败显式报告，不用有限大残差或截温掩盖失效。

### 3.2 三类校准

| 任务 | 目标与优化方法 |
|---|---|
| DC | 有界最小二乘；线性电流与有符号asinh残差按窗口组合。分阶段配方依次校准HRS传导、SET动力学、LRS传导、RESET动力学及条件联合目标，可使用初值筛选和差分进化全局搜索。对正值参数采用对数坐标，SET可用参考场有效能垒坐标减轻退化；精确设置见配置 |
| Pulse | 每条HiRes独立进行有界least-squares；波形目标按READ→RESET→SET优化，窄脉冲峰值目标先约束READ再强调SET/RESET峰值。使用按参数边界跨度定义的有限差分Jacobian；保留先前目标的接受规则，拒绝退化候选。初态电导、通道响应、电容与延迟可有软约束 |
| LTP/LTD | 每组独立有界least-squares，soft-L1损失、`f_scale=0.1`；筛选暖启动及多种初始直径后联合优化。完整积分所有周期，仅压缩READ残差：每阶段最多20个趋势窗口，并约束首末和早期READ；不拟合WRITE瞬态电流 |

LTP/LTD当前配方每组最多20次函数评价，`ftol=0.03`、`xtol=1e-4`、`diff_step=0.001`、`x_scale=jac`。DC/Pulse采用各自保存配方与预算，不能用LTP设置代替。优化器success或达到评价上限只说明数值终止状态，不代表物理参数正确或误差达标。

### 3.3 验证与候选运行

固定参数细化步长、网格和容差，使用相同测点、READ窗口及误差定义比较；分别记录拟合误差、数值变化和独立预测结果。物理/求解器修改还要核对公式极限、状态边界、连续性、电热/电容守恒及冷却。

```bash
uv run pytest
uvx --from ruff==0.16.10 ruff check .
uvx --from ruff==0.16.10 ruff format --check .
```

新运行默认写独立候选目录，读取已采纳参数不等于允许覆盖原结果。例如：

```bash
uv run python fit_dc.py simulate outputs/dc/cycle1/config_used.json --output /tmp/rram-dc-replay
uv run python fit_dc.py fit configs/dc_cycle1_balanced.json --output /tmp/rram-dc-candidate
uv run python fit_pulse.py fit --device hires --dataset pulse_iv_20ns_HiRes.csv \
  --seed-config outputs/pulse/20ns_hires/fit_config.json --output /tmp/rram-pulse-candidate
uv run python fit_ltp.py --unified --output /tmp/rram-ltp-candidate
```

重新拟合可能落入不同局部解，不能将其称为保存结果的精确重放。Pulse验证应先复制待验证组到独立目录，再把 `--output` 指向副本；`fit-batch`即使候选 `--output` 在别处，仍会采纳到固定的 `outputs/pulse/` 并清理未列出项。`simulate_ltp_independent.py`目前也固定写入已采纳仿真目录。**未经用户要求更新已采纳结果，不得直接执行这些会写回采纳目录的入口；需要隔离副本或先为候选运行实现独立输出。**

## 4. 仓库目录和功能

### 4.1 输入、配置与说明文件

| 路径 | 类型与用途 |
|---|---|
| `data/dc/dc_iv_cycle1.csv` | DC原始输入，404点扫描；`data/dc/prepared/`为转换副本、审计和配方 |
| `data/pulse/pulse_iv_*_HiRes.csv` | 八组单周期原始输入；`data/pulse/prepared/`为转换副本与审计 |
| `data/ltp_ltd/ltp_ltd_{100,1}ns.csv` | 两组多脉冲原始输入；10 ns已撤回，不恢复 |
| `configs/dc_cycle1_balanced.json` | DC分阶段校准配方，不等于保存的最终参数 |
| `configs/accepted_pulse_fits.json` | 八组Pulse的采纳清单、种子、网格、目标和优化预算 |
| `configs/pulse_reproduction_report.json` | 采纳Pulse批次的来源及历史复现对照，不是原始测量 |
| `configs/ltp_ltd_hires.json` | 两组LTP自含暖启动及热/电容选项；历史来源路径只用于追溯 |
| `MODEL_NOTES.md`、`DC_CALIBRATION.md` | 模型公式/假设与DC基线/保留对照的设计说明 |
| `AGENTS.md`、`TODO.md`、`DEBUG_HISTORY.log` | 最重要的工作约束、待办和失败/撤回历史；历史记录不是当前采纳清单 |
| `pyproject.toml`、`uv.lock` | Python依赖及测试/Ruff配置 |

原始CSV不覆盖、不排序、不取电流绝对值、不平均周期或静默丢点；转换只在副本进行，保留源列和SHA256。HiRes统一使用 `I=vsense_v/50 Ω`、`V_DUT=vforce_v−vsense_v`，已扣除50 Ω压降、不限流；不能套用DC的0.5 ms/点和SET 1 mA限流。

### 4.2 Python脚本与对应输出

| 脚本 | 功能 | 当前对应的结果或产物 |
|---|---|---|
| `rram_model.py` | 共同器件方程、DC/瞬态前向求解器；由其他脚本调用 | 无独立默认输出，支撑全部仿真 |
| `fit_dc.py` | `simulate`固定参数仿真、`fit`校准、`validate`数值细化、`demo`合成示例 | 已采纳DC两组目录；新任务由 `--output` 指定，导出参数、配置、预测、报告、曲线或数值检查 |
| `fit_pulse.py` | `prepare`转换数据、`fit`逐组校准、`fit-batch`批量校准并采纳、`validate`固定参数细化 | 准备副本与八组Pulse目录；注意batch的固定采纳路径，普通fit/validate也不能指向只读结果 |
| `fit_ltp.py` | 准备100/1 ns数据，重建驱动、连续校准、数值对照及汇总 | 当前 `ltp_ltd_hires/`；新候选必须显式给独立 `--output`，默认目录受只读规则保护 |
| `simulate_ltp_independent.py` | 读取各组LTP最终参数，按缩短协议预测x/w/G/T及周期峰温，并细化对照 | 固定默认写 `ltp_ltd_simulate/`；读取源为 `ltp_ltd_hires/`，未经授权不能直接运行覆盖 |
| `tests/test_model.py` | 器件公式、物理极限及基础求解 | 只产生测试报告/缓存 |
| `tests/test_fit.py` | DC数据、优化、CLI和采纳DC固定参数重放 | 同上 |
| `tests/test_transient.py` | 瞬态、电热/电容、Pulse数据与采纳Pulse重放 | 同上 |
| `tests/test_ltp_data.py` | LTP换算、READ目标、暖启动和两组CLI | 同上 |
| `tests/test_ltp.py` | LTP的SET→RESET几何和温度连续性 | 同上 |
| `tests/test_simulate_ltp_independent.py` | 固定参数仿真的READ观测定义 | 同上 |

### 4.3 输出文件含义

`outputs/`存放生成结果，包含已采纳结果和未来可能新增的待选结果。常见文件含义如下，具体以各目录实际内容为准。

| 文件 | 含义 |
|---|---|
| `parameters.json`、`initial_state.json`、`transient_options.json` | 器件参数、初态、热/电容选项；有些任务将这些合并在运行配置中 |
| `config_used.json`、`fit_config.json` | 自含运行/校准配置、输入来源、求解或观测设置及最终坐标 |
| `prediction.csv` | 与实测点对应的预测，或逐READ的x/w/G/T预测，列定义见文件头 |
| `trajectory.csv` | 完整协议的连续输出节点与状态，不是每个内部自适应积分步 |
| `report.json`、`summary.json`、`summary.md` | 拟合误差、终止原因、参数来源、接受/数值检查或跨组汇总 |
| `numerical_validation.json`、JSON内的 `numerical_checks` | 固定参数数值细化记录，与拟合质量分开 |
| `audit.json`、`prepared.csv`、`reads.csv` | 数据来源和SHA256、转换副本、原实验READ提取；大体积准备副本通常不跟踪 |
| `seed_screen.json` | 初始直径候选及其目标代价 |
| `comparison.png`、`fit.png`、`states.png` | 曲线比较、READ拟合或状态图，不是额外测量数据 |

## 5. 已采纳结果和对应仿真方法与假设

### 5.1 明确的采纳清单与只读边界

**以下明确列出的结果为已采纳结果，默认只读。除非用户明确要求更新相应已采纳结果，否则不得修改、覆盖、删除、重命名或重新生成其中的任何文件，包括参数、配置、数据副本、图和汇总。**可以只读检查，或复制到独立位置进行重放与验证。本文描述这些结果，不自动授权重跑写回。

| 已采纳路径 | 范围与身份 |
|---|---|
| `outputs/dc/cycle1/` | 当前DC基线，全部内容 |
| `outputs/dc/cycle1_balanced/` | 明确保留的DC条件对照，全部内容；不替换cycle1基线 |
| `outputs/pulse/{1,2,5,10,20,100,200,500}ns_hires/` | 八组当前Pulse结果，各组全部内容 |
| `outputs/ltp_ltd_hires/{100,1}ns/` | 两组LTP校准来源，各组全部内容 |
| `outputs/ltp_ltd_hires/{summary.json,summary.md,comparison.png}` | 上述两组的顶层汇总 |
| `outputs/ltp_ltd_simulate/{100,1}ns/` | 两组固定参数预测，各组全部内容 |
| `outputs/ltp_ltd_simulate/{summary.json,comparison.png}` | 上述两组的顶层汇总 |

花括号表示明确枚举，不表示接受任意新增条件。新增待选目录或文件不自动进入清单；只有用户明确采纳后才更新本节并纳入只读范围。仅要求修改代码、文档、运行测试或生成候选，不构成更新已采纳结果的授权。

### 5.2 DC基线与保留对照

输入为404点实测扫描，约0.5 ms/点，仅SET限流1 mA。初态x=1 nm、w=0.5 nm、x0=1 nm为假设；采用准静态电热求解及连续几何状态积分。

| 结果 | 全周期电流RMSE | 方法与限制 |
|---|---:|---|
| [cycle1](outputs/dc/cycle1/comparison.png) | 46.48 μA | 基线校准；保存的最终配方与参数见本目录config_used.json |
| [cycle1_balanced](outputs/dc/cycle1_balanced/comparison.png) | 45.66 μA | 分区强调正偏HRS/SET，条件联合细化；改善局部但返回LRS退化，因此保留作对照 |

对照的正偏HRS、SET邻域、返回LRS、RESET/HRS区域RMSE为0.65、27.99、60.42、57.31 μA。扩径尺度与能垒/场增强相关性仍限制物理解释；热阻、几何及边界是条件假设。数值细化见对照目录，设计与权衡见 [DC_CALIBRATION.md](DC_CALIBRATION.md)。完整分阶段配方重新优化不保证复现保存的局部解。

### 5.3 八组HiRes Pulse

输入为实测器件端差分电压，重建舍入重复时间；两个半周期无额外等待连接，完整执行SET–READ–RESET–READ。器件方程与DC相同，增加热弛豫、端子电容及独立观测响应；测量电流不能直接等同导电电流。

八组独立校准，初态HRS电导85 μS为软目标。保存的观测软先验中心为延迟14.5 ns、响应7.352864 ns、电容5.394015 pF，相对尺度20%，最终值允许优化，并非固定共享材料常数。必须整体使用各组保存的初态、参数、最终坐标、网格与观测选项。

| 脉宽 | 保存的优化目标 | 主网格间隔 | 全波形RMSE（μA） |
|---|---|---:|---:|
| 1 ns | narrow_peaks | 0.5 ns | 32.18 |
| 2 ns | narrow_peaks | 0.125 ns | 34.23 |
| 5 ns | narrow_peaks | 0.5 ns | 49.04 |
| 10 ns | narrow_peaks | 0.5 ns | 48.20 |
| 20 ns | waveform | 0.5 ns | 67.08 |
| 100 ns | waveform | 1 ns | 67.32 |
| 200 ns | waveform | 0.5 ns | 37.36 |
| 500 ns | waveform | 1 ns | 41.59 |

目标、阶段预算和来源见 [Pulse采纳清单](configs/accepted_pulse_fits.json)，历史复现对照见 [复现报告](configs/pulse_reproduction_report.json)。各组图、参数、预测与局部误差位于对应已采纳目录。全波形RMSE不代表每个READ/WRITE局部均吻合；采纳也不证明参数唯一或全部条件已独立验证。通用入口保留B1530格式支持，当前原始Pulse数据只有HiRes八组。

### 5.4 LTP/LTD独立校准

100/1 ns周期均为2 μs，SET和RESET各50/1000次，每次WRITE后READ。WRITE按+1.2/−1.3 V源设定和10%脉宽边沿重建，再扣除插值的实测sense压降；零源段零偏冷却。READ源波形在1.0–1.05 μs降至−0.4 V，1.5–1.55 μs返回零。

实测READ电导为周期内1.15–1.45 μs的平均电流/平均器件电压，模型在同一窗口用7节点平均导电电流/平均器件电压。全协议连续积分，仅优化残差采用趋势压缩。初始有效直径在0.5–2.5 nm拟合，扩径上限5 nm，初态电导不强制100 μS；筛选暖启动及0.65/0.8/1.0/1.5/2.5 nm直径候选。

| 脉宽 | 范围归一化RMSE | 首READ偏差 | 初始/最终直径 |
|---|---:|---:|---:|
| 100 ns | 13.11% | +214.66 μS | 1.417/5.000 nm |
| 1 ns | 6.82% | +12.90 μS | 0.829/5.000 nm |

见[两组校准比较图](outputs/ltp_ltd_hires/comparison.png)、[可读摘要](outputs/ltp_ltd_hires/summary.md)及[机器摘要](outputs/ltp_ltd_hires/summary.json)。保存的趋势及数值对照通过：趋势要求方向正确、首尾10%窗口中位变化量为实测的0.5–2倍；这是工程判据，不是实验误差界。

优化rtol=1e-3、最大步长50 ns，固定参数用3e-4/25 ns细化，逐READ差异门槛为2 μS＋2%×细化值。初次失败保留，再以1e-4/12.5 ns和3e-5/6.25 ns继续对照，不覆盖初次判定。

100 ns首READ偏高、生成能垒近上界及SET早期饱和仍未解决；1 ns WRITE欠采样重建存在输入不确定性。几何与温度不是独立测量；本目录温度为每周期1.3 μs采样，不是严格峰温。

### 5.5 LTP/LTD固定参数仿真

`ltp_ltd_simulate`各组读取自身 `ltp_ltd_hires` 最终参数、初态和热/电容选项，**不重新拟合**。100/1 ns分别执行10/1000次SET及同样次数RESET：各取原SET、RESET阶段前N周期，提前连接RESET，保持原周期内加压/READ/冷却时序及连续状态。

结果见[两组x/w/G/T比较图](outputs/ltp_ltd_simulate/comparison.png)及[参数和收敛记录](outputs/ltp_ltd_simulate/summary.json)。x、w、T取周期内1.3 μs，G沿用1.15–1.45 μs READ窗口；另存每周期内部积分的峰值温度。准备副本和reads.csv仍对应原完整实验，不能作为缩短序列的对应实测验证。

两组保存的数值检查通过：对x、w、x0、G、READ温度和峰温使用1%相对容差，绝对容差为1 pm、1 pm、1 pm、0.01 μS、0.1 K、0.1 K。缩短后的RESET没有对应实测验证；组间参数、脉冲数、READ次数和总历时均不同，不能据此推断纯脉宽效应。
