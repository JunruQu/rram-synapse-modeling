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
| DC cycle 1 | 全部 404 点 | 46.48 µA |
| HiRes 500 ns SET–READ–RESET–READ | 全部 25000 点 | 50.09 µA |
| 同一 HiRes 的 READ-LRS | 周期内 1.15–1.45 µs | 4.00 µA |
| 同一 HiRes 的 READ-HRS | 周期内 3.15–3.45 µs | 19.42 µA |

DC 与 HiRes 参数分别保存，未假定可以直接通用。DC cycle 1 初态采用模型默认高阻态（`x=1 nm, w=0.5 nm, x0=1 nm`），是条件假设，不是独立实测。以上是单周期、给定几何和初态的条件校准，不是跨器件、跨脉宽预测验证。全周期误差与固定 READ 窗口误差均按原始采样点计算，没有通过平均或删除测量点降低误差。

### 保留文件

- DC 原始数据：`data/dc/dc_iv_cycle1.csv`；整理数据、配置及审计：`data/dc/prepared/`。
- DC 最终结果：`outputs/dc/cycle1/`，统一包含 `parameters.json`、`prediction.csv`、`comparison.png`、`report.json` 和运行配置。
- HiRes 原始数据：`data/pulse/pulse_iv_500ns_HiRes.csv`；整理数据为 `data/pulse/prepared/500ns_hires_prepared.csv`，审计为同目录 `audit.json`。
- HiRes 最终结果：`outputs/pulse/500ns_hires/`，与 DC 统一使用 `parameters.json`、`prediction.csv`、`comparison.png` 和 `report.json`，并保留 `fit_config.json`、`transient_options.json`。

上述是当前接受的实验基线。中间参数轨迹、初始预测曲线、多条件对比报告及旧汇总已清理；新的 DC 校准候选按下节独立存放。

## DC 校准算法更新

针对正偏 HRS 电流偏低和 SET 过渡偏缓，新增分区混合目标、传导/SET/LRS/RESET 分阶段校准及最终条件联合细化。HRS 使用有符号 `asinh` 误差，SET 保留局部线性电流约束；耦合参数先联合搜索，数值 Jacobian 可按边界跨度控制差分步长。全部 404 点仍参与最终目标，每个窗口保留完整状态历史。原因、参数作用、窗口与权重见 [DC_CALIBRATION.md](DC_CALIBRATION.md)，配置见 [configs/dc_cycle1_balanced.json](configs/dc_cycle1_balanced.json)。

```bash
uv run python fit_dc.py fit configs/dc_cycle1_balanced.json --output outputs/dc/cycle1_balanced
uv run python fit_dc.py simulate outputs/dc/cycle1_balanced/config_used.json --output /tmp/dc_cycle1_replay
uv run python fit_dc.py validate outputs/dc/cycle1_balanced/config_used.json --output outputs/dc/cycle1_balanced
```

本次[候选曲线](outputs/dc/cycle1_balanced/comparison.png)的正偏 HRS / SET 邻域 RMSE 分别为 0.65 / 27.99 µA，旧基线为 76.01 / 168.43 µA；全周期由 46.48 降至 45.66 µA，但返回 LRS 区由 11.15 增至 60.42 µA。扩径尺度接近上界，参数仍强相关；这是揭示模型折中的候选，不替换接受基线。相同测点、窗口和数值收敛结果见 [DC_CALIBRATION.md](DC_CALIBRATION.md)。

新结果写入独立目录；接受基线不变。保存的 DC 配置现在可直接重放最终参数；`validate` 固定参数，在相同测点逐级细化积分并导出数值变化，不能代替实验验证。DC 拟合不再自动生成 `initial/`。

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
uv run python fit_dc.py fit data/dc/prepared/fit_config.json --output outputs/dc/cycle1/refit
```

HiRes 的完整初始参数、优化边界、最终坐标及固定状态均保存在 `outputs/pulse/500ns_hires/fit_config.json`。单独检验数值细化可运行 `uv run python fit_pulse.py validate`；该命令只处理当前存在的数据与配置，不重拟合参数，并会生成新的数值检查文件。

今后新增实验请保留实际器件端 V–t/I–t、编程和读出条件、初始阻态、脉冲间隔及重复统计。优先以共享参数预测未参与校准的条件，再依据独立证据扩展物理状态。

## 本次 DC cycle 1 debug 总结（2026-10-26）

本次针对正偏高阻态电流偏低、SET 电流跃迁过缓重新设计校准流程。沿用 404 个原始测点、0.5 ms/点和仅正偏 SET 限流 1 mA 的条件；器件方程与积分器保持原样，拟合候选单独保存。

### 物理机制与参数耦合

| 机制 | 相关参数 | 本次诊断 |
| --- | --- | --- |
| HRS hopping 传导 | `hopping_density_a_m2`（`J0`）、`hopping_voltage_v`（`V_T`）、间隙与 `hopping_length_m`、有效面积 | `J0` 控制幅值，`V_T` 控制电压非线性；初始间隙、衰减长度和面积可与幅值互相补偿，单条回线不能唯一分离。旧模型在 0.4 V 仅为 4.73 µA，实测为 10.5 µA。 |
| SET 间隙闭合与后续扩径 | `generation_ev`（`Ea`）、`generation_alpha_m`（`alpha_a`）、`radial_shell_m`（`Delta w`） | 场辅助空位生成先使间隙闭合，再使细丝扩径。单独降低 `Ea` 容易使 SET 提前；`alpha_a` 影响场敏感度，`Delta w` 直接影响闭合后的直径增长。旧模型在约 0.78 V 已闭合并缓慢扩径，而实测到 0.86 V 仍只有 29.9 µA。 |
| 电热反馈与金属传导 | `thermal_k_w`（`Rth`）、`conductivity_s_m`、`conduction_ev` | 自热加快离子过程，同时因 `sigma(T)=sigma0*exp(E_AC/(kBT))` 的正指数而降低金属电导率，SET、LRS 和 RESET 会一起改变。释放热阻的对照损害了 LRS，本次候选保留基线 `Rth=5,000 K/W`，未据此认定真实热阻。 |
| 亚原子间隙与闭合后的局部场 | `x`、`x0`、原子尺度 `a` 及既有场闭合假设 | SET 间隙闭合后采用 `E=abs(U)/max(x0,a)`；该场分配与亚原子区压降插值可能影响 SET 形状。固定初态 `x=x0=1 nm, w=0.5 nm` 是条件假设，不能当作实测形貌。 |

旧基线收紧积分容差与步长后，电流 RMS 变化仅约 0.0043 µA，远小于拟合误差，说明缓慢 SET 主要不能归因于积分误差。实测突变只被定位在 0.86–0.88 V、0.5 ms 的采样间隔内，不能从插值曲线推断更快的本征开关时间。

### 拟合与校准算法调整

1. **分区混合目标。** 全周期线性电流误差继续约束 mA 级区域；HRS 增加以 1 µA 为尺度的有符号 `asinh` 误差，SET 邻域增加局部线性约束。各块按点数归一化，显式权重表达校准优先级，不能当作仪器噪声方差；最终目标覆盖全部 404 点。
2. **先传导、后动力学。** 先校准初始低场段的 `J0,V_T`，再联合搜索 SET 的能垒、场敏感度和扩径尺度，随后校准 LRS 金属电导及 RESET 动力学。各窗口均保留此前状态历史，仅可省去目标末点之后的未来计算。
3. **参考场坐标与联合搜索。** 用 `b=Ea-Z*alpha_a*Eref`、`alpha_a`、`Delta w` 搜索，参考场取 `Eref=8.7e8 V/m`；采用固定随机种子的差分进化再局部细化，并核对坐标搜索域不超出声明的物理能垒边界。`b` 是优化坐标，不是新增物理能垒，也不替代实际局部场；本次对照仍有强相关，坐标变换不能补足缺失的实验信息。
4. **条件联合细化与可辨识性检查。** `Ea` 与 `alpha_a` 的局部相关系数曾接近 1。最终候选固定已搜索的 SET 参数，再联合优化传导和 RESET；报告区分固定参数与待拟合参数，并检查边界、相关性及优化器梯度。`success`、`xtol` 或条件 Jacobian 满秩均不证明物理校准完成。
5. **匹配数值导数与积分精度。** 新增按优化坐标边界跨度确定差分步长的方式，内部用中心差分、边界用单侧差分，避免同一个相对步长对 `log(alpha_a)` 和 eV 能垒产生失衡扰动。新增 `fit_dc.py validate` 固定参数逐级细化积分，检查电流、端电压、温度、几何和限流标记；保存配置可直接重放最终参数，拟合不再自动导出 `initial/`。

### 同测点结果与保留结论

下表为电流 RMSE（µA），行区间从 0 开始、右端不含；全周期使用相同的 404 个原始测点。

| 区域 | 原始行区间 | 接受基线 | 新候选 |
| --- | --- | --- | --- |
| 初始低场 HRS | `[0,36)` | 6.52 | 0.50 |
| 正偏 HRS | `[0,44)` | 76.01 | 0.65 |
| SET 邻域 | `[35,50)` | 168.43 | 27.99 |
| 返回 LRS / RESET 起始 | `[160,240)` | 11.15 | 60.42 |
| RESET 及返回 HRS | `[240,404)` | 51.56 | 57.31 |
| 全周期 | `[0,404)` | 46.48 | 45.66 |

新候选在 0.4 V 为 10.19 µA，0.88 V 为 0.892 mA，0.90 V 到达 1 mA 限流。正偏 HRS 和 SET 局部明显改善，但返回 LRS 退化，且扩径尺度约 4.997 nm、接近 5 nm 搜索上界；最终虽按 `xtol` 停止，梯度指标仍不小。因此保留为条件校准候选，不替换已接受的 DC / HiRes 基线。

最终配置重放的 404 点电流逐点一致。两级积分细化的电流 RMS 变化约为 0.0072 / 0.00027 µA，限流标记不变；SET 附近间隙的最大变化约为 0.091 / 0.0021 nm，说明电流稳定不等于亚原子几何已被唯一确定。验证为 40 项测试通过，Ruff 静态与格式检查通过。

完整设计见 [DC_CALIBRATION.md](DC_CALIBRATION.md)，重新拟合配方见 [configs/dc_cycle1_balanced.json](configs/dc_cycle1_balanced.json)。候选结果在 `outputs/dc/cycle1_balanced/`，按现有 `.gitignore` 规则保留为本地生成结果；重放使用该目录的 `config_used.json`，完整分阶段重新拟合可能到达不同局部候选。后续应以独立初态、低扰动 I–V、温度约束及更密集 SET 数据检验几何与场闭合假设。
