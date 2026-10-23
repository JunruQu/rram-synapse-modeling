"""Huang 等（IEEE TED, 2013）的确定性 RRAM 紧凑模型：DC 与实测脉冲。

物理图像
========
氧空位聚集形成近似圆柱形导电细丝（CF）。SET 时，电场降低氧空位生成
能垒，细丝先沿轴向跨越间隙，再沿径向扩展；RESET 时，电极释放的氧离子
迁移至细丝并与空位复合，造成间隙增长。本文的 x 是间隙长度，w 是直径，
不是半径。模型以连续速率代替单个离子的随机事件，不模拟随机开关。

方程来源
========
原文式 (5)：sigma = sigma0 * exp(E_AC / (k_B*T))；注意这里是正号，故升温
会降低金属型细丝电导。式 (19)：J_hop = J0*exp(-x/x_T)*sinh(V_gap/V_T)。
原文 I0 给出的单位为电流/面积，因此这里明确以电流密度 J0 表示并乘面积。
式 (20)：T = T0 + I_switch*V_internal*R_th，不包括外部接触电阻的耗散。
式 (11)--(13)、(15)、(18) 见 switching_rates 的中文说明。

首版闭合假设（并非原文逐项给出的方程）
====================================
采用图 5 的等效路径：残余金属段与“新生细丝金属段+间隙”串联，同时有
未被新生细丝占据的面积通过 x0 长度的 hopping 路径导电。为减少未知量，
背景路径忽略残余金属段压降。x0 为上次断裂区域尺度，RESET 过程中跟随
更大的 x 增长；小于历史最大间隙的复杂局部回线不作为本版验证目标。
电场取实际间隙压降/x（亚原子区压降也随 x 缩小），间隙闭合后取
V_internal/max(x0,a)，避免 x=0 时出现无穷大场。x<a 的亚原子区将 hopping 压降乘 x/a，使接触
电阻连续归零；这是首版数值闭合假设，x>=a 时严格使用式 (19)。几何
厚度、初态和边界需由器件信息给出。

数值方法
========
电流、内部电压和准静态温度联立求解；限流通过降低器件端实际电压实现。
状态采用带误差控制的投影 Heun 积分，边界附近自动缩短步长。原子振动
频率作为场辅助跃迁的上限（有效能垒不低于零），避免负能垒指数溢出。
所有计算使用 SI 单位；能量参数输入为 eV，k_B 也以 eV/K 表示，电场降低
项 Z*alpha*E 的数值单位为 eV，因此无需再乘一次基本电荷。

DC 内核仅依赖标准库；新增瞬态求解使用 NumPy。数据、优化器和绘图
分别位于 fit_dc.py、fit_pulse.py。脉冲扩展使用有限热弛豫时间和跨实测
端子的有效电容，保留原 DC 接口，详见 TransientOptions。
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from itertools import pairwise

KB_EV = 8.617333262145e-5  # 玻尔兹曼常数，eV/K；所有热激活指数使用此单位。


class ModelError(RuntimeError):
    """电热求解或状态积分失败；不能将失败点默默作为拟合结果。"""


@dataclass(frozen=True)
class Parameters:
    """器件参数。下面每项说明单位、来源、物理意义及校准建议。

    标为“表 I”的默认值来自原文；几何、温度和数值边界是首版示例假设。
    默认值不是本项目实验器件的校准结果。DC 单条回线无法唯一识别全部
    参数，尤其 f 与能垒、面积与 sigma0/J0、x 与 x_T 常高度相关。
    """

    # ---------- 几何：SI 长度均为 m ----------
    thickness_m: float = 5e-9
    # 示例氧化层厚度 L；不是表 I 数值。应从工艺确定并固定，增大 L 增大金属电阻。
    filament_max_m: float = 5e-9
    # 示例残余细丝直径 w0，也为新生细丝的直径上界；截面积 pi*w0^2/4。
    # 较大直径降低 LRS 电阻，但与电导率强相关，不宜同时任意拟合。
    seed_diameter_m: float = 0.5e-9
    # 示例最细桥接细丝直径；SET 闭合前 w 保持给定初值，闭合后扩展。
    # 用于定义合法直径下界，不代表论文独立提取过该数值。
    atomic_step_m: float = 0.25e-9
    # 表 I 的 a：空位间距/原子跳跃步长，同时作为局部场分母的最小尺度。

    # ---------- 场辅助离子动力学：表 I ----------
    attempt_hz: float = 1e13
    # f，氧原子振动频率 Hz；速率的共同前因子。默认固定，以减轻与能垒的退化。
    charge_number: float = 2.0
    # Z=2，氧离子电荷数的绝对值；不携带电压符号，方向由 set_polarity 指定。
    generation_ev: float = 1.2
    # E_a，氧空位生成能垒 eV；较高时 SET 变慢/阈值升高，可选少量校准。
    hopping_ev: float = 1.0
    # E_h，氧离子迁移能垒 eV；较高时 RESET 后期变慢。
    release_ev: float = 1.2
    # E_i，电极释放氧离子的能垒 eV；常控制 DC RESET 初期。
    recombination_ev: float = 0.8
    # Delta E_r，复合弛豫能 eV；高场快速 RESET 时可能成为瓶颈。
    generation_alpha_m: float = 0.75e-9
    # alpha_a，场致生成能垒降低的有效长度 m；增大可加速 SET，非几何跳跃距离 a。
    hopping_alpha_m: float = 0.75e-9
    # alpha_h，场致迁移能垒降低的有效长度 m；增大可加速场驱动迁移。
    release_gamma: float = 1.5
    # gamma，无量纲电压增强系数；释放能垒降低为 gamma*Z*|V_internal| eV。
    radial_shell_m: float = 0.5e-9
    # Delta w，式 (18) 的有效径向生成区域尺度 m；增大使闭合后的扩径加快。

    # ---------- 电子传导：表 I + 式 (19) ----------
    conductivity_s_m: float = 1e6
    # sigma0，S/m；金属型传导前因子，增大降低 LRS 电阻。
    conduction_ev: float = 0.1
    # E_AC，eV；式 (5) 指数是正号。需结合多温度数据识别，默认固定。
    hopping_density_a_m2: float = 1e13
    # J0=10 microampere/nm^2=1e13 A/m^2，式 (19) 给出的面积归一化电流。
    # 与有效面积退化；拟合 J0 时应固定几何。
    hopping_length_m: float = 0.4e-9
    # x_T，指数衰减长度 m；越大，给定间隙电流越大。
    # 表 I 印为 nm^-1，但式 (19) 为 exp(-x/x_T)，本版按长度解释并明确记录。
    hopping_voltage_v: float = 0.4
    # V_T，非线性电压尺度 V；越大，同电压 hopping 电流越小。
    # 表 I 印为 V^-1，本版依式 (19) 的 V/V_T 按 V 解释。

    # ---------- 热与电路 ----------
    ambient_k: float = 300.0
    # T0，环境温度 K；示例取 300 K，原文表 I 给 k_B*T0 约 0.026 eV。
    thermal_k_w: float = 5e5
    # 表 I 的 R_th，K/W（表中 k/W），有效热阻；增大产生更强焦耳热与热激活。
    # 只有 DC 数据时容易与能垒退化，默认不和全部能垒一起释放。
    series_ohm: float = 40.0
    # 总外部接触串联电阻。图 7 两端各标 R_c，本版按每端 20 ohm 合计 40 ohm。
    # 与其他文献采用“总 R_c=20 ohm”的定义不同，拟合配置应明确使用总值。
    parallel_ohm: float | None = None
    # 可选并联漏电阻 R_p；None 表示开路，原文未提供表 I 数值。
    # 其功耗不计入局部 CF 温升；背景 hopping 属于开关元件，计入温升。
    set_polarity: int = 1
    # +1：正偏 SET、负偏 RESET；-1：反向。改变接线约定，不改变势垒符号。
    temperature_limit_k: float = 3000.0
    # 数值/适用性保护，非论文参数。解超过该温度时报错，不以截温掩盖失效。

    def __post_init__(self):
        """拒绝非物理输入；0 热阻/串联电阻分别用于关闭自热/接触压降。"""
        nonnegative = {"thermal_k_w", "series_ohm", "conduction_ev"}
        for field in fields(self):
            name, value = field.name, getattr(self, field.name)
            if name == "parallel_ohm" and value is None:
                continue
            if name == "set_polarity":
                if value not in (-1, 1):
                    raise ValueError("set_polarity 必须为 +1 或 -1")
                continue
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{name} 必须为有限数值")
            if value < 0 or (value == 0 and name not in nonnegative):
                raise ValueError(f"{name} 必须为正数（热阻、串联电阻、传导能允许为零）")
        if self.seed_diameter_m > self.filament_max_m:
            raise ValueError("seed_diameter_m 不能大于 filament_max_m")
        if self.atomic_step_m > self.thickness_m:
            raise ValueError("atomic_step_m 不能超过氧化层厚度")
        if self.temperature_limit_k <= self.ambient_k:
            raise ValueError("temperature_limit_k 必须高于环境温度")


@dataclass(frozen=True)
class State:
    """几何状态，全部单位 m。x0 是断裂区域记忆，必须不小于当前 x。

    x_m：实际绝缘间隙；0 表示细丝已接通。
    w_m：新生/有效导电细丝的直径；间隙未闭合时不发生径向扩展。
    x0_m：断裂区域长度；RESET 扩大间隙时同步增长，SET 时保持。
    首版不模拟原子级成核，使用非零 seed_diameter_m 避免零面积奇点。
    """

    x_m: float = 1e-9
    w_m: float = 0.5e-9
    x0_m: float = 1e-9

    def validate(self, p: Parameters):
        if not all(math.isfinite(v) for v in (self.x_m, self.w_m, self.x0_m)):
            raise ValueError("状态必须为有限数值")
        if not 0 <= self.x_m <= self.x0_m <= p.thickness_m:
            raise ValueError("状态须满足 0 <= x_m <= x0_m <= thickness_m")
        if not p.seed_diameter_m <= self.w_m <= p.filament_max_m:
            raise ValueError("细丝直径必须位于 seed_diameter_m 与 filament_max_m 之间")


@dataclass(frozen=True)
class OperatingPoint:
    """电热解；所有电流/电压保留外部极性，温度及功率为非负实数。"""

    requested_v: float
    terminal_v: float  # 理想限流后、包含接触电阻的实际两端电压。
    internal_v: float  # 扣除接触电阻压降，作用于开关元件与并联漏电支路。
    current_a: float  # 外部总电流，包含可选 R_p 漏电。
    switching_a: float  # 开关元件电流，不包含 R_p。
    gap_v: float  # 主细丝支路的绝缘间隙压降；闭合时为零。
    temperature_k: float
    limited: bool


@dataclass(frozen=True)
class Rates:
    """状态导数 m/s 及 RESET 各候选速率；正负方向只体现在 dx、dw 中。"""

    dx_m_s: float
    dw_m_s: float
    reset_hopping_m_s: float
    reset_release_m_s: float
    reset_recombination_m_s: float
    bottleneck: str


@dataclass(frozen=True)
class SimulationResult:
    """按输入采样时刻保存的有序结果；采样点之前的历史已累积到 state。"""

    time_s: list[float]
    points: list[OperatingPoint]
    states: list[State]
    accepted_steps: int


@dataclass(frozen=True)
class TransientOptions:
    """实测电压驱动的脉冲扩展，不改变原 DC 接口及已保存的 DC 参数。

    thermal_time_s: 热弛豫时间 tau=Rth*Cth，单位 s；零表示原文准静态
    热近似。正值时求 dT/dt=(T0+Pcond*Rth-T)/tau，断电后连续冷却，
    而非每个采样点立即回到环境温度。Rth 和 tau 不应同时无约束释放。

    pad_capacitance_f: 跨实测器件端子的有效并联电容，F；源/返回端测得的
    电流包含 Icap=Cpad*dVterminal/dt。它概括电极/夹具的可见电容，不应
    直接解释成氧化层本征电容，也不将充电功率误计为细丝焦耳热。
    本接口由已测的器件差分电压强制驱动，不求发生器输出电路；若只有
    源设定电压，必须先扩展源阻抗/RC 电压状态，不能直接套用该接口。

    所有脉冲均不限流；电流限幅是仪器/外部电路行为，不是本征脉冲模型。
    """

    thermal_time_s: float = 0.0
    pad_capacitance_f: float = 0.0

    def __post_init__(self):
        for value in (self.thermal_time_s, self.pad_capacitance_f):
            if not math.isfinite(value) or value < 0:
                raise ValueError("热时间常数和端子电容须为非负有限数")


@dataclass(frozen=True)
class TransientResult:
    """电流单位 A，温度 K，时间 s；导电和电容电流分别保留以核查边沿。"""

    time_s: list[float]
    points: list[OperatingPoint]
    states: list[State]
    capacitive_a: list[float]
    total_a: list[float]
    accepted_steps: int


def conductivity(temperature_k: float, p: Parameters) -> float:
    """式 (5)，sigma0*exp(+E_AC/kBT)，单位 S/m。正号体现类金属温度系数。"""
    if not math.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("温度必须为有限正数")
    return p.conductivity_s_m * math.exp(p.conduction_ev / (KB_EV * temperature_k))


def hopping_density(voltage_v: float, gap_m: float, p: Parameters) -> float:
    """式 (19) 的面积归一化版本，A/m^2；固定几何下电流关于电压为奇函数。

    仅用于独立传导检查；实际串联电路以其反函数求解，避免大 sinh 溢出。
    """
    if not math.isfinite(voltage_v) or not math.isfinite(gap_m) or gap_m < 0:
        raise ValueError("电压须有限，间隙须为非负有限数")
    u = abs(voltage_v) / p.hopping_voltage_v
    if u == 0:
        return 0.0
    log_sinh = (
        math.log(math.sinh(u))
        if u < 20
        else u - math.log(2) + math.log1p(-math.exp(-2 * u))
    )
    log_j = math.log(p.hopping_density_a_m2) - gap_m / p.hopping_length_m + log_sinh
    if log_j > 700:
        raise ModelError("hopping 电流超出浮点范围；请检查电压/参数单位")
    return math.copysign(math.exp(log_j), voltage_v)


def _bisect(fn, low: float, high: float, *, voltage_tol: float = 1e-11) -> float:
    """有括区间的电路标量求根；残差单位 V，失败时不返回伪解。"""
    fl, fh = fn(low), fn(high)
    if not math.isfinite(fl) or not math.isfinite(fh):
        raise ModelError("电路残差不是有限数")
    if abs(fl) <= voltage_tol:
        return low
    if abs(fh) <= voltage_tol:
        return high
    if fl * fh > 0:
        raise ModelError("电路求根无法建立变号区间")
    for _ in range(100):
        mid = (low + high) / 2
        fm = fn(mid)
        if not math.isfinite(fm):
            raise ModelError("电路求解出现非有限残差")
        if abs(fm) <= voltage_tol or high - low <= max(abs(mid) * 1e-12, 1e-300):
            return mid
        if (fl < 0) == (fm < 0):
            low, fl = mid, fm
        else:
            high = mid
    raise ModelError("电路求根超过迭代次数")


def _geometry(s: State, p: Parameters):
    """图 5 的首版几何闭合：金属电阻的长度/面积因子、两条 hopping 面积。

    旧细丝区长度 L-x0、面积 A0；新细丝金属区长度 x0-x、面积 A。
    背景 hopping 面积 A0-A，长度 x0。背景路径忽略旧细丝串联压降。
    因而金属电阻 R_cf = [(L-x0)/A0 + (x0-x)/A]/sigma(T)。
    """
    a0 = math.pi * p.filament_max_m**2 / 4
    area = math.pi * s.w_m**2 / 4
    factor = (p.thickness_m - s.x0_m) / a0 + (s.x0_m - s.x_m) / area
    return factor, area, max(0.0, a0 - area)


def _gap_voltage(current_a: float, log_scale: float, p: Parameters) -> float:
    """式 (19) 反函数：V_gap=V_T*asinh(I/[A*J0*exp(-x/x_T)])。

    在对数域处理极小间隙电流尺度，避免除法上溢；函数保留电流符号。
    """
    if current_a == 0:
        return 0.0
    log_ratio = math.log(abs(current_a)) - log_scale
    value = (
        log_ratio + math.log(2) if log_ratio > 350 else math.asinh(math.exp(log_ratio))
    )
    return math.copysign(p.hopping_voltage_v * value, current_a)


def operating_point(
    voltage_v: float,
    state: State,
    params: Parameters | None = None,
    compliance_a: float | None = None,
    *,
    compliance_mode: str = "both",
) -> OperatingPoint:
    """固定几何下求电流、电压和自热，并可施加理想电流限幅。

    compliance_mode="both" 保持原有的正负对称限流；"set" 仅在 SET
    极性限流（包含 SET 极性的返回扫描），RESET 不限流。SET 极性由
    set_polarity 决定，因此反接器件也不必手工改限流电流的符号。

    求解变量为非负开关电流 J。设内部电压 U、总串联电阻 Rs、漏电阻 Rp：
      U = (|V_requested|-Rs*J)/(1+Rs/Rp)
      T = T0 + J*U*Rth
      I_main = J - I_background(U)
      U = I_main*R_cf(T) + V_gap(I_main)
    上式将电路和自热化为一个标量零点；并非先求冷态电流再事后加温。

    若达到限流，则固定总电流 Icc，改以 U 为未知量重新求解；因此限流
    会降低实际端电压、间隙电场和温升，而非只裁剪绘图输出电流。
    """
    p = params or Parameters()
    state.validate(p)
    if not math.isfinite(voltage_v):
        raise ValueError("电压必须有限")
    if compliance_a is not None and (
        not math.isfinite(compliance_a) or compliance_a <= 0
    ):
        raise ValueError("compliance_a 必须为有限正数")
    if compliance_mode not in ("both", "set"):
        raise ValueError("compliance_mode 必须为 both 或 set")
    # SET 极性由器件接线约定决定；即使同一条回线也不可误裁剪 RESET 电流。
    # 在 SET 极性返回扫描时仍保持仪器限流，零偏压不需要限流。
    if compliance_mode == "set" and p.set_polarity * voltage_v <= 0:
        compliance_a = None
    if voltage_v == 0:
        return OperatingPoint(0, 0, 0, 0, 0, 0, p.ambient_k, False)
    sign = math.copysign(1, voltage_v)
    target = abs(voltage_v)
    metal_factor, area, background_area = _geometry(state, p)
    leak = 0.0 if p.parallel_ohm is None else 1 / p.parallel_ohm
    log_scale = math.log(area * p.hopping_density_a_m2) - state.x_m / p.hopping_length_m

    def background(u):
        return (
            background_area * hopping_density(u, state.x0_m, p)
            if background_area
            else 0.0
        )

    def quantities(j, u):
        temperature = p.ambient_k + j * u * p.thermal_k_w
        main = j - background(u)
        # 亚原子间隙的连续接触插值：避免 x=0 到 x>0 的有限电阻跳变。
        # 该插值只作用于 x<a，必须与完整 hopping 式的适用区区分。
        gap = (
            min(1.0, state.x_m / p.atomic_step_m) * _gap_voltage(main, log_scale, p)
            if state.x_m > 0
            else 0.0
        )
        residual = main * metal_factor / conductivity(temperature, p) + gap - u
        return residual, temperature, gap

    def residual(j):
        u = max(0.0, (target - p.series_ohm * j) / (1 + p.series_ohm * leak))
        return quantities(j, u)[0]

    # 使用冷态金属上界；全绝缘厚度且 metal_factor=0 时用 hopping 直接上界。
    if p.series_ohm:
        upper = target / p.series_ohm
    elif metal_factor > 0:
        upper = target * conductivity(p.ambient_k, p) / metal_factor + background(
            target
        )
    else:
        upper = area * hopping_density(target, state.x_m, p) + background(target)
    j = _bisect(residual, 0.0, upper)
    u = max(0.0, (target - p.series_ohm * j) / (1 + p.series_ohm * leak))
    total = j + u * leak
    limited = compliance_a is not None and total > compliance_a
    if limited:

        def limited_residual(v):
            return quantities(max(0.0, compliance_a - v * leak), v)[0]

        # 非负开关支路电流要求 U<=Icc*Rp。
        upper_u = min(u, compliance_a / leak) if leak else u
        u = _bisect(limited_residual, 0.0, upper_u)
        total = compliance_a
        j = max(0.0, total - u * leak)
    _, temperature, gap = quantities(j, u)
    if temperature > p.temperature_limit_k:
        raise ModelError(f"局部温度 {temperature:.1f} K 超过模型保护上限")
    if gap < -1e-8:
        raise ModelError("得到不符合无源导电假设的间隙压降")
    terminal = u + p.series_ohm * total
    return OperatingPoint(
        voltage_v,
        sign * terminal,
        sign * u,
        sign * total,
        sign * j,
        sign * max(0.0, gap),
        temperature,
        bool(limited),
    )


def switching_rates(
    voltage_v: float,
    state: State,
    params: Parameters | None = None,
    point: OperatingPoint | None = None,
) -> Rates:
    """式 (11)--(13)、(15)、(18)：连续的、场/温度辅助的几何演化速率。

    RESET（取正的速率大小，再令 dx/dt>0）：
      v_h = a*f*exp(-E_h/kBT)*sinh(alpha_h*Z*E/kBT)       (11)
      v_i = a*f*exp[-(E_i-gamma*Z*|U|)/kBT]             (12)
      v_r = a*f*exp(-Delta E_r/kBT)                     (13)
      dx/dt = min(v_h,v_i,v_r)，即串联物理过程的最慢步骤控制。
    SET：
      x>0 时 dx/dt = -a*f*exp[-(E_a-alpha_a*Z*E)/kBT]   (15)
      x=0 时 dw/dt = (Delta w+Delta w^2/(2*w))*f
                         *exp[-(E_a-alpha_a*Z*E)/kBT]   (18)

    正偏/负偏仅表示接线约定，能垒降低使用相应方向的电压/场大小。
    在恰好零电压时冻结状态：首版不包含保持/零场弛豫模型。
    速率的 Boltzmann 因子最大为 1，是负有效能垒下的尝试频率饱和假设。
    """
    p = params or Parameters()
    state.validate(p)
    op = point or operating_point(voltage_v, state, p)
    v = p.set_polarity * op.internal_v
    if v == 0:
        return Rates(0, 0, 0, 0, 0, "zero_bias")
    kt = KB_EV * op.temperature_k
    # x<a 时 gap_v 已乘 x/a，因此 gap_v/x 在 x->0 时仍有限；不能再将
    # 分母换成 a，否则会重复缩小场，使 RESET 的微小间隙无法长大。
    field = (
        abs(op.gap_v) / state.x_m
        if state.x_m > 0
        else abs(op.internal_v) / max(state.x0_m, p.atomic_step_m)
    )
    # 无场近闭合 RESET 使用原子尺度下的内部电压，启动首个断裂步骤。
    if state.x_m == 0:
        field = abs(op.internal_v) / p.atomic_step_m if v < 0 else field
    log_h = p.hopping_alpha_m * p.charge_number * field / kt
    log_sinh = (
        math.log(math.sinh(log_h))
        if 0 < log_h < 20
        else (log_h - math.log(2) if log_h else -math.inf)
    )
    hop = (
        p.atomic_step_m
        * p.attempt_hz
        * math.exp(min(0.0, -p.hopping_ev / kt + log_sinh))
    )
    release = (
        p.atomic_step_m
        * p.attempt_hz
        * math.exp(
            min(
                0.0,
                -(p.release_ev - p.release_gamma * p.charge_number * abs(op.internal_v))
                / kt,
            )
        )
    )
    recombine = p.atomic_step_m * p.attempt_hz * math.exp(-p.recombination_ev / kt)
    if v < 0:
        candidates = {"hopping": hop, "release": release, "recombination": recombine}
        limiting = min(candidates, key=candidates.get)
        dx = candidates[limiting] if state.x_m < p.thickness_m else 0.0
        return Rates(dx, 0, hop, release, recombine, limiting)
    activation = p.attempt_hz * math.exp(
        min(
            0.0,
            -(p.generation_ev - p.generation_alpha_m * p.charge_number * field) / kt,
        )
    )
    if state.x_m > 0:
        return Rates(
            -p.atomic_step_m * activation, 0, hop, release, recombine, "set_gap"
        )
    dw = (
        (p.radial_shell_m + p.radial_shell_m**2 / (2 * state.w_m)) * activation
        if state.w_m < p.filament_max_m
        else 0.0
    )
    return Rates(0, dw, hop, release, recombine, "set_width")


def simulate_dc(
    time_s: Sequence[float],
    voltage_v: Sequence[float],
    initial_state: State | None = None,
    params: Parameters | None = None,
    *,
    compliance_a: float | None = None,
    rtol: float = 1e-3,
    atol_m: float = 1e-13,
    max_step_s: float | None = None,
    max_steps: int = 200000,
    compliance_mode: str = "both",
) -> SimulationResult:
    """按采样顺序积分 DC 扫描，采样点之间的施加电压做线性插值。

    时间必须严格递增；绝不按电压排序，否则往返扫描的历史会被破坏。
    输入时间只用其差值。首个输出就是给定初态，尚未施加额外隐含停留。
    如需电压阶跃/恒压停留，应明确给出其时间采样；不模拟理想不连续阶跃。

    每步用 Euler 预测与 Heun 校正之差控制误差，并限制单步几何变化，
    到达边界后投影并重新计算下一阶段速率。x 小于 atol_m 时接通；此
    数值接触容差不是物理拟合参数，可通过收紧 atol_m 验证结果收敛。
    max_steps 包含拒绝步数，防止不合理参数让拟合永久卡住。
    compliance_mode="set" 时，每个积分子步按 SET 极性施加 compliance_a，
    极性转入 RESET 后解除限流，不能用对称限流近似仅 SET 限流的实验。
    """
    p = params or Parameters()
    state = initial_state or State()
    state.validate(p)
    ts, vs = [float(t) for t in time_s], [float(v) for v in voltage_v]
    if len(ts) != len(vs) or not ts:
        raise ValueError("时间/电压必须等长且非空")
    if not all(math.isfinite(v) for v in ts + vs) or any(
        b <= a for a, b in pairwise(ts)
    ):
        raise ValueError("时间/电压须有限，时间须严格递增")
    if (
        not math.isfinite(rtol)
        or rtol <= 0
        or not math.isfinite(atol_m)
        or not 0 < atol_m < p.atomic_step_m
    ):
        raise ValueError("rtol 必须为正，atol_m 必须为小于原子步长的正数")
    if max_step_s is not None and (not math.isfinite(max_step_s) or max_step_s <= 0):
        raise ValueError("max_step_s 必须为有限正数")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps 必须为正整数")

    def project(x, w, old):
        x = min(p.thickness_m, max(0.0, x))
        # 只在 SET 向零闭合时吸附到接触边界；RESET 的微小正增量须累积。
        if 0 < x < atol_m and x < old.x_m:
            x = 0.0
        w = min(p.filament_max_m, max(p.seed_diameter_m, w))
        return State(x, w, max(old.x0_m, x))

    def derivative(v, s):
        point = operating_point(v, s, p, compliance_a, compliance_mode=compliance_mode)
        return switching_rates(v, s, p, point)

    points = [
        operating_point(vs[0], state, p, compliance_a, compliance_mode=compliance_mode)
    ]
    states = [state]
    accepted = attempts = 0
    next_h = math.inf
    for index in range(1, len(ts)):
        start, end = ts[index - 1], ts[index]
        duration = end - start
        t = 0.0

        def voltage(local_t, index=index, duration=duration):
            return vs[index - 1] + (vs[index] - vs[index - 1]) * (local_t / duration)

        # 将零偏压交点作为分段边界，避免跨过极性变化而混合不同状态机制。
        boundaries = [duration]
        if vs[index - 1] * vs[index] < 0:
            boundaries.insert(
                0, duration * (-vs[index - 1]) / (vs[index] - vs[index - 1])
            )
        for boundary in boundaries:
            while t < boundary:
                attempts += 1
                if attempts > max_steps:
                    raise ModelError("状态积分超过 max_steps；检查时间尺度、参数或容差")
                remaining = boundary - t
                h = min(remaining, duration / 8, next_h, max_step_s or math.inf)
                v0 = voltage(t)
                r0 = derivative(v0, state)
                # 限制几何增量，防止在一个时间步内掠过窄的开关区。
                speed = max(abs(r0.dx_m_s), abs(r0.dw_m_s))
                if speed:
                    h = min(h, 0.1 * p.atomic_step_m / speed)
                if h <= 0 or t + h == t:
                    raise ModelError("积分步长低于浮点分辨率")
                euler = project(
                    state.x_m + h * r0.dx_m_s, state.w_m + h * r0.dw_m_s, state
                )
                r1 = derivative(voltage(t + h), euler)
                # 闭合事件的步内不混入下一阶段扩径；下一步会从新阶段重新计算。
                dx1 = (
                    r1.dx_m_s
                    if (r0.bottleneck == r1.bottleneck or r0.dx_m_s * r1.dx_m_s >= 0)
                    else 0.0
                )
                dw1 = r1.dw_m_s if state.x_m == 0 else 0.0
                corrected = project(
                    state.x_m + 0.5 * h * (r0.dx_m_s + dx1),
                    state.w_m + 0.5 * h * (r0.dw_m_s + dw1),
                    state,
                )
                scale_x = atol_m + rtol * max(abs(state.x_m), abs(corrected.x_m))
                scale_w = atol_m + rtol * max(abs(state.w_m), abs(corrected.w_m))
                error = max(
                    abs(corrected.x_m - euler.x_m) / scale_x,
                    abs(corrected.w_m - euler.w_m) / scale_w,
                )
                if error <= 1:
                    state = corrected
                    t = min(boundary, t + h)
                    if boundary - t <= max(duration * 1e-14, math.ulp(boundary)):
                        t = boundary
                    accepted += 1
                next_h = h * (
                    2 if error == 0 else min(2.0, max(0.1, 0.85 / math.sqrt(error)))
                )
        states.append(state)
        points.append(
            operating_point(
                vs[index], state, p, compliance_a, compliance_mode=compliance_mode
            )
        )
    return SimulationResult(ts, points, states, accepted)


def _fixed_temperature_point(
    voltage_v: float, state: State, p: Parameters, temperature_k: float
) -> OperatingPoint:
    """给定温度的瞬时电子传导，与 DC 电路拓扑相同，供热 ODE 内部调用。

    将原 operating_point 中温度固定，以带括区间的 Newton 法解主电流。
    解析导数包括背景 hopping 对内部电压的导数和串联接触电阻反馈；
    因而不是把总端电压直接施加到间隙。符号、漏电和面积均沿用 DC。
    不在这里施加自热平衡，避免把动态温度与准静态升温重复相加。
    """
    if voltage_v == 0:
        return OperatingPoint(0, 0, 0, 0, 0, 0, temperature_k, False)
    sign = math.copysign(1.0, voltage_v)
    target = abs(voltage_v)
    factor, area, bg_area = _geometry(state, p)
    resistance = factor / conductivity(temperature_k, p)
    leak = 0 if p.parallel_ohm is None else 1 / p.parallel_ohm
    scale = area * p.hopping_density_a_m2 * math.exp(-state.x_m / p.hopping_length_m)
    bg_scale = (
        bg_area * p.hopping_density_a_m2 * math.exp(-state.x0_m / p.hopping_length_m)
    )
    vt = p.hopping_voltage_v * min(1.0, state.x_m / p.atomic_step_m)
    divider = 1 + p.series_ohm * leak
    du = -p.series_ohm / divider
    if p.series_ohm:
        upper = target / p.series_ohm
    elif resistance:
        upper = target / resistance + bg_scale * math.sinh(target / p.hopping_voltage_v)
    else:
        upper = scale * math.sinh(target / vt) + bg_scale * math.sinh(
            target / p.hopping_voltage_v
        )
    g = (
        1 / (resistance + vt / max(scale, 1e-300))
        if resistance + vt / max(scale, 1e-300)
        else 1e300
    )
    g += bg_scale / p.hopping_voltage_v
    j = min(upper, target * g / (1 + p.series_ohm * (g + leak)))
    lower = 0.0
    for _ in range(70):
        u = max(0.0, (target - p.series_ohm * j) / divider)
        z = u / p.hopping_voltage_v
        bg = bg_scale * math.sinh(z)
        main = j - bg
        gap = vt * math.asinh(main / max(scale, 1e-300)) if vt else 0.0
        residual = resistance * main + gap - u
        # 在极小间隙、极低电流时，电流括区间可能先达到浮点相对精度，
        # 此时继续要求电压残差绝对小于 1e-10 会产生虚假的“不收敛”。
        # 与 DC 二分求根相同，同时允许电流区间的相对收敛。
        if abs(residual) < 1e-10 or upper - lower <= max(abs(j) * 1e-12, 1e-30):
            total = j + u * leak
            return OperatingPoint(
                voltage_v,
                voltage_v,
                sign * u,
                sign * total,
                sign * j,
                sign * max(0.0, gap),
                temperature_k,
                False,
            )
        if residual < 0:
            lower = j
        else:
            upper = j
        derivative = (resistance + vt / math.hypot(main, max(scale, 1e-300))) * (
            1 - bg_scale * math.cosh(z) / p.hopping_voltage_v * du
        ) - du
        candidate = j - residual / derivative if derivative > 0 else math.inf
        j = candidate if lower < candidate < upper else (lower + upper) / 2

    # 拐点附近 Newton 可能长期停留在括区间一端；退回单调电路的二分法，
    # 而不是把纯数值失败当成器件不支持该参数。仍使用同一物理残差。
    def fallback_residual(current):
        core_v = max(0.0, (target - p.series_ohm * current) / divider)
        main_i = current - bg_scale * math.sinh(core_v / p.hopping_voltage_v)
        barrier_v = vt * math.asinh(main_i / max(scale, 1e-300)) if vt else 0.0
        return main_i * resistance + barrier_v - core_v

    j = _bisect(fallback_residual, lower, upper)
    u = max(0.0, (target - p.series_ohm * j) / divider)
    main = j - bg_scale * math.sinh(u / p.hopping_voltage_v)
    gap = vt * math.asinh(main / max(scale, 1e-300)) if vt else 0.0
    return OperatingPoint(
        voltage_v,
        voltage_v,
        sign * u,
        sign * (j + u * leak),
        sign * j,
        sign * max(0.0, gap),
        temperature_k,
        False,
    )


def simulate_transient(
    time_s: Sequence[float],
    voltage_v: Sequence[float],
    initial_state: State | None = None,
    params: Parameters | None = None,
    options: TransientOptions | None = None,
    *,
    initial_temperature_k: float | None = None,
    rtol: float = 1e-3,
    atol_m: float = 1e-13,
    max_step_s: float = 5e-9,
    max_steps: int = 500000,
) -> TransientResult:
    """将连续 SET–READ–RESET–READ 电压波形转为连续状态及观测电流。

    输入须为实际两端电压、严格递增时间；原始数据的时间重建、示波器
    50 ohm 换算和电流极性均在数据层显式完成，不能在器件方程中猜测。
    输入点间线性插值、遇采样节点截步，避免跨过短脉冲。首状态位于首个
    采样时刻，不凭空补出未测的预触发波形。整个记录不按 cycle 重置。

    几何用投影 Heun 积分、局部误差控制和原子尺度步长上限；温度采用
    冻结功率下的解析指数更新，再以首尾功率平均校正，避免 tau<<dt 时
    显式热积分不稳定。误差检查同时包含 x、w 和温度。x0 继续作为断裂
    区域记忆，不引入未经证实的弛豫或随机状态。

    输出总电流=串联接触后的导电/漏电支路电流+Cpad*dVterminal/dt。
    在输入节点处用相邻区间斜率的加权中心值；用于仪器有限窗口拟合时
    应在更细时间网格模拟，再由观测层做真实积分窗口平均。
    """
    import numpy as np

    p, opts = params or Parameters(), options or TransientOptions()
    state = initial_state or State()
    state.validate(p)
    ts, vs = list(map(float, time_s)), list(map(float, voltage_v))
    if len(ts) != len(vs) or len(ts) < 2 or not all(math.isfinite(v) for v in ts + vs):
        raise ValueError("瞬态输入须等长、至少两点且有限")
    if any(b <= a for a, b in pairwise(ts)):
        raise ValueError("瞬态时间须严格递增；不能排序或自动丢弃重复时间戳")
    if not math.isfinite(rtol) or rtol <= 0 or not 0 < atol_m < p.atomic_step_m:
        raise ValueError("瞬态积分容差非法")
    if not math.isfinite(max_step_s) or max_step_s <= 0 or max_steps <= 0:
        raise ValueError("瞬态步长和步数上限须为正")
    temperature = (
        p.ambient_k if initial_temperature_k is None else initial_temperature_k
    )
    if not math.isfinite(temperature) or not 0 < temperature <= p.temperature_limit_k:
        raise ValueError("初温非法")

    def point(v, s, temp):
        return (
            _fixed_temperature_point(v, s, p, temp)
            if opts.thermal_time_s
            else operating_point(v, s, p)
        )

    def project(x, w, old):
        x = min(p.thickness_m, max(0.0, x))
        if 0 < x < atol_m and x < old.x_m:
            x = 0.0
        return State(
            x, min(p.filament_max_m, max(p.seed_diameter_m, w)), max(old.x0_m, x)
        )

    def thermal(old, power, h):
        equilibrium = p.ambient_k + max(0.0, power) * p.thermal_k_w
        if not opts.thermal_time_s:
            return equilibrium
        return equilibrium + (old - equilibrium) * math.exp(-h / opts.thermal_time_s)

    points, states = [point(vs[0], state, temperature)], [state]
    accepted = attempts = 0
    next_h = max_step_s
    for k in range(1, len(ts)):
        duration = ts[k] - ts[k - 1]
        local = 0.0
        while local < duration:
            attempts += 1
            if attempts > max_steps:
                raise ModelError("瞬态积分超过 max_steps")
            h = min(duration - local, next_h, max_step_s)
            v0 = vs[k - 1] + (vs[k] - vs[k - 1]) * local / duration
            op0 = point(v0, state, temperature)
            r0 = switching_rates(v0, state, p, op0)
            speed = max(abs(r0.dx_m_s), abs(r0.dw_m_s))
            if speed:
                h = min(h, 0.1 * p.atomic_step_m / speed)
            if h <= 0 or local + h == local:
                raise ModelError("瞬态步长低于浮点分辨率")
            v1 = vs[k - 1] + (vs[k] - vs[k - 1]) * (local + h) / duration
            predicted = project(
                state.x_m + h * r0.dx_m_s, state.w_m + h * r0.dw_m_s, state
            )
            power0 = op0.switching_a * op0.internal_v
            temp_predict = thermal(temperature, power0, h)
            if temp_predict > p.temperature_limit_k:
                raise ModelError("瞬态温度超过保护上限")
            op1 = point(v1, predicted, temp_predict)
            r1 = switching_rates(v1, predicted, p, op1)
            dx1 = r1.dx_m_s if r0.dx_m_s * r1.dx_m_s >= 0 else 0.0
            dw1 = r1.dw_m_s if state.x_m == 0 else 0.0
            corrected = project(
                state.x_m + 0.5 * h * (r0.dx_m_s + dx1),
                state.w_m + 0.5 * h * (r0.dw_m_s + dw1),
                state,
            )
            temp_correct = thermal(
                temperature, 0.5 * (power0 + op1.switching_a * op1.internal_v), h
            )
            error = max(
                abs(corrected.x_m - predicted.x_m)
                / (atol_m + rtol * max(state.x_m, corrected.x_m)),
                abs(corrected.w_m - predicted.w_m)
                / (atol_m + rtol * max(state.w_m, corrected.w_m)),
            )
            if opts.thermal_time_s:
                error = max(
                    error,
                    abs(temp_correct - temp_predict)
                    / (0.02 + rtol * max(temperature, temp_correct)),
                )
            if error <= 1:
                state, temperature = corrected, temp_correct
                local = min(duration, local + h)
                if duration - local < max(duration * 1e-13, math.ulp(duration)):
                    local = duration
                accepted += 1
            next_h = h * (
                2 if error == 0 else min(2.0, max(0.1, 0.85 / math.sqrt(error)))
            )
        output = point(vs[k], state, temperature)
        temperature = output.temperature_k
        if temperature > p.temperature_limit_k:
            raise ModelError("瞬态温度超过保护上限")
        points.append(output)
        states.append(state)
    slopes = np.gradient(np.asarray(vs), np.asarray(ts))
    cap = [opts.pad_capacitance_f * float(slope) for slope in slopes]
    return TransientResult(
        ts,
        points,
        states,
        cap,
        [op.current_a + ic for op, ic in zip(points, cap, strict=True)],
        accepted,
    )
