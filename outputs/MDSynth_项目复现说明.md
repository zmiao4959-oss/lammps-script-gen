# MDSynth 项目复现说明

本文档说明 `lammps-script-gen` 项目在做什么、如何组织代码、每个大模块怎样实现，以及如何从零复现一个功能等价的最小版本。目标读者是另一个 AI 或开发者：看完后应能重建该项目的核心架构与主要行为。

## 1. 项目定位

MDSynth 是一个面向 LAMMPS 的 AI-native 分子动力学实验编译器。它的目标不是让大语言模型直接生成 LAMMPS 脚本，而是把自然语言研究需求转换为结构化中间表示，再由确定性规则生成脚本。

核心思想：

- LLM 只负责理解用户意图。
- 物理知识、材料知识、LAMMPS 命令模板都由代码确定性实现。
- 所有修复都作用在 MD Typed IR 上，而不是直接改生成后的脚本。
- 最终输出不仅包含 `in.main.lammps`，还包含 IR、验证报告、预检查报告、假设、修复历史和溯源文件。

端到端流程：

```text
Natural Language
  -> ScientificIntentSpec
  -> MDTypedIR
  -> Physics Validation
  -> Constrained Repair
  -> LAMMPS Compilation
  -> Sandbox Preflight
  -> Evidence Package
```

## 2. 最小技术栈

使用 Python 包实现。

推荐依赖：

```toml
requires-python = ">=3.10"
dependencies = [
  "openai>=1.0.0",
  "pydantic>=2.0.0",
  "pyyaml>=6.0",
  "structlog>=23.0.0",
  "numpy>=1.24.0",
]

[project.optional-dependencies]
test = [
  "pytest>=7.0.0",
  "pytest-cov>=4.0.0",
]
```

包名为 `mdsynth`。推荐目录：

```text
mdsynth/
  pipeline.py
  config.py
  intent/
    spec.py
    extractor.py
    taxonomy.py
  ir/
    md_ir.py
    planner.py
    task_families.py
    defaults.py
  knowledge/
    materials.py
    force_fields.py
    heuristics.py
    lammps_commands.py
  validator/
    diagnostic.py
    engine.py
    repair_hints.py
    rules/
      completeness.py
      boundary.py
      force_field.py
      ensemble.py
      numerics.py
      observables.py
  repair/
    permissions.py
    engine.py
    strategies.py
    classifier.py
  backend/
    compiler.py
    lowered_ir.py
    dependency_graph.py
    units.py
    emitters/
      header.py
      system.py
      forcefield.py
      neighbor.py
      protocol.py
      output.py
  sandbox/
    runner.py
    report.py
    diagnostics.py
  evidence/
    builder.py
    manifest.py
    report_templates.py
  llm/
    base.py
    openai_backend.py
    prompts/
```

## 3. 顶层流水线

顶层入口类是 `MDSynthPipeline`。它在初始化时装配所有模块：

```python
class MDSynthPipeline:
    def __init__(self, llm_backend=None, config=None):
        self.config = config or get_config()
        self.llm = llm_backend or MockLLMBackend()
        self.intent_extractor = ScientificIntentExtractor(self.llm)
        self.ir_planner = MDTypedIRPlanner(self.llm)
        self.validator = PhysicsValidator()
        self.repair = ConstrainedRepairEngine(
            max_auto_rounds=self.config.max_repair_rounds
        )
        self.compiler = LAMMPSBackendCompiler()
        self.sandbox = SandboxPreflightRunner(
            lmp_executable=self.config.lammps_executable
        )
        self.evidence_builder = EvidencePackageBuilder()
```

`run(user_request)` 的逻辑：

```python
def run(user_request: str) -> MDSynthOutput:
    intent_spec = intent_extractor.extract(user_request)
    intent_spec.original_user_text = user_request

    md_ir = ir_planner.plan(intent_spec)

    for round_num in range(config.max_repair_rounds):
        diagnostics = validator.validate(md_ir)
        errors = validator.get_errors(diagnostics)
        if not errors:
            break
        actions = repair.generate_actions(md_ir, errors)
        md_ir = repair.apply_actions(md_ir, actions)
    else:
        if validator.get_errors(validator.validate(md_ir)):
            raise RepairExhaustedError(...)

    lammps_script, lowered_ir = compiler.compile(md_ir)

    if config.sandbox_enabled:
        preflight_report = sandbox.run(lammps_script, md_ir)
    else:
        preflight_report = PreflightReport(passed=True, script_compiles=True)

    return evidence_builder.build(
        intent_spec=intent_spec,
        md_ir=md_ir,
        diagnostics=diagnostics,
        lammps_script=lammps_script,
        preflight_report=preflight_report,
        repair_history=repair.get_history(),
    )
```

`save(output, directory)` 调用 `EvidencePackageBuilder.write_to_directory()`，写出完整证据包。

## 4. ScientificIntentSpec：上游意图结构

该结构只表达用户“想研究什么”，不包含 LAMMPS 命令、势函数选择或具体模拟协议。

核心类型：

```python
class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class FieldStatus(str, Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    DEFAULTED = "defaulted"
    INFERRED = "inferred"

class MaterialClass(str, Enum):
    METAL = "metal"
    ALLOY = "alloy"
    CERAMIC = "ceramic"
    SEMICONDUCTOR = "semiconductor"

class Morphology(str, Enum):
    BULK = "bulk"
    NANOWIRE = "nanowire"
    THIN_FILM = "thin_film"
    NANOPARTICLE = "nanoparticle"
    UNSPECIFIED = "unspecified"
```

意图字段：

```python
@dataclass
class QuantitySpec:
    value: float | None = None
    unit: str | None = None
    status: FieldStatus = FieldStatus.UNKNOWN
    source: str = ""

@dataclass
class MaterialSpec:
    name: str | None = None
    material_class: MaterialClass | None = None
    morphology: Morphology = Morphology.UNSPECIFIED
    crystal_structure: str | None = None
    composition: dict[str, float] = field(default_factory=dict)
    structure_source: str | None = None

@dataclass
class TargetProperty:
    name: str
    priority: str = "primary"

@dataclass
class ScientificIntentSpec:
    task_type: str = ""
    task_description: str = ""
    extraction_confidence: Confidence = Confidence.MEDIUM
    material: MaterialSpec = field(default_factory=MaterialSpec)
    target_properties: list[TargetProperty] = field(default_factory=list)
    temperature: QuantitySpec = field(default_factory=QuantitySpec)
    pressure: QuantitySpec = field(default_factory=QuantitySpec)
    deformation_axis: str | None = None
    deformation_mode: str | None = None
    simulation_time: QuantitySpec = field(default_factory=QuantitySpec)
    speed_vs_accuracy: str = "balanced"
    desired_outputs: list[str] = field(default_factory=list)
    missing_blocking: list[str] = field(default_factory=list)
    missing_defaultable: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    original_user_text: str = ""
```

## 5. Intent Extraction 实现

模块：`mdsynth.intent.extractor`

职责：把中文或英文自然语言转为 `ScientificIntentSpec`。

实现步骤：

```python
def extract(user_request: str) -> ScientificIntentSpec:
    raw_json = _call_llm(user_request)
    validated = _validate_schema(raw_json)
    spec = _build_spec(validated)
    spec = _classify_missing(spec)
    spec = _flag_risks(spec)
    return spec
```

推荐行为：

1. `_call_llm()` 使用后端抽象 `LLMBackend.generate_structured()`，输入 prompt，输出 JSON。
2. `_validate_schema()` 校验 JSON 是否包含任务类型、材料、温度、压力、时长等字段；缺失字段保留为空，不直接报错。
3. `_build_spec()` 把 JSON 转为 dataclass，并把单位、状态、来源写入 `QuantitySpec`。
4. `_classify_missing()` 区分两类缺失：
   - `missing_blocking`：无法安全默认的字段，如材料。
   - `missing_defaultable`：可以使用默认值的字段，如压力、时长、应变速率。
5. `_flag_risks()` 根据任务和参数添加风险标签，如高应变速率、缺少温度点、压力未指定等。

LLM 后端需至少实现：

```python
class LLMBackend(ABC):
    def generate_structured(self, prompt: str, schema: dict | None = None) -> dict:
        ...
```

为了测试，必须提供 `MockLLMBackend`，通过关键词规则识别材料、任务类型和参数。例如：

- 输入含 `NPT` 或 `npt` -> `equilibration_npt`
- 输入含 `NVT` -> `equilibration_nvt`
- 输入含 `tension`、`拉伸` -> `uniaxial_tension`
- 输入含 `thermal expansion`、`热膨胀` -> `thermal_expansion`
- 输入含 `relaxation`、`弛豫` -> `structure_relaxation`

## 6. 支持任务族

模块：`mdsynth.ir.task_families`

定义 5 个 MVP 任务类型，每个任务族包含：

- `required_intent_fields`
- `default_stages`
- `required_observables`
- `required_analysis`
- `default_parameters`
- `risk_checks`

任务清单：

| task_type | 必需字段 | 默认阶段 | 观测量 | 分析 |
|---|---|---|---|---|
| `structure_relaxation` | material | minimization | temperature, potential_energy, pressure | final_energy, force_convergence |
| `equilibration_nvt` | material, temperature | minimization + NVT | temperature, energy, pressure, volume | temperature_stability, energy_conservation |
| `equilibration_npt` | material, temperature | minimization + NPT | temperature, potential_energy, pressure, volume, density | temperature/pressure/volume stability |
| `uniaxial_tension` | material, temperature, deformation_axis | minimization + NPT + deformation | temperature, stress_tensor, strain, pressure | stress_strain_curve, youngs_modulus_fit |
| `thermal_expansion` | material | temperature sweep with NPT | temperature, volume, density | thermal_expansion_coefficient_fit |

默认值示例：

- `equilibration_nvt`：默认平衡时长 100 ps，`tdamp = 100 * timestep`
- `equilibration_npt`：默认平衡时长 200 ps，压力 0 bar，`tdamp = 100 * timestep`，`pdamp = 1000 * timestep`
- `uniaxial_tension`：默认应变速率 `1.0e8 1/s`，最大应变 0.2
- `thermal_expansion`：默认温度点 `[250, 300, 350] K`

## 7. 材料知识库

模块：`mdsynth.knowledge.materials`

最小知识库包含 6 种金属：

| name | symbol | lattice | a0 Angstrom | potential | potential_file | mass |
|---|---|---|---:|---|---|---:|
| copper | Cu | fcc | 3.615 | eam | Cu_u3.eam | 63.546 |
| aluminum | Al | fcc | 4.05 | eam | Al_mm.eam | 26.982 |
| iron | Fe | bcc | 2.866 | eam | Fe_mm.eam | 55.845 |
| gold | Au | fcc | 4.078 | eam | Au_u3.eam | 196.967 |
| tungsten | W | bcc | 3.165 | eam | W_mm.eam | 183.84 |
| nickel | Ni | fcc | 3.52 | eam | Ni_u3.eam | 58.693 |

数据结构：

```python
@dataclass
class MaterialData:
    name: str
    symbol: str
    material_class: str
    crystal_structure: str
    lattice_constant: float
    default_potential: str
    potential_file: str
    mass: float
    density: float
    melting_point: float
    aliases: list[str]
```

实现 `get_material(name)`：

1. 小写并去除空格。
2. 先按 canonical key 查。
3. 再遍历 `aliases` 查。
4. 未找到返回 `None`。

## 8. MDTypedIR：核心中间表示

`MDTypedIR` 是项目最重要的数据结构。它表达一个 MD 实验设计，但不包含 LAMMPS 命令。

### 8.1 System IR

```python
class StructureSourceType(str, Enum):
    GENERATED = "generated"
    FILE = "file"

@dataclass
class CrystalGenerator:
    lattice_type: str
    lattice_constant: Quantity
    replication: tuple[int, int, int] = (10, 10, 10)

@dataclass
class StructureSpec:
    source: StructureSourceType
    generator: CrystalGenerator | None = None
    file_path: str | None = None
    file_checksum: str | None = None

@dataclass
class Species:
    element: str
    role: str = "bulk_atom"
    mass: Quantity | None = None

@dataclass
class SystemIR:
    material_name: str = ""
    material_class: str = "metal"
    structure: StructureSpec | None = None
    species: list[Species] = field(default_factory=list)
    total_atoms: int | None = None
```

### 8.2 ForceField IR

```python
class ForceFieldFamily(str, Enum):
    EAM = "eam"
    MEAM = "meam"
    TERSOFF = "tersoff"
    LJ = "lj"

@dataclass
class PotentialFile:
    name: str
    source: str = "builtin"
    checksum: str | None = None

@dataclass
class ForceFieldIR:
    family: ForceFieldFamily = ForceFieldFamily.EAM
    atom_style: str = "atomic"
    pair_style: str = ""
    pair_style_args: list[str] = field(default_factory=list)
    potential_files: list[PotentialFile] = field(default_factory=list)
    elements_covered: list[str] = field(default_factory=list)
    verification_status: str = "unverified"
    provenance: str = ""
    notes: list[str] = field(default_factory=list)
```

### 8.3 Cell / Group / Protocol

```python
@dataclass
class CellIR:
    boundary: BoundaryCondition
    pressure_control_allowed: dict[str, bool]
    deformation_allowed: dict[str, bool]
```

`CellIR.__post_init__()` 应从边界条件自动推导：

- 只有 periodic 方向允许 pressure control。
- 默认 periodic 方向允许 deformation。

协议阶段：

```python
class EnsembleType(str, Enum):
    NVE = "nve"
    NVT = "nvt"
    NPT = "npt"

class StageType(str, Enum):
    ENERGY_MINIMIZATION = "energy_minimization"
    EQUILIBRATION = "equilibration"
    PRODUCTION = "production"
    DEFORMATION = "deformation"

@dataclass
class EnsembleSpec:
    type: EnsembleType = EnsembleType.NVT
    group: str = "all"
    temperature: Quantity | None = None
    temperature_range: QuantityRange | None = None
    pressure_control: dict[str, Quantity | None] | None = None
    tdamp: Quantity | None = None
    pdamp: Quantity | None = None

@dataclass
class MinimizationSpec:
    energy_tolerance: Quantity | None = None
    force_tolerance: Quantity | None = None
    max_iterations: int = 10000
    max_evaluations: int = 100000

@dataclass
class DeformationSpec:
    mode: str = "uniaxial_tension"
    axis: str = "x"
    strain_rate: Quantity | None = None
    max_strain: Quantity | None = None
    thermostat: EnsembleSpec | None = None

@dataclass
class ProtocolStage:
    id: str
    type: StageType
    ensemble: EnsembleSpec | None = None
    minimization: MinimizationSpec | None = None
    deformation: DeformationSpec | None = None
    duration: Quantity | None = None
    nsteps: int | None = None
```

### 8.4 Output / Analysis / Top-level

```python
@dataclass
class Observable:
    id: str
    type: str
    scope: str = "global"
    group: str | None = None
    components: list[str] = field(default_factory=list)

@dataclass
class ThermoOutput:
    interval: int = 100
    fields: list[str] = ["step", "temp", "pe", "ke", "etotal", "press", "vol", "lx", "ly", "lz"]

@dataclass
class DumpOutput:
    enabled: bool = True
    interval: int = 1000
    fields: list[str] = ["id", "type", "x", "y", "z"]
    format: str = "custom"

@dataclass
class OutputIR:
    thermo: ThermoOutput
    dump: DumpOutput
    restart: bool = True
    restart_interval: int | None = None

@dataclass
class AnalysisStep:
    id: str
    type: str
    inputs: dict = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)

@dataclass
class MDTypedIR:
    ir_version: str = "0.1.0"
    source_intent_id: str = ""
    task_type: str = ""
    system: SystemIR = field(default_factory=SystemIR)
    force_field: ForceFieldIR = field(default_factory=ForceFieldIR)
    cell: CellIR = field(default_factory=CellIR)
    groups: list[GroupIR] = field(default_factory=list)
    stages: list[ProtocolStage] = field(default_factory=list)
    timestep: Quantity | None = None
    units: str = "metal"
    observables: list[Observable] = field(default_factory=list)
    outputs: OutputIR = field(default_factory=OutputIR)
    analysis: list[AnalysisStep] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
```

## 9. IR Planner 实现

模块：`mdsynth.ir.planner`

职责：把 `ScientificIntentSpec` 转为完整 `MDTypedIR`。

主流程：

```python
def plan(intent):
    task_family = get_task_family(intent.task_type)
    if task_family is None:
        raise UnsupportedTaskError(...)

    system_ir = _build_system(intent)
    ff_ir = _build_force_field(intent, system_ir)
    cell_ir = _build_cell(intent)
    stages = _build_stages(intent, task_family, system_ir)
    timestep = _determine_timestep(system_ir)
    observables = _build_observables(intent, task_family)
    analysis = _build_analysis(intent, task_family)
    groups = _build_groups(intent, stages)

    return MDTypedIR(
        ir_version="0.1.0",
        task_type=intent.task_type,
        system=system_ir,
        force_field=ff_ir,
        cell=cell_ir,
        groups=groups,
        stages=stages,
        timestep=timestep,
        units="metal",
        observables=observables,
        outputs=_build_outputs(intent),
        analysis=analysis,
        assumptions=intent.assumptions,
        provenance={"intent_task_type": intent.task_type},
    )
```

关键构建方法：

- `_build_system()`：查材料数据库，生成 `SystemIR`。默认使用晶体生成器，结构源为 `GENERATED`，复制胞常用 `(10, 10, 10)`，根据晶格类型估算总原子数。
- `_build_force_field()`：基于材料默认势函数构建 `ForceFieldIR`。MVP 默认金属走 EAM、`atom_style = atomic`、`pair_style = eam`。
- `_build_cell()`：默认三方向 periodic。若 morphology 是薄膜、纳米线等，可把非周期方向设为 nonperiodic。
- `_determine_timestep()`：metal 单位默认 `0.001 ps`，即 1 fs。
- `_build_stages()`：遍历任务族的 `default_stages`，派发到 minimization / equilibration / deformation / temperature sweep 构建器。
- `_build_equilibration_stage()`：
  - 从 intent 读取温度；缺失时默认 300 K。
  - NPT 压力默认 0 bar。
  - duration 来自 intent 或任务默认。
  - 根据 timestep 折算 `nsteps = duration / timestep`。
- `_build_deformation_stage()`：
  - axis 来自 intent，缺失时默认 x。
  - strain_rate 默认 `1.0e8 1/s`。
  - max_strain 默认 `0.2`。
  - 使用 NVT thermostat。
- `_build_temperature_sweep_stages()`：
  - 对每个温度点创建 NPT equilibration/sample stage。

## 10. Physics Validator 实现

模块：`mdsynth.validator`

验证器接收 `MDTypedIR`，输出 `list[Diagnostic]`。

基本接口：

```python
class Rule:
    rule_id: str
    def check(self, ir: MDTypedIR) -> list[Diagnostic]:
        ...

class PhysicsValidator:
    def validate(self, ir):
        diagnostics = []
        for rule in self.rules:
            try:
                diagnostics.extend(rule.check(ir))
            except Exception as e:
                diagnostics.append(Diagnostic(
                    rule_id=getattr(rule, "rule_id", "unknown"),
                    severity=Severity.ERROR,
                    message=f"Rule execution failed: {e}",
                    path="",
                ))
        return diagnostics
```

默认规则集：

```text
Completeness:
  CheckStructureRequired
  CheckForceFieldRequired
  CheckTemperatureRequired

Boundary:
  CheckBarostatBoundaryCompatibility
  CheckDeformationAxisNotBarostatted

Force field:
  CheckForceFieldCoversElements
  CheckAtomStyleCompatible

Ensemble:
  CheckNoDoubleIntegration
  CheckFixedGroupNotIntegrated

Observables:
  CheckRequiredObservables
  CheckAnalysisInputsExist
  CheckThermalExpansionHasSweep
  CheckTensileHasStopCondition
  CheckDeformationHasAxisAndStop

Numerics:
  CheckTimestepReasonable
  CheckDurationToStepsConversion
  CheckSystemSizeReasonable
```

`Diagnostic` 至少包含：

```python
class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class Diagnostic:
    rule_id: str
    severity: Severity
    message: str
    path: str
    repair_permission: RepairPermission | None = None
    suggested_fix: str | None = None
```

规则要写清楚 `path`，例如：

- `system.structure`
- `force_field.potential_files`
- `stages[1].ensemble.temperature`
- `stages[2].deformation.axis`

这样 repair 引擎才能定位字段。

## 11. Constrained Repair 实现

模块：`mdsynth.repair`

修复权限分级：

```text
A: 可以自动修复，低风险，例如补默认 thermo 输出、补 observables。
B: 可以自动修复，但需要记录假设，例如默认压力 0 bar。
C: 候选修复，需要中等置信度，例如调整应变速率或温度点。
D: 不能自动修复，需要用户确认，例如选择未知势函数。
E: 阻塞，必须用户提供信息，例如未知材料。
```

生成动作：

```python
def generate_actions(ir, errors):
    actions = []
    for error in errors:
        permission = error.repair_permission
        if permission in (A, B):
            action = _auto_repair(ir, error)
            if action:
                actions.append(action)
        elif permission == C:
            action = _candidate_repair(ir, error)
            if action:
                action.confidence = "medium"
                actions.append(action)
        elif permission in (D, E):
            actions.append(RepairAction(
                id=f"blocked_{error.rule_id}",
                permission=permission,
                target_path=error.path,
                action_type="blocked",
                reason=error.message,
                confidence="low",
            ))
    return actions
```

`RepairAction` 建议字段：

```python
@dataclass
class RepairAction:
    id: str
    permission: RepairPermission
    target_path: str
    action_type: str
    reason: str
    confidence: str = "high"
    old_value: Any = None
    new_value: Any = None
```

修复策略：

- `apply_set_value(ir, path, new_value)`：按点路径设置字段。
- `apply_multiply(ir, path, factor)`：把数值字段乘以系数。
- `apply_disable_pressure_control(ir, stage_idx, axis)`：关闭某阶段某轴压力控制。
- `apply_add_observables(ir, observables)`：补缺失观测量。
- `apply_add_temperature_points(ir, temperatures)`：给 thermal expansion 添加更多温度点。

所有策略必须 `deepcopy(ir)` 后返回新 IR，避免原地修改导致历史不可追踪。

## 12. LAMMPS Backend Compiler

模块：`mdsynth.backend`

职责：把验证后的 `MDTypedIR` 编译为 LAMMPS 输入脚本。

主流程：

```python
def compile(ir):
    commands = []
    commands.extend(emit_header(ir))
    commands.extend(emit_system(ir))
    commands.extend(emit_forcefield(ir))
    commands.extend(emit_neighbor(ir))
    commands.extend(emit_protocol(ir))
    commands.extend(emit_output(ir))

    commands = [
        cmd for cmd in commands
        if not (cmd.kind == "variable" and not cmd.args)
    ]

    sorted_commands = topological_sort(commands)
    lowered_ir = LAMMPSLoweredIR(
        commands=sorted_commands,
        metadata={
            "task_type": ir.task_type,
            "material": ir.system.material_name,
            "ir_version": ir.ir_version,
            "units": ir.units,
            "num_atoms": ir.system.total_atoms,
            "num_stages": len(ir.stages),
        },
    )
    return lowered_ir.render(), lowered_ir
```

`LAMMPSCommand`：

```python
@dataclass
class LAMMPSCommand:
    kind: str
    args: list[str]
    comment: str = ""
    depends_on: list[str] = field(default_factory=list)
```

`LAMMPSLoweredIR.render()` 将命令渲染为文本：

```text
clear
units metal    # metal units...
atom_style atomic
...
```

### 12.1 Header emitter

生成：

```text
clear
units <ir.units>
atom_style <ir.force_field.atom_style>
boundary <x> <y> <z>
```

边界字符：

- periodic -> `p`
- fixed/nonperiodic -> `f` 或对应 LAMMPS 字符

### 12.2 System emitter

若 `StructureSourceType.GENERATED`：

```text
lattice <lattice_type> <lattice_constant>
region simbox block 0 <rx> 0 <ry> 0 <rz>
create_box <ntypes> simbox
create_atoms 1 region simbox
```

若 `StructureSourceType.FILE`：

```text
read_data <file_path>
```

### 12.3 Force field emitter

生成：

```text
mass <type_id> <mass>
pair_style <pair_style> <pair_style_args...>
pair_coeff * * <potential_file>
```

### 12.4 Protocol emitter

全局：

```text
timestep <dt>
```

Minimization：

```text
minimize <etol> <ftol> <max_iterations> <max_evaluations>
```

Equilibration：

```text
velocity all create <T> <seed> dist gaussian
fix <fix_id> <group> nvt temp <T> <T> <tdamp>
run <nsteps>
unfix <fix_id>
```

NPT：

```text
fix <fix_id> <group> npt temp <T> <T> <tdamp> iso <P> <P> <pdamp>
```

Deformation：

```text
fix <therm_id> all nvt temp <T> <T> <tdamp>
fix <deform_id> all deform 1 <axis> erate <strain_rate_per_ps> units box
run <nsteps>
unfix <therm_id>
unfix <deform_id>
```

应变速率换算：

```python
rate_lammps = strain_rate_1_per_s * 1e-12  # 1/s -> 1/ps
strain_per_step = rate_lammps * timestep_ps
nsteps = int(max_strain / strain_per_step)
```

随机种子应可复现：用 `md5(stage_id)` 转整数，再限制到 LAMMPS 合法范围。

### 12.5 Output emitter

应至少生成：

```text
thermo <interval>
thermo_style custom <fields...>
dump <id> all custom <interval> dump.lammpstrj <fields...>
restart <interval> restart.*
```

具体字段来自 `OutputIR.thermo.fields` 和 `OutputIR.dump.fields`。

## 13. Sandbox Preflight

模块：`mdsynth.sandbox`

职责：在真正交付前，对 LAMMPS 脚本做轻量预检查。若没有 LAMMPS 环境，可返回 passed 并给 warning。

`SandboxPreflightRunner.run(script, ir)` 阶段：

1. Phase 1：build check。逐行执行脚本中非 run 类命令，检查脚本是否能解析。
2. Phase 2：initial check。读取初始势能、动能，检查是否 finite。
3. Phase 3：若有 minimization，则执行 minimization，检查收敛和能量 finite。
4. Phase 4：短 NVE，约 100 steps，检查能量漂移。
5. Phase 5：短 ensemble，约 200 steps，检查温度稳定性。
6. cleanup 临时目录和 LAMMPS 对象。

`PreflightReport` 应包含：

```python
@dataclass
class PreflightReport:
    passed: bool = False
    script_compiles: bool = False
    initial_energy_finite: bool = False
    force_finite: bool = False
    minimization_converged: bool = False
    short_run_completed: bool = False
    temperature_stable: bool = False
    energy_drift_percent: float | None = None
    nan_detected: bool = False
    lost_atoms: bool = False
    error_messages: list[str] = field(default_factory=list)
    warning_messages: list[str] = field(default_factory=list)
```

没有安装 LAMMPS 时：

```python
PreflightReport(
    passed=True,
    script_compiles=True,
    warning_messages=["LAMMPS not available; sandbox skipped"]
)
```

## 14. Evidence Package

模块：`mdsynth.evidence`

职责：把所有产物打包成可复现输出。

构建逻辑：

```python
def build(intent_spec, md_ir, diagnostics, lammps_script, preflight_report, repair_history):
    output = MDSynthOutput()
    output.lammps_script = lammps_script
    output.md_ir_yaml = to_yaml(md_ir)
    output.intent_spec_yaml = to_yaml(intent_spec)

    errors = [d for d in diagnostics if d.severity.value == "error"]
    warnings = [d for d in diagnostics if d.severity.value == "warning"]
    infos = [d for d in diagnostics if d.severity.value == "info"]

    validation_report = {
        "static_checks": {
            "passed": len(infos),
            "warnings": len(warnings),
            "failed": len(errors),
        },
        "errors": [{"rule_id": d.rule_id, "message": d.message, "path": d.path} for d in errors],
        "warnings": [{"rule_id": d.rule_id, "message": d.message, "path": d.path} for d in warnings],
    }

    output.validation_report_json = to_json(validation_report)
    output.preflight_report_json = to_json(preflight_report)
    output.provenance_lock_yaml = to_yaml(manifest)
    output.readme_md = build_readme(...)
    output.assumptions_md = build_assumptions(...)
    output.semantic_diff_md = build_semantic_diff(repair_history)
    output.success = not errors and preflight_report.passed
    return output
```

写盘文件：

```text
output/
  in.main.lammps
  md_ir.yaml
  intent_spec.yaml
  validation_report.json
  preflight_report.json
  provenance.lock
  README.md
  assumptions.md
  semantic_diff.md
```

`semantic_diff.md` 只在存在修复历史时写出。

## 15. 序列化工具

需要支持 dataclass、Enum、Path、嵌套列表和字典。

建议实现：

```python
class MDSynthEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

def to_json(obj):
    return json.dumps(obj, cls=MDSynthEncoder, indent=2, ensure_ascii=False)

def to_yaml(obj):
    return yaml.safe_dump(json.loads(to_json(obj)), allow_unicode=True, sort_keys=False)
```

## 16. 配置

`MDSynthConfig` 至少包含：

```python
@dataclass
class MDSynthConfig:
    max_repair_rounds: int = 3
    sandbox_enabled: bool = False
    lammps_executable: str = ""
    output_dir: str = "./output"
    default_units: str = "metal"
```

`get_config()` 可以读取环境变量：

- `MDSYNTH_MAX_REPAIR_ROUNDS`
- `MDSYNTH_SANDBOX_ENABLED`
- `LAMMPS_EXECUTABLE`

## 17. 示例运行

```python
from mdsynth import MDSynthPipeline
from mdsynth.llm.base import MockLLMBackend

pipeline = MDSynthPipeline(llm_backend=MockLLMBackend())
output = pipeline.run("copper NPT equilibration at 300K, 0 bar for 200ps")
pipeline.save(output, "./output")

print(output.lammps_script)
```

期望脚本包含：

```text
clear
units metal
atom_style atomic
boundary p p p
lattice fcc 3.615
region simbox block 0 10 0 10 0 10
create_box 1 simbox
create_atoms 1 region simbox
mass 1 63.546
pair_style eam
pair_coeff * * Cu_u3.eam
timestep 0.001
minimize ...
velocity all create 300 ...
fix fx_equil all npt temp 300 300 0.1 iso 0 0 1.0
run 200000
unfix fx_equil
thermo 100
```

## 18. 测试验收标准

至少写以下测试：

### Intent extractor

- copper NPT 输入能提取 `task_type=equilibration_npt`
- tension 输入能提取 `task_type=uniaxial_tension` 和 deformation_axis
- thermal expansion 输入能提取 `task_type=thermal_expansion`
- 缺少温度时进入 `missing_defaultable`

### IR planner

- 支持 5 个任务类型。
- copper NPT 生成 SystemIR、ForceFieldIR、CellIR、ProtocolStage。
- unknown task 抛出 `UnsupportedTaskError`。
- thermal expansion 生成多个温度 stage。

### Validator

- 缺少结构报 error。
- 缺少力场报 error。
- NPT 在 nonperiodic 方向控压报 error。
- 拉伸方向和 barostat 方向冲突报 error 或 warning。
- 有效最小 IR 无 blocking error。

### Repair

- A/B/C 权限生成可执行 repair action。
- D/E 权限生成 blocked action。
- apply_actions 后 IR 改变且历史被记录。

### Compiler

- compile 返回脚本文本和 LoweredIR。
- 脚本包含 units、atom_style、boundary、lattice、create_box、create_atoms、mass、pair_style、pair_coeff、timestep、thermo。
- 命令顺序正确：header -> system -> forcefield -> protocol -> output。

### Pipeline integration

- 用 MockLLMBackend 跑 5 个 supported tasks 都能返回 `MDSynthOutput`。
- `save()` 写出完整 output 文件夹。
- 所有测试通过，例如当前项目为 `66 passed`。

## 19. 复现优先级

如果从零实现，按这个顺序推进：

1. 实现 `ScientificIntentSpec`、`MDTypedIR`、`Quantity` 等数据结构。
2. 实现材料库和任务族。
3. 实现 `MockLLMBackend` 和 `ScientificIntentExtractor`。
4. 实现 `MDTypedIRPlanner`，先覆盖 5 个任务。
5. 实现 `PhysicsValidator` 和最小规则集。
6. 实现 repair action 与 3-5 个常用策略。
7. 实现 `LAMMPSCommand`、emitters、compiler。
8. 实现 evidence package 写盘。
9. 最后接入真实 OpenAI backend 和可选 LAMMPS sandbox。

这样可以在没有真实 LLM 和没有 LAMMPS 的情况下，先得到一个可测试、可演示、可扩展的确定性系统。

