# LAMMPS 脚本自动生成系统 — 完整开发规格说明书

> **文档版本**: 1.0  
> **目标读者**: Claude Code / AI 编码智能体  
> **目的**: 基于本文档，AI 编码智能体可独立完成整套系统的开发，无需频繁追问设计决策  

---

## 目录

1. [项目概述与目标](#1-项目概述与目标)
2. [总体架构](#2-总体架构)
3. [模块 1: Scientific Intent Spec](#3-模块-1-scientific-intent-spec)
4. [模块 2: MD Typed IR](#4-模块-2-md-typed-ir)
5. [模块 3: 物理约束检查器](#5-模块-3-物理约束检查器)
6. [模块 4: LAMMPS 确定性编译器](#6-模块-4-lammps-确定性编译器)
7. [模块 5: 沙箱预运行与数值诊断](#7-模块-5-沙箱预运行与数值诊断)
8. [模块 6: 受约束自动修复](#8-模块-6-受约束自动修复)
9. [模块 7: 脚本 + 证据包 + 可复现包](#9-模块-7-脚本--证据包--可复现包)
10. [MVP 任务族定义](#10-mvp-任务族定义)
11. [LLM 调用规范](#11-llm-调用规范)
12. [错误处理与日志](#12-错误处理与日志)
13. [测试策略](#13-测试策略)
14. [项目目录结构](#14-项目目录结构)
15. [实现顺序](#15-实现顺序)

---

## 1. 项目概述与目标

### 1.1 一句话描述

构建一个 **Python 库**，接收用户的自然语言 MD 研究目标，自动生成可运行、物理可信、可复现的 LAMMPS 输入脚本及完整证据包。

### 1.2 MVP 硬约束

| 维度 | MVP 范围 | 明确排除 |
|------|----------|----------|
| **体系类型** | 金属晶体（fcc/bcc），含 EAM/MEAM 势函数 | 分子体系、聚合物、ReaxFF、ML 势 |
| **任务类型** | 结构弛豫、NVT/NPT 平衡、单轴拉伸、热膨胀系数 | 热导率、GCMC、界面、缺陷形成能 |
| **LAMMPS units** | metal | lj / real / si / cgs / electron |
| **atom_style** | atomic | charge / full / molecular |
| **交互类型** | pair only（无 bond/angle/dihedral） | bonded interactions |
| **LLM** | OpenAI GPT-4o（可切换） | 本地模型（架构预留切换能力） |
| **语言** | 中文 + 英文输入 | — |
| **输出** | LAMMPS 脚本 + JSON 证据包 + 可复现目录 | GUI / Web 界面 |

### 1.3 设计原则

1. **LLM 只负责理解意图和补全缺失信息** — 不直接生成 LAMMPS 命令
2. **所有 LAMMPS 知识都是确定性的** — 命令模板、参数约束、单位换算由代码保证
3. **先检查后生成** — 物理冲突在 IR 层发现，不在 LAMMPS 报错时发现
4. **修复改 IR，不改脚本** — 自动修复总是修改 IR 然后重新编译
5. **每个决策都有证据** — 脚本携带来源、假设、验证结果、限制条件

---

## 2. 总体架构

### 2.1 数据流

```
用户自然语言
      │
      ▼
┌─────────────────────────────────┐
│  ScientificIntentExtractor      │  ← LLM (GPT-4o)
│  输出: ScientificIntentSpec     │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  MDTypedIRPlanner               │  ← 规则引擎 + LLM
│  输出: MDTypedIR                │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  PhysicsConstraintValidator     │  ← 纯规则引擎
│  输出: list[Diagnostic]         │
└──────────────┬──────────────────┘
               │ (有 ERROR → 进入修复循环)
               ▼
┌─────────────────────────────────┐
│  LAMMPSBackendCompiler          │  ← 纯代码生成器
│  输出: LAMMPSLoweredIR → .in    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  SandboxPreflightRunner         │  ← LAMMPS Python API
│  输出: PreflightReport          │
└──────────────┬──────────────────┘
               │ (失败 → 进入自动修复)
               ▼
┌─────────────────────────────────┐
│  ConstrainedRepairEngine        │  ← 规则优先 + LLM 辅助
│  输出: MDTypedIR Patch          │
└──────────────┬──────────────────┘
               │ (成功后)
               ▼
┌─────────────────────────────────┐
│  EvidencePackageBuilder         │  ← 纯组装器
│  输出: 脚本 + 证据包 + 可复现包  │
└─────────────────────────────────┘
```

### 2.2 核心 Python 包结构

```
mdsynth/                        # 顶层包名 (Molecular Dynamics SYNTHesis)
├── __init__.py
├── pipeline.py                 # 主流水线编排器 MDSynthPipeline
│
├── intent/                     # 模块 1: Scientific Intent Spec
│   ├── __init__.py
│   ├── spec.py                 # ScientificIntentSpec 数据类
│   ├── extractor.py            # ScientificIntentExtractor (LLM)
│   └── taxonomy.py             # 任务分类法 + 材料分类法
│
├── ir/                         # 模块 2: MD Typed IR
│   ├── __init__.py
│   ├── md_ir.py                # MDTypedIR 及所有子类型
│   ├── planner.py              # IRPlanner (从 IntentSpec 生成 MDTypedIR)
│   ├── task_families.py        # 任务族定义 (每个任务需要什么)
│   └── defaults.py             # 默认值策略
│
├── validator/                  # 模块 3: 物理约束检查器
│   ├── __init__.py
│   ├── engine.py               # PhysicsValidator 引擎
│   ├── diagnostic.py           # Diagnostic / Severity 类型
│   ├── rules/                  # 检查规则插件
│   │   ├── __init__.py
│   │   ├── completeness.py     # 完整性规则
│   │   ├── boundary.py         # 边界条件规则
│   │   ├── force_field.py      # 力场兼容性规则
│   │   ├── ensemble.py         # 系综/积分器规则
│   │   ├── observables.py      # 目标观测量规则
│   │   └── numerics.py         # 数值启发式规则
│   └── repair_hints.py         # 修复提示生成
│
├── backend/                    # 模块 4: LAMMPS 编译器
│   ├── __init__.py
│   ├── compiler.py             # LAMMPSBackendCompiler
│   ├── lowered_ir.py           # LAMMPSLoweredIR
│   ├── command_templates.py    # 命令模板库
│   ├── dependency_graph.py     # 命令依赖 DAG + 拓扑排序
│   ├── emitters/               # 分块代码生成器
│   │   ├── __init__.py
│   │   ├── header.py           # units / atom_style / boundary
│   │   ├── system.py           # lattice / region / create_box / create_atoms
│   │   ├── forcefield.py       # pair_style / pair_coeff / mass
│   │   ├── neighbor.py         # neighbor / neigh_modify
│   │   ├── protocol.py         # minimize / velocity / fix / run
│   │   └── output.py           # thermo / dump / restart
│   └── units.py                # 单位系统处理
│
├── sandbox/                    # 模块 5: 沙箱预运行
│   ├── __init__.py
│   ├── runner.py               # SandboxPreflightRunner
│   ├── diagnostics.py          # 数值诊断器
│   └── report.py               # PreflightReport
│
├── repair/                     # 模块 6: 受约束自动修复
│   ├── __init__.py
│   ├── engine.py               # ConstrainedRepairEngine
│   ├── classifier.py           # 错误分类器
│   ├── strategies.py           # 修复策略库
│   └── permissions.py          # 修复权限分级 (A/B/C/D/E)
│
├── evidence/                   # 模块 7: 证据包
│   ├── __init__.py
│   ├── builder.py              # EvidencePackageBuilder
│   ├── manifest.py             # ProvenanceManifest
│   └── report_templates.py     # 报告模板
│
├── knowledge/                  # 知识库 (不依赖 LLM 的确定性数据)
│   ├── __init__.py
│   ├── lammps_commands.py      # 命令本体库
│   ├── force_fields.py         # 力场注册表
│   ├── materials.py            # 材料-势函数映射
│   └── heuristics.py           # 经验规则 (timestep 范围等)
│
├── llm/                        # LLM 抽象层
│   ├── __init__.py
│   ├── base.py                 # LLMBackend 抽象基类
│   ├── openai_backend.py       # OpenAI 实现
│   └── prompts/                # Prompt 模板
│       ├── __init__.py
│       ├── intent_extraction.py
│       └── ir_planning.py
│
└── utils/                      # 工具函数
    ├── __init__.py
    ├── quantities.py           # Quantity / Unit 类型
    ├── serialization.py        # YAML/JSON 序列化
    └── logging.py              # 结构化日志
```

### 2.3 Pipeline 主入口伪代码

```python
class MDSynthPipeline:
    def __init__(
        self,
        llm_backend: LLMBackend,
        validator: PhysicsValidator,
        compiler: LAMMPSBackendCompiler,
        sandbox: SandboxPreflightRunner,
        repair: ConstrainedRepairEngine,
        evidence_builder: EvidencePackageBuilder,
        max_repair_rounds: int = 3,
    ): ...

    def run(self, user_request: str) -> MDSynthOutput:
        # Step 1: NL → Intent Spec
        intent_spec = self.intent_extractor.extract(user_request)
        
        # Step 2: Intent Spec → MD Typed IR
        md_ir = self.ir_planner.plan(intent_spec)
        
        # Step 3: Validate IR (loop with repair)
        for round_num in range(self.max_repair_rounds):
            diagnostics = self.validator.validate(md_ir)
            errors = [d for d in diagnostics if d.severity == Severity.ERROR]
            
            if not errors:
                break
            
            repair_actions = self.repair.generate_actions(md_ir, errors)
            md_ir = self.repair.apply_actions(md_ir, repair_actions)
        else:
            # Max rounds exceeded
            raise RepairExhaustedError(...)
        
        # Step 4: Compile to LAMMPS
        lammps_script, lowered_ir = self.compiler.compile(md_ir)
        
        # Step 5: Sandbox preflight
        preflight_report = self.sandbox.run(lammps_script, md_ir)
        
        # Step 6: Build evidence package
        output = self.evidence_builder.build(
            intent_spec=intent_spec,
            md_ir=md_ir,
            diagnostics=diagnostics,
            lammps_script=lammps_script,
            preflight_report=preflight_report,
            repair_history=self.repair.get_history(),
        )
        
        return output
```

---

## 3. 模块 1: Scientific Intent Spec

### 3.1 职责

将用户的自然语言研究目标转换为结构化的科学意图说明书。**完全不涉及 LAMMPS 命令、势函数选择、模拟协议**。

### 3.2 核心数据结构

```python
# mdsynth/intent/spec.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FieldStatus(str, Enum):
    KNOWN = "known"          # 用户明确提供
    UNKNOWN = "unknown"      # 缺失且无法推断
    DEFAULTED = "defaulted"  # 系统默认值
    INFERRED = "inferred"    # 从上下文推断


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


@dataclass
class QuantitySpec:
    """带状态和来源的物理量"""
    value: Optional[float] = None
    unit: Optional[str] = None
    status: FieldStatus = FieldStatus.UNKNOWN
    source: str = ""  # "user" | "default" | "inferred_from_xxx"


@dataclass
class MaterialSpec:
    """研究对象描述"""
    name: Optional[str] = None           # e.g. "copper", "Cu"
    material_class: Optional[MaterialClass] = None
    morphology: Morphology = Morphology.UNSPECIFIED
    crystal_structure: Optional[str] = None  # "fcc", "bcc", "hcp"
    composition: dict[str, float] = field(default_factory=dict)  # element -> fraction
    structure_source: Optional[str] = None  # "generated" | uploaded file path


@dataclass
class TargetProperty:
    """用户想要计算的科学性质"""
    name: str
    priority: str = "primary"  # "primary" | "secondary"


@dataclass
class ScientificIntentSpec:
    """
    科学意图说明书。
    
    这是整个系统最上游的数据结构。
    它描述用户想研究什么，但不涉及如何实现。
    """
    # 任务类型（枚举值，从 taxonomy 中选取）
    task_type: str = ""
    task_description: str = ""
    extraction_confidence: Confidence = Confidence.MEDIUM
    
    # 研究对象
    material: MaterialSpec = field(default_factory=MaterialSpec)
    
    # 目标性质
    target_properties: list[TargetProperty] = field(default_factory=list)
    
    # 条件
    temperature: QuantitySpec = field(default_factory=QuantitySpec)
    pressure: QuantitySpec = field(default_factory=QuantitySpec)
    
    # 变形相关（如果适用）
    deformation_axis: Optional[str] = None       # "x" | "y" | "z"
    deformation_mode: Optional[str] = None       # "uniaxial_tension" | "uniaxial_compression"
    
    # 用户约束
    simulation_time: QuantitySpec = field(default_factory=QuantitySpec)
    speed_vs_accuracy: str = "balanced"          # "speed" | "balanced" | "accuracy"
    
    # 期望输出
    desired_outputs: list[str] = field(default_factory=list)
    
    # 缺失信息分类
    missing_blocking: list[str] = field(default_factory=list)
    missing_defaultable: list[str] = field(default_factory=list)
    
    # 假设与风险
    assumptions: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    
    # 溯源
    original_user_text: str = ""
```

### 3.3 任务分类法 (Taxonomy)

```python
# mdsynth/intent/taxonomy.py

# MVP 支持的任务类型
MVP_TASK_TYPES = {
    "structure_relaxation": {
        "display_name": "结构弛豫/能量最小化",
        "description": "对初始结构进行能量最小化，得到稳定构型",
        "required_fields": ["material"],
        "defaultable_fields": [],
    },
    "equilibration_nvt": {
        "display_name": "NVT 平衡",
        "description": "在恒定温度下进行等温平衡模拟",
        "required_fields": ["material", "temperature"],
        "defaultable_fields": ["duration", "pressure"],
    },
    "equilibration_npt": {
        "display_name": "NPT 平衡",
        "description": "在恒定温度和压力下进行等温等压平衡",
        "required_fields": ["material", "temperature"],
        "defaultable_fields": ["pressure", "duration"],
    },
    "uniaxial_tension": {
        "display_name": "单轴拉伸",
        "description": "沿指定方向对材料进行拉伸变形",
        "required_fields": ["material", "temperature", "deformation_axis"],
        "defaultable_fields": ["strain_rate", "max_strain", "pressure"],
    },
    "thermal_expansion": {
        "display_name": "热膨胀系数",
        "description": "通过多温度点 NPT 模拟计算热膨胀系数",
        "required_fields": ["material"],
        "defaultable_fields": ["temperature_range", "num_points"],
    },
}

# MVP 支持的材料类型
MVP_MATERIALS = {
    "copper": {
        "symbol": "Cu",
        "material_class": "metal",
        "crystal_structure": "fcc",
        "lattice_constant": 3.615,  # Å
        "default_potential": "eam/Cu_u3",
        "potential_file": "Cu_u3.eam",
        "mass": 63.546,
    },
    "aluminum": {
        "symbol": "Al",
        "material_class": "metal",
        "crystal_structure": "fcc",
        "lattice_constant": 4.05,
        "default_potential": "eam/Al_mm",
        "potential_file": "Al_mm.eam",
        "mass": 26.982,
    },
    "iron": {
        "symbol": "Fe",
        "material_class": "metal",
        "crystal_structure": "bcc",
        "lattice_constant": 2.866,
        "default_potential": "eam/Fe_mm",
        "potential_file": "Fe_mm.eam",
        "mass": 55.845,
    },
    "gold": {
        "symbol": "Au",
        "material_class": "metal",
        "crystal_structure": "fcc",
        "lattice_constant": 4.078,
        "default_potential": "eam/Au_u3",
        "potential_file": "Au_u3.eam",
        "mass": 196.967,
    },
    "tungsten": {
        "symbol": "W",
        "material_class": "metal",
        "crystal_structure": "bcc",
        "lattice_constant": 3.165,
        "default_potential": "eam/W_mm",
        "potential_file": "W_mm.eam",
        "mass": 183.84,
    },
}
```

### 3.4 IntentExtractor 实现规范

```python
# mdsynth/intent/extractor.py

class ScientificIntentExtractor:
    """
    使用 LLM 从自然语言中提取科学意图。
    
    流程:
    1. 发送 prompt（包含用户输入 + JSON schema）
    2. 使用 GPT-4o structured output 获取 JSON
    3. 后处理：校验字段、规范单位、分类缺失信息
    4. 返回 ScientificIntentSpec
    """
    
    def __init__(self, llm_backend: LLMBackend):
        self.llm = llm_backend
    
    def extract(self, user_request: str) -> ScientificIntentSpec:
        """
        从自然语言提取科学意图。
        
        Args:
            user_request: 用户原始输入（中文或英文）
            
        Returns:
            结构化的 ScientificIntentSpec
        """
        # Step 1: LLM 提取
        raw_json = self.llm.extract_intent(user_request)
        
        # Step 2: Schema 验证
        validated = self._validate_schema(raw_json)
        
        # Step 3: 补齐和分类
        spec = self._build_spec(validated)
        
        # Step 4: 缺失信息分类
        spec = self._classify_missing(spec)
        
        # Step 5: 风险标记
        spec = self._flag_risks(spec)
        
        return spec
```

**Prompt 设计要点**：
- 使用 GPT-4o 的 `response_format` 为 structured JSON
- 要求 LLM 识别：任务类型、材料、温度、压力、变形参数
- 区分 "用户明确说了" vs "用户未提及"
- 材料名映射到标准符号（如 "铜" → "Cu", "copper" → "Cu"）

### 3.5 输入/输出示例

**输入**:
```
计算铜在 300K 下的拉伸模量，沿 x 方向拉伸
```

**输出 ScientificIntentSpec**:
```yaml
task_type: uniaxial_tension
task_description: "计算铜在 300K 下的拉伸力学响应，估计杨氏模量"
extraction_confidence: high
material:
  name: copper
  material_class: metal
  morphology: bulk
  crystal_structure: fcc
  structure_source: generated
target_properties:
  - name: youngs_modulus
    priority: primary
  - name: stress_strain_curve
    priority: primary
temperature:
  value: 300
  unit: K
  status: known
  source: user
pressure:
  value: null
  unit: atm
  status: unknown
deformation_axis: x
deformation_mode: uniaxial_tension
missing_blocking: []
missing_defaultable:
  - strain_rate
  - max_strain
  - equilibration_duration
  - system_size
assumptions:
  - 使用 EAM 势函数描述 Cu 原子间相互作用（系统默认）
  - 使用 fcc 晶格生成初始结构（系统默认）
  - 晶格常数取 3.615 Å（系统默认）
risk_flags:
  - MD 拉伸应变率远高于实验应变率
  - 杨氏模量值对应变率敏感
```

---

## 4. 模块 2: MD Typed IR

### 4.1 职责

将 ScientificIntentSpec 转换为带类型的模拟实验设计图。**表达"为了完成科学目标，需要构建什么体系、用什么相互作用、经历什么阶段、输出什么可观测量"**，但不直接包含 LAMMPS 命令。

### 4.2 核心数据结构

```python
# mdsynth/ir/md_ir.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Literal


# ============================================================
# 基础类型
# ============================================================

class Dimension(str, Enum):
    """物理量纲"""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    TIME = "time"
    LENGTH = "length"
    ENERGY = "energy"
    FORCE = "force"
    VELOCITY = "velocity"
    DIMENSIONLESS = "dimensionless"
    STRAIN_RATE = "strain_rate"
    DENSITY = "density"


@dataclass
class Quantity:
    """带类型和单位的值"""
    value: float
    unit: str
    dimension: Dimension
    source: str = "default"   # "user" | "default" | "inferred" | "repaired"
    confidence: str = "medium"


@dataclass
class QuantityRange:
    """范围量"""
    start: Quantity
    stop: Quantity


class BoundaryType(str, Enum):
    PERIODIC = "periodic"
    NONPERIODIC = "nonperiodic"
    FIXED = "fixed"
    SHRINK_WRAP = "shrink_wrap"


@dataclass
class BoundaryCondition:
    x: BoundaryType = BoundaryType.PERIODIC
    y: BoundaryType = BoundaryType.PERIODIC
    z: BoundaryType = BoundaryType.PERIODIC


# ============================================================
# 体系 IR
# ============================================================

class StructureSourceType(str, Enum):
    GENERATED = "generated"
    FILE = "file"


@dataclass
class CrystalGenerator:
    """晶体结构生成器参数"""
    lattice_type: str  # "fcc", "bcc", "hcp"
    lattice_constant: Quantity  # Å
    replication: tuple[int, int, int] = (10, 10, 10)


@dataclass
class StructureSpec:
    source: StructureSourceType
    generator: Optional[CrystalGenerator] = None
    file_path: Optional[str] = None
    file_checksum: Optional[str] = None


@dataclass
class Species:
    element: str
    role: str  # "bulk_atom"
    mass: Optional[Quantity] = None


@dataclass
class SystemIR:
    material_name: str
    material_class: str
    structure: StructureSpec
    species: list[Species] = field(default_factory=list)
    total_atoms: Optional[int] = None


# ============================================================
# 力场 IR
# ============================================================

class ForceFieldFamily(str, Enum):
    EAM = "eam"
    MEAM = "meam"
    TERSOFF = "tersoff"
    LJ = "lj"


@dataclass
class PotentialFile:
    name: str
    source: str  # "builtin" | "user_upload" | "openkim"
    checksum: Optional[str] = None


@dataclass
class ForceFieldIR:
    family: ForceFieldFamily
    atom_style: str = "atomic"
    pair_style: str = ""
    pair_style_args: list[str] = field(default_factory=list)
    potential_files: list[PotentialFile] = field(default_factory=list)
    elements_covered: list[str] = field(default_factory=list)
    verification_status: str = "unverified"  # "verified" | "user_supplied" | "unverified"
    provenance: str = ""
    notes: list[str] = field(default_factory=list)


# ============================================================
# 盒子 IR
# ============================================================

@dataclass
class CellIR:
    boundary: BoundaryCondition
    pressure_control_allowed: dict[str, bool] = field(default_factory=dict)
    deformation_allowed: dict[str, bool] = field(default_factory=dict)
    
    def __post_init__(self):
        # 自动推导哪些方向可以控压
        if not self.pressure_control_allowed:
            self.pressure_control_allowed = {
                axis: getattr(self.boundary, axis) == BoundaryType.PERIODIC
                for axis in ("x", "y", "z")
            }
        # 默认所有周期方向都可以变形
        if not self.deformation_allowed:
            self.deformation_allowed = {
                axis: getattr(self.boundary, axis) == BoundaryType.PERIODIC
                for axis in ("x", "y", "z")
            }


# ============================================================
# 组 IR
# ============================================================

class GroupSelectorType(str, Enum):
    ALL = "all"
    REGION = "region"
    TYPE = "type"
    SUBTRACT = "subtract"
    UNION = "union"
    INTERSECT = "intersect"


@dataclass
class RegionSpec:
    type: str  # "block", "sphere", "cylinder"
    args: dict = field(default_factory=dict)


@dataclass
class GroupIR:
    id: str
    selector_type: GroupSelectorType
    region: Optional[RegionSpec] = None
    atom_types: Optional[list[int]] = None
    subtract_groups: list[str] = field(default_factory=list)
    description: str = ""


# ============================================================
# 协议 IR
# ============================================================

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
    type: EnsembleType
    group: str = "all"
    temperature: Optional[Quantity] = None
    temperature_range: Optional[QuantityRange] = None
    # NPT 专用
    pressure_control: Optional[dict[str, Optional[Quantity]]] = None
    # 阻尼参数
    tdamp: Optional[Quantity] = None
    pdamp: Optional[Quantity] = None


@dataclass
class MinimizationSpec:
    energy_tolerance: Quantity = field(default_factory=lambda: 
        Quantity(1e-6, "dimensionless", Dimension.DIMENSIONLESS))
    force_tolerance: Quantity = field(default_factory(lambda:
        Quantity(1e-8, "kcal/mol/Å", Dimension.FORCE))
    max_iterations: int = 10000
    max_evaluations: int = 100000


@dataclass
class DeformationSpec:
    mode: str  # "uniaxial_tension", "uniaxial_compression"
    axis: str  # "x", "y", "z"
    strain_rate: Quantity
    max_strain: Quantity
    thermostat: EnsembleSpec


@dataclass
class ProtocolStage:
    id: str
    type: StageType
    ensemble: Optional[EnsembleSpec] = None
    minimization: Optional[MinimizationSpec] = None
    deformation: Optional[DeformationSpec] = None
    duration: Optional[Quantity] = None
    nsteps: Optional[int] = None


# ============================================================
# 可观测量 IR
# ============================================================

@dataclass
class Observable:
    id: str
    type: str  # "thermodynamic_scalar", "tensor", "msd", "rdf", "per_atom"
    scope: str = "global"  # "global", "per_atom", "group"
    group: Optional[str] = None
    components: list[str] = field(default_factory=list)


# ============================================================
# 输出 IR
# ============================================================

@dataclass
class ThermoOutput:
    interval: int = 100
    fields: list[str] = field(default_factory=lambda: [
        "step", "temp", "pe", "ke", "etotal", "press", "vol", "lx", "ly", "lz"
    ])


@dataclass
class DumpOutput:
    enabled: bool = True
    interval: int = 1000
    fields: list[str] = field(default_factory=lambda: ["id", "type", "x", "y", "z"])
    format: str = "custom"  # "custom", "xyz", "lammpstrj"


@dataclass
class OutputIR:
    thermo: ThermoOutput = field(default_factory=ThermoOutput)
    dump: DumpOutput = field(default_factory=DumpOutput)
    restart: bool = True
    restart_interval: Optional[int] = None  # None = only at end


# ============================================================
# 分析 IR
# ============================================================

@dataclass
class AnalysisStep:
    id: str
    type: str  # "linear_fit", "time_average", "msd_fit"
    inputs: dict = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)


# ============================================================
# 顶层 MD Typed IR
# ============================================================

@dataclass
class MDTypedIR:
    """MD 实验设计图 — 系统核心数据结构"""
    # 元数据
    ir_version: str = "0.1.0"
    source_intent_id: str = ""
    task_type: str = ""
    
    # 模拟对象
    system: SystemIR = field(default_factory=SystemIR)
    
    # 相互作用
    force_field: ForceFieldIR = field(default_factory=ForceFieldIR)
    
    # 模拟盒子
    cell: CellIR = field(default_factory=CellIR)
    
    # 分组
    groups: list[GroupIR] = field(default_factory=list)
    
    # 模拟协议
    stages: list[ProtocolStage] = field(default_factory=list)
    
    # 全局参数
    timestep: Optional[Quantity] = None
    units: str = "metal"
    
    # 可观测量
    observables: list[Observable] = field(default_factory=list)
    
    # 输出配置
    outputs: OutputIR = field(default_factory=OutputIR)
    
    # 分析步骤
    analysis: list[AnalysisStep] = field(default_factory=list)
    
    # 假设与溯源
    assumptions: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
```

### 4.3 IRPlanner 实现规范

```python
# mdsynth/ir/planner.py

class MDTypedIRPlanner:
    """
    将 ScientificIntentSpec 转换为 MDTypedIR。
    
    转换过程:
    1. 根据 task_type 选择任务族模板
    2. 根据 material 确定体系生成方式
    3. 根据任务族要求生成协议阶段
    4. 自动配置 observables 和 analysis
    5. 用默认值补全非关键参数（标记来源）
    """
    
    def __init__(self, llm_backend: Optional[LLMBackend] = None):
        self.llm = llm_backend
        self.task_families = load_task_families()
        self.defaults = load_defaults()
    
    def plan(self, intent: ScientificIntentSpec) -> MDTypedIR:
        """
        从科学意图规划完整的 MD 实验。
        
        核心是确定性规则，LLM 仅在处理歧义时辅助。
        """
        # Step 1: 加载任务族
        task_family = self.task_families.get(intent.task_type)
        if task_family is None:
            raise UnsupportedTaskError(intent.task_type)
        
        # Step 2: 构建 SystemIR
        system_ir = self._build_system(intent)
        
        # Step 3: 构建 ForceFieldIR
        ff_ir = self._build_force_field(intent, system_ir)
        
        # Step 4: 构建 CellIR
        cell_ir = self._build_cell(intent)
        
        # Step 5: 构建 ProtocolStages（核心逻辑）
        stages = self._build_stages(intent, task_family)
        
        # Step 6: 确定 timestep
        timestep = self._determine_timestep(system_ir, ff_ir)
        
        # Step 7: 构建 observables 和 analysis
        observables = self._build_observables(intent, task_family)
        analysis = self._build_analysis(intent, task_family)
        
        # Step 8: 构建 Groups（如果任务需要）
        groups = self._build_groups(intent, stages)
        
        # Step 9: 组装
        return MDTypedIR(
            task_type=intent.task_type,
            system=system_ir,
            force_field=ff_ir,
            cell=cell_ir,
            groups=groups,
            stages=stages,
            timestep=timestep,
            units="metal",
            observables=observables,
            outputs=self._build_outputs(intent),
            analysis=analysis,
            assumptions=intent.assumptions,
            provenance={"intent": intent},
        )
```

### 4.4 任务族定义

```python
# mdsynth/ir/task_families.py

@dataclass
class TaskFamily:
    task_type: str
    required_intent_fields: list[str]
    default_stages: list[dict]  # 协议骨架
    required_observables: list[str]
    required_analysis: list[str]
    default_parameters: dict
    risk_checks: list[str]

TASK_FAMILIES = {
    "structure_relaxation": TaskFamily(
        task_type="structure_relaxation",
        required_intent_fields=["material"],
        default_stages=[
            {"type": "energy_minimization", "id": "relax"},
        ],
        required_observables=["temperature", "potential_energy", "pressure"],
        required_analysis=["final_energy", "force_convergence"],
        default_parameters={},
        risk_checks=["minimization_not_converged"],
    ),
    
    "equilibration_nvt": TaskFamily(
        task_type="equilibration_nvt",
        required_intent_fields=["material", "temperature"],
        default_stages=[
            {"type": "energy_minimization", "id": "min"},
            {"type": "equilibration", "id": "equil", "ensemble": "nvt",
             "duration": {"value": 100, "unit": "ps"}},
        ],
        required_observables=["temperature", "potential_energy", "total_energy", 
                              "pressure", "volume"],
        required_analysis=["temperature_stability", "energy_conservation"],
        default_parameters={"tdamp": "100*timestep"},
        risk_checks=["temperature_not_equilibrated"],
    ),
    
    "equilibration_npt": TaskFamily(
        task_type="equilibration_npt",
        required_intent_fields=["material", "temperature"],
        default_stages=[
            {"type": "energy_minimization", "id": "min"},
            {"type": "equilibration", "id": "equil", "ensemble": "npt",
             "duration": {"value": 200, "unit": "ps"}},
        ],
        required_observables=["temperature", "potential_energy", "pressure", 
                              "volume", "density"],
        required_analysis=["temperature_stability", "pressure_stability", 
                          "volume_convergence"],
        default_parameters={"tdamp": "100*timestep", "pdamp": "1000*timestep",
                           "pressure": 0.0},
        risk_checks=["volume_collapse", "pressure_not_converged"],
    ),
    
    "uniaxial_tension": TaskFamily(
        task_type="uniaxial_tension",
        required_intent_fields=["material", "temperature", "deformation_axis"],
        default_stages=[
            {"type": "energy_minimization", "id": "min"},
            {"type": "equilibration", "id": "equil", "ensemble": "npt",
             "duration": {"value": 100, "unit": "ps"}},
            {"type": "deformation", "id": "loading", "ensemble": "nvt",
             "strain_rate": {"value": 1.0e8, "unit": "1/s"},
             "max_strain": {"value": 0.2, "unit": "dimensionless"}},
        ],
        required_observables=["temperature", "stress_tensor", "strain", "pressure"],
        required_analysis=["stress_strain_curve", "youngs_modulus_fit"],
        default_parameters={"strain_rate": 1.0e8, "max_strain": 0.2},
        risk_checks=["high_strain_rate", "size_effects", "loading_direction_barostat"],
    ),
    
    "thermal_expansion": TaskFamily(
        task_type="thermal_expansion",
        required_intent_fields=["material"],
        default_stages=[
            # 多温度点: 每个温度一个 equil + sample
            {"type": "temperature_sweep",
             "temperatures": [250, 300, 350],  # K
             "ensemble": "npt",
             "equil_duration": {"value": 100, "unit": "ps"},
             "sample_duration": {"value": 50, "unit": "ps"}},
        ],
        required_observables=["temperature", "volume", "density"],
        required_analysis=["thermal_expansion_coefficient_fit"],
        default_parameters={"temperature_range": 100, "num_points": 5},
        risk_checks=["insufficient_temperature_points", "volume_not_converged"],
    ),
}
```

### 4.5 关键默认值策略

```python
# mdsynth/ir/defaults.py

# 金属体系默认值 (units = metal)
METAL_DEFAULTS = {
    "timestep": 0.001,       # ps (= 1 fs)
    "tdamp_factor": 100,     # tdamp = timestep * tdamp_factor
    "pdamp_factor": 1000,    # pdamp = timestep * pdamp_factor
    "replication": (10, 10, 10),
    "neighbor_skin": 2.0,    # Å
    "temperature_damping": 0.1,  # ps (default for NVT/NPT)
    "pressure_damping": 1.0,     # ps (default for NPT)
    "strain_rate": 1.0e8,        # 1/s
    "max_strain": 0.2,
    "minimization_etol": 1.0e-6,
    "minimization_ftol": 1.0e-8,
    "equilibration_duration": 100,  # ps
    "production_duration": 200,     # ps
}

# 根据材料类别的 timestep 建议范围 (ps)
TIMESTEP_RECOMMENDATIONS = {
    "metal": {
        "min": 0.0005,   # 0.5 fs
        "max": 0.005,    # 5 fs
        "default": 0.001, # 1 fs
    },
    "ceramic": {
        "min": 0.0001,
        "max": 0.002,
        "default": 0.0005,
    },
}
```

---

## 5. 模块 3: 物理约束检查器

### 5.1 职责

在 MD Typed IR 进入 LAMMPS 编译器之前，系统性拦截不合法、不自洽、不可执行、明显不可信的模拟设计。

### 5.2 核心类型

```python
# mdsynth/validator/diagnostic.py

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    ERROR = "error"      # 阻塞编译，必须修复
    WARNING = "warning"  # 允许继续，写入证据包
    INFO = "info"        # 记录默认值和说明


class RepairPermission(str, Enum):
    """修复权限分级"""
    A = "A"  # 纯语法修复，可自动执行
    B = "B"  # 不改变物理意义的结构修复，可自动执行但记录 diff
    C = "C"  # 改变数值稳定性的修复，生成候选，必须重新验证
    D = "D"  # 改变力场/系综/科学目标，禁止静默执行
    E = "E"  # 缺乏物理依据，阻断生成，需要用户介入


@dataclass
class Diagnostic:
    """检查诊断结果"""
    rule_id: str
    severity: Severity
    message: str
    path: str  # IR 中的路径，如 "stages[0].ensemble.type"
    suggestion: Optional[str] = None
    evidence: dict = field(default_factory=dict)
    repair_permission: Optional[RepairPermission] = None
    repair_action: Optional[dict] = None
```

### 5.3 验证引擎

```python
# mdsynth/validator/engine.py

class PhysicsValidator:
    """
    可扩展的规则引擎。
    
    每个规则实现 check(ir) -> list[Diagnostic] 接口。
    引擎收集所有规则的结果并按严重程度分类。
    """
    
    def __init__(self, rules: list[Rule] = None):
        self.rules = rules or self._default_rules()
    
    def _default_rules(self) -> list[Rule]:
        return [
            # 完整性
            CheckStructureRequired(),
            CheckForceFieldRequired(),
            CheckTemperatureRequired(),
            # 边界条件
            CheckBarostatBoundaryCompatibility(),
            CheckDeformationAxisNotBarostatted(),
            # 力场
            CheckForceFieldCoversElements(),
            CheckAtomStyleCompatible(),
            # 积分器
            CheckNoDoubleIntegration(),
            CheckFixedGroupNotIntegrated(),
            # 可观测量
            CheckRequiredObservables(),
            CheckAnalysisInputsExist(),
            # 任务特定
            CheckThermalExpansionHasSweep(),
            CheckTensileHasStopCondition(),
            # 数值
            CheckTimestepReasonable(),
            CheckDurationToStepsConversion(),
            CheckSystemSizeReasonable(),
        ]
    
    def validate(self, ir: MDTypedIR) -> list[Diagnostic]:
        diagnostics = []
        for rule in self.rules:
            try:
                result = rule.check(ir)
                diagnostics.extend(result)
            except Exception as e:
                diagnostics.append(Diagnostic(
                    rule_id=rule.rule_id,
                    severity=Severity.ERROR,
                    message=f"Rule execution failed: {e}",
                    path="",
                ))
        return diagnostics
    
    def has_blocking_errors(self, diagnostics: list[Diagnostic]) -> bool:
        return any(d.severity == Severity.ERROR for d in diagnostics)
```

### 5.4 MVP 必须实现的 15 条 ERROR 规则（详细定义）

每条规则如下，实现时必须严格遵循：

```
规则 1: STRUCTURE_REQUIRED
  - 检查: ir.system.structure 不为 None
  - 错误: "No structure source specified"
  - 修复权限: D (需要提供结构)
  - path: "system.structure"

规则 2: FORCE_FIELD_REQUIRED
  - 检查: ir.force_field 已定义且 family 非空
  - 错误: "No force field specified"
  - 修复权限: D (需要用户提供或确认力场)
  - path: "force_field"

规则 3: FORCE_FIELD_COVERS_ALL_SPECIES
  - 检查: ir.force_field.elements_covered 包含 ir.system.species 中的所有元素
  - 错误: "Force field does not cover element {X}"
  - 修复权限: D
  - path: "force_field.elements_covered"

规则 4: ATOM_STYLE_COMPATIBLE
  - 检查: atom_style 与体系需求匹配（metal 体系 atomic 是兼容的）
  - 错误: "atom_style incompatible with system topology"
  - 修复权限: B
  - path: "force_field.atom_style"

规则 5: BAROSTAT_PERIODIC_ONLY
  - 检查: 每个 NPT stage 的 pressure_control 只作用在 periodic 方向
  - 错误: "Pressure control enabled on {axis} but boundary is {type}"
  - 修复权限: C (可以自动关闭非周期方向控压)
  - path: "stages[{i}].ensemble.pressure_control.{axis}"

规则 6: NO_DOUBLE_INTEGRATION
  - 检查: 同一 stage 内没有两个积分器控制同一个 group
  - 错误: "Group {group} is integrated by multiple fixes"
  - 修复权限: C
  - path: "stages[{i}].ensemble.group"

规则 7: FIXED_GROUP_NOT_INTEGRATED
  - 检查: 被固定的 group 不被 thermostat/barostat 积分
  - 错误: "Fixed group {group} is being integrated"
  - 修复权限: C
  - path: "stages[{i}].ensemble.group"

规则 8: DEFORMATION_HAS_AXIS_AND_STOP
  - 检查: 变形 stage 有 axis、strain_rate、max_strain
  - 错误: "Deformation stage missing axis/strain_rate/max_strain"
  - 修复权限: C (可以用默认值补全)
  - path: "stages[{i}].deformation"

规则 9: DEFORMATION_NOT_BAROSTATTED_ON_SAME_AXIS
  - 检查: 拉伸/压缩方向没有同时被 NPT 控压
  - 错误: "Deformation axis {axis} also has pressure control"
  - 修复权限: C (自动关闭加载方向控压)
  - path: "stages[{i}].deformation.axis / stages[{j}].ensemble.pressure_control"

规则 10: REQUIRED_OBSERVABLES
  - 检查: task_type 对应的 required_observables 都存在
  - 错误: "Task {type} requires observables {missing}"
  - 修复权限: B (自动添加必需 observable)
  - path: "observables"

规则 11: ANALYSIS_INPUTS_EXIST
  - 检查: 每个 analysis step 的 inputs 对应的 observable 都存在
  - 错误: "Analysis {id} requires input {x} but no observable provides it"
  - 修复权限: B
  - path: "analysis[{i}]"

规则 12: MSD_HAS_TRACKED_GROUP (MVP 可能不涉及但保留接口)
  - 检查: diffusion 类任务的 tracked_species 已定义
  - 错误: "MSD requires a tracked species/group"
  - 修复权限: D

规则 13: RDF_HAS_PAIR_SELECTION (MVP 可能不涉及但保留接口)
  - 检查: RDF 分析指定了物种对
  - 错误: "RDF requires pair selection"
  - 修复权限: D

规则 14: THERMAL_EXPANSION_HAS_TEMPERATURE_SWEEP
  - 检查: thermal_expansion 任务至少 3 个不同温度点
  - 错误: "Thermal expansion requires ≥3 temperature points, got {n}"
  - 修复权限: C (可以自动生成温度点)
  - path: "stages"

规则 15: DURATION_CAN_CONVERT_TO_STEPS
  - 检查: duration 的物理时间和 timestep 一致（单位可比）
  - 错误: "Duration unit {u} incompatible with timestep unit {t}"
  - 修复权限: C
  - path: "stages[{i}].duration"
```

### 5.5 WARNING 规则

```
WARN 1: TIMESTEP_REASONABLE
  - 检查: timestep 在材料建议范围内
  - 警告: "Timestep {x} ps may be too large for {material}"

WARN 2: PRODUCTION_TIME_SUFFICIENT
  - 检查: 生产阶段至少 10000 步
  - 警告: "Production run may be too short for reliable statistics"

WARN 3: STRAIN_RATE_REASONABLE
  - 检查: strain_rate ≤ 1e10 1/s
  - 警告: "Strain rate is extremely high; MD results may differ from experiments"

WARN 4: SYSTEM_SIZE_REASONABLE
  - 检查: 原子数 ≥ 1000
  - 警告: "System size < 1000 atoms may have significant size effects"

WARN 5: FORCE_FIELD_CONFIDENCE
  - 检查: 力场验证状态
  - 警告: "Force field is user-supplied and unverified; results should be validated"
```

### 5.6 规则接口定义

```python
# mdsynth/validator/rules/__init__.py

from typing import Protocol, runtime_checkable

@runtime_checkable
class Rule(Protocol):
    rule_id: str
    
    def check(self, ir: "MDTypedIR") -> list[Diagnostic]:
        """执行检查，返回诊断结果列表"""
        ...
```

---

## 6. 模块 4: LAMMPS 确定性编译器

### 6.1 职责

将 MD Typed IR 编译为 LAMMPS 输入脚本。这一步是**纯确定性代码生成**，不涉及任何 LLM 调用。编译器理解 IR 语义并按照 LAMMPS 命令依赖 DAG 生成排序后的脚本。

### 6.2 LAMMPS Lowered IR

在生成最终文本之前，先产生一个接近 LAMMPS 命令的中间表示。

```python
# mdsynth/backend/lowered_ir.py

@dataclass
class LAMMPSCommand:
    """单条 LAMMPS 命令的 lowered 表示"""
    kind: str         # "units", "atom_style", "pair_style", "fix", "run", etc.
    args: list[str] = field(default_factory=list)
    comment: str = ""
    line_number: int = 0  # 生成时自动填充

@dataclass
class LAMMPSLoweredIR:
    """接近 LAMMPS 命令的中间表示"""
    commands: list[LAMMPSCommand] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

### 6.3 命令依赖 DAG 与拓扑排序

```python
# mdsynth/backend/dependency_graph.py

# 命令之间的 "must come before" 关系
# 格式: 后置命令 -> [前置命令列表]
COMMAND_DEPENDENCIES = {
    "atom_style":   ["units"],
    "boundary":     ["units"],
    "lattice":      ["units"],
    "region":       ["units"],
    "create_box":   ["units", "boundary", "region"],
    "create_atoms": ["create_box", "lattice"],
    "read_data":    ["units", "atom_style", "boundary"],
    "mass":         ["create_box"],  # 或 read_data 之后
    "pair_style":   ["units", "atom_style"],
    "pair_coeff":   ["pair_style", "mass"],
    "neighbor":     ["units"],
    "neigh_modify": ["neighbor"],
    "group":        ["create_atoms"],  # 或 read_data
    "region":       ["units"],
    "velocity":     ["create_atoms", "mass"],
    "fix":          ["create_atoms", "group"],
    "compute":      ["create_atoms", "group"],
    "variable":     [],
    "thermo":       ["units"],
    "thermo_style": ["thermo"],
    "dump":         ["create_atoms"],
    "timestep":     ["units"],
    "run":          ["timestep", "fix", "velocity"],
    "minimize":     ["create_atoms", "pair_coeff"],
    "write_restart":["run"],
    "unfix":        ["fix"],
}
```

### 6.4 LAMMPS 编译器实现规范

```python
# mdsynth/backend/compiler.py

class LAMMPSBackendCompiler:
    """
    将 MDTypedIR 编译为 LAMMPS 输入脚本。
    
    编译器流程:
    1. 展开 IR 的每个部分为 LAMMPSCommand 列表
    2. 检查每个命令的 style 是否在 target LAMMPS 中可用
    3. 根据依赖 DAG 进行拓扑排序
    4. 渲染为文本
    """
    
    def __init__(self, lammps_runtime: "LAMMPSRuntime" = None):
        self.runtime = lammps_runtime  # 用于查询可用的 packages/styles
    
    def compile(self, ir: MDTypedIR) -> tuple[str, LAMMPSLoweredIR]:
        """
        编译 IR 为 LAMMPS 输入脚本。
        
        Returns:
            (script_text, lowered_ir) 元组
        """
        commands = []
        
        # 按顺序生成命令
        commands.append(self._emit_clear())
        commands.extend(self._emit_header(ir))
        commands.extend(self._emit_system(ir))
        commands.extend(self._emit_forcefield(ir))
        commands.extend(self._emit_neighbor(ir))
        commands.extend(self._emit_groups_and_computes(ir))
        commands.extend(self._emit_protocol(ir))
        commands.extend(self._emit_output(ir))
        
        # 拓扑排序（确保依赖关系正确）
        sorted_commands = self._topological_sort(commands)
        
        # 构建 LoweredIR
        for i, cmd in enumerate(sorted_commands):
            cmd.line_number = i + 1
        
        lowered_ir = LAMMPSLoweredIR(commands=sorted_commands)
        
        # 渲染为文本
        script_text = self._render(lowered_ir)
        
        return script_text, lowered_ir
    
    def _emit_header(self, ir: MDTypedIR) -> list[LAMMPSCommand]:
        """生成 units, atom_style, boundary"""
        return [
            LAMMPSCommand(kind="units", args=[ir.units]),
            LAMMPSCommand(kind="atom_style", args=[ir.force_field.atom_style]),
            LAMMPSCommand(kind="boundary", args=[
                self._boundary_char(ir.cell.boundary.x),
                self._boundary_char(ir.cell.boundary.y),
                self._boundary_char(ir.cell.boundary.z),
            ]),
        ]
    
    def _emit_system(self, ir: MDTypedIR) -> list[LAMMPSCommand]:
        """生成体系构建命令"""
        if ir.system.structure.source == StructureSourceType.GENERATED:
            gen = ir.system.structure.generator
            lat_const = gen.lattice_constant.value
            
            return [
                LAMMPSCommand(kind="lattice", args=[
                    gen.lattice_type, str(lat_const)
                ]),
                LAMMPSCommand(kind="region", args=[
                    "simbox", "block",
                    "0", str(gen.replication[0]),
                    "0", str(gen.replication[1]),
                    "0", str(gen.replication[2]),
                ]),
                LAMMPSCommand(kind="create_box", args=["1", "simbox"]),
                LAMMPSCommand(kind="create_atoms", args=["1", "region", "simbox"]),
            ]
        elif ir.system.structure.source == StructureSourceType.FILE:
            return [
                LAMMPSCommand(kind="read_data", args=[ir.system.structure.file_path]),
            ]
        else:
            raise CompilerError("Unknown structure source")
    
    def _emit_forcefield(self, ir: MDTypedIR) -> list[LAMMPSCommand]:
        """生成力场相关命令"""
        cmds = []
        
        # mass
        for species in ir.system.species:
            if species.mass:
                cmds.append(LAMMPSCommand(
                    kind="mass",
                    args=["1", str(species.mass.value)],  # type 1 for single-species
                    comment=f"# {species.element}"
                ))
        
        # pair_style
        ff = ir.force_field
        cmds.append(LAMMPSCommand(
            kind="pair_style",
            args=[ff.pair_style] + ff.pair_style_args,
            comment=f"# {ff.family.value} potential"
        ))
        
        # pair_coeff
        for pf in ff.potential_files:
            cmds.append(LAMMPSCommand(
                kind="pair_coeff",
                args=["*", "*", pf.name],
                comment=f"# {pf.source}"
            ))
        
        return cmds
    
    def _emit_protocol(self, ir: MDTypedIR) -> list[LAMMPSCommand]:
        """生成模拟协议命令 (minimize, velocity, fix, run 等)"""
        cmds = []
        
        # timestep
        if ir.timestep:
            cmds.append(LAMMPSCommand(
                kind="timestep", args=[str(ir.timestep.value)]
            ))
        
        for stage in ir.stages:
            if stage.type == StageType.ENERGY_MINIMIZATION:
                cmds.extend(self._emit_minimization(stage))
            elif stage.type == StageType.EQUILIBRATION:
                cmds.extend(self._emit_equilibration(stage))
            elif stage.type == StageType.DEFORMATION:
                cmds.extend(self._emit_deformation(stage))
            elif stage.type == StageType.PRODUCTION:
                cmds.extend(self._emit_production(stage))
        
        return cmds
    
    def _emit_equilibration(self, stage: ProtocolStage) -> list[LAMMPSCommand]:
        """生成平衡阶段命令"""
        cmds = []
        
        # velocity
        if stage.ensemble and stage.ensemble.temperature:
            T = stage.ensemble.temperature.value
            cmds.append(LAMMPSCommand(
                kind="velocity",
                args=["all", "create", str(T), "4928459", 
                      "mom", "yes", "rot", "yes", "dist", "gaussian"],
            ))
        
        # fix
        fix_id = f"fx_{stage.id}"
        if stage.ensemble:
            cmds.extend(self._emit_ensemble_fix(fix_id, stage.ensemble))
        
        # run
        if stage.duration:
            nsteps = self._duration_to_steps(stage.duration, stage.get_timestep())
            cmds.append(LAMMPSCommand(kind="run", args=[str(nsteps)]))
        
        # unfix
        cmds.append(LAMMPSCommand(kind="unfix", args=[fix_id]))
        
        return cmds
    
    def _emit_ensemble_fix(self, fix_id: str, ensemble: EnsembleSpec) -> list[LAMMPSCommand]:
        """生成系综 fix 命令"""
        group = ensemble.group
        T = ensemble.temperature.value if ensemble.temperature else 300
        
        if ensemble.type == EnsembleType.NVE:
            return [LAMMPSCommand(kind="fix", args=[fix_id, group, "nve"])]
        
        elif ensemble.type == EnsembleType.NVT:
            tdamp = ensemble.tdamp.value if ensemble.tdamp else 0.1
            return [LAMMPSCommand(
                kind="fix",
                args=[fix_id, group, "nvt", "temp", str(T), str(T), str(tdamp)]
            )]
        
        elif ensemble.type == EnsembleType.NPT:
            tdamp = ensemble.tdamp.value if ensemble.tdamp else 0.1
            pdamp = ensemble.pdamp.value if ensemble.pdamp else 1.0
            
            # 构建 NPT 参数
            args = [fix_id, group, "npt", "temp", str(T), str(T), str(tdamp)]
            
            if ensemble.pressure_control:
                pc = ensemble.pressure_control
                # 根据控压方向选择 iso/aniso/x/y/z
                active_axes = [ax for ax in ["x", "y", "z"] 
                             if pc.get(ax) is not None]
                if len(active_axes) == 3:
                    args.append("iso")
                    args.append(str(pc["x"].value))
                    args.append(str(pc["x"].value))
                elif len(active_axes) == 0:
                    pass  # no pressure control, this should have been caught by validator
                else:
                    # 非各向同性控压
                    args.append("aniso")
                    for ax in ["x", "y", "z"]:
                        if ax in active_axes:
                            p_val = pc[ax].value if pc[ax] else 0.0
                            args.extend([str(p_val), str(p_val)])
                        else:
                            args.extend(["NULL", "NULL"])
                
                args.append(str(pdamp))
            
            return [LAMMPSCommand(kind="fix", args=args)]
        
        raise CompilerError(f"Unknown ensemble type: {ensemble.type}")
    
    def _emit_deformation(self, stage: ProtocolStage) -> list[LAMMPSCommand]:
        """生成变形阶段命令"""
        cmds = []
        deform = stage.deformation
        
        # thermostat fix
        therm_id = f"fx_{stage.id}_therm"
        if deform.thermostat:
            cmds.extend(self._emit_ensemble_fix(therm_id, deform.thermostat))
        
        # deform fix
        deform_id = f"fx_{stage.id}_deform"
        axis = deform.axis
        strain_rate = deform.strain_rate.value  # 1/s → 需要转换为 metal units
        
        # 应变率转换: 1/s → 1/ps (metal units 时间单位是 ps)
        # strain_rate 1/s = strain_rate * 1e-12 1/ps
        # 实际上在 metal units 中，时间就是 ps
        # 对于 LAMMPS fix deform，erate 的单位是 1/时间单位
        rate_in_lammps = strain_rate  # 1/s, 但 metal units 期望 1/ps
        
        cmds.append(LAMMPSCommand(
            kind="fix",
            args=[deform_id, "all", "deform", "1", axis, 
                  "erate", str(rate_in_lammps),
                  "units", "box"],
        ))
        
        # run until max_strain
        max_strain = deform.max_strain.value
        rate_per_step = rate_in_lammps * float(ir.timestep.value)  # strain per step
        nsteps = int(max_strain / rate_per_step) if rate_per_step > 0 else 100000
        cmds.append(LAMMPSCommand(kind="run", args=[str(nsteps)]))
        
        # unfix
        cmds.append(LAMMPSCommand(kind="unfix", args=[therm_id]))
        cmds.append(LAMMPSCommand(kind="unfix", args=[deform_id]))
        
        return cmds
    
    def _topological_sort(self, commands: list[LAMMPSCommand]) -> list[LAMMPSCommand]:
        """
        根据 COMMAND_DEPENDENCIES 对命令进行拓扑排序。
        保持尽可能接近原始顺序（稳定排序）。
        """
        # 实现 Kahn's algorithm
        # 如果两个命令没有依赖关系，保持原有相对顺序
        ...
    
    def _render(self, lowered_ir: LAMMPSLoweredIR) -> str:
        """将 LoweredIR 渲染为 LAMMPS 输入脚本文本"""
        lines = []
        lines.append("# ========================================")
        lines.append("# LAMMPS input script generated by MDSynth")
        lines.append(f"# Task type: {self.current_ir.task_type}")
        lines.append(f"# IR version: {self.current_ir.ir_version}")
        lines.append("# ========================================")
        lines.append("")
        
        for cmd in lowered_ir.commands:
            comment = f" {cmd.comment}" if cmd.comment else ""
            line = f"{cmd.kind} {' '.join(cmd.args)}{comment}"
            lines.append(line)
        
        return "\n".join(lines) + "\n"
```

### 6.5 生成的示例脚本

对于 "copper NPT equilibration at 300K, 0 bar for 200ps"：

```
# ========================================
# LAMMPS input script generated by MDSynth
# Task type: equilibration_npt
# IR version: 0.1.0
# ========================================

clear
units metal
atom_style atomic
boundary p p p

# ---------- system ----------
lattice fcc 3.615
region simbox block 0 10 0 10 0 10
create_box 1 simbox
create_atoms 1 region simbox

# ---------- force field ----------
mass 1 63.546  # Cu
pair_style eam  # eam potential
pair_coeff * * Cu_u3.eam  # builtin

# ---------- neighbor ----------
neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes

# ---------- protocol ----------
timestep 0.001
velocity all create 300 4928459 mom yes rot yes dist gaussian
fix fx_equil all npt temp 300 300 0.1 iso 0 0 1.0
run 200000
unfix fx_equil

# ---------- output ----------
thermo 100
thermo_style custom step temp pe ke etotal press vol lx ly lz density
write_restart final.restart
```

---

## 7. 模块 5: 沙箱预运行与数值诊断

### 7.1 职责

使用 LAMMPS Python API 在隔离环境中运行生成的脚本，提取数值诊断信息。

### 7.2 实现

```python
# mdsynth/sandbox/runner.py

import os
import tempfile
from lammps import lammps  # type: ignore


@dataclass
class PreflightReport:
    """沙箱预运行报告"""
    passed: bool = False
    script_compiles: bool = False
    
    # 基本信息
    num_atoms: int = 0
    total_charge: float = 0.0
    
    # 能量检查
    initial_energy_finite: bool = False
    initial_potential_energy: float = 0.0
    initial_kinetic_energy: float = 0.0
    
    # 力检查
    maximum_initial_force: float = 0.0
    force_finite: bool = False
    
    # 最小化检查
    minimization_converged: bool = False
    final_energy: float = 0.0
    final_force_max: float = 0.0
    
    # 短运行检查
    short_run_completed: bool = False
    temperature_stable: bool = False
    energy_drift_percent: float = 0.0
    volume_change_percent: float = 0.0
    
    # 错误/警告
    lost_atoms: int = 0
    nan_detected: bool = False
    error_messages: list[str] = field(default_factory=list)
    warning_messages: list[str] = field(default_factory=list)
    
    # 时间序列（短运行的前 100 步）
    thermo_history: list[dict] = field(default_factory=list)


class SandboxPreflightRunner:
    """
    使用 LAMMPS Python API 运行沙箱测试。
    
    测试步骤:
    1. 0-step run: 检查构建是否成功
    2. 能量计算: 检查初始能量是否有限
    3. 最小化: 如果协议包含最小化
    4. 短 NVE: 100 步，检查能量守恒
    5. 短系综运行: 200 步，检查温度/压力稳定性
    """
    
    def __init__(self, lmp_executable: str = ""):
        self.lmp = None
        self.work_dir = tempfile.mkdtemp(prefix="mdsynth_sandbox_")
    
    def run(self, script: str, ir: MDTypedIR) -> PreflightReport:
        """
        运行沙箱预检查。
        
        对每个阶段分别创建一个测试脚本：
        - 初始化部分（system + force field）
        - 0-step run
        - 短最小化（如果 IR 有 minimization stage）
        - 短平衡（100 steps）
        """
        report = PreflightReport()
        
        try:
            # Phase 1: 基础构建检查
            self._phase_build_check(script, report)
            
            if not report.script_compiles:
                return report
            
            # Phase 2: 初始能量/力检查
            self._phase_initial_check(ir, report)
            
            # Phase 3: 最小化（如果适用）
            if self._has_minimization(ir):
                self._phase_minimization(ir, report)
            
            # Phase 4: 短 NVE 检查
            self._phase_short_nve(ir, report)
            
            # Phase 5: 短目标系综检查
            self._phase_short_ensemble(ir, report)
            
        except Exception as e:
            report.error_messages.append(str(e))
        
        finally:
            self._cleanup()
        
        # 综合判断
        report.passed = (
            report.script_compiles
            and report.initial_energy_finite
            and report.force_finite
            and not report.nan_detected
            and report.lost_atoms == 0
        )
        
        return report
    
    def _phase_build_check(self, script: str, report: PreflightReport):
        """0-step run 检查"""
        try:
            self.lmp = lammps()
            # 逐行执行，检查每行是否合法
            for line in script.strip().split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("run "):
                    # 先把 run 替换为 run 0
                    self.lmp.command("run 0")
                else:
                    self.lmp.command(line)
            report.script_compiles = True
        except Exception as e:
            report.error_messages.append(f"Build check failed: {e}")
            report.script_compiles = False
    
    def _phase_initial_check(self, ir: MDTypedIR, report: PreflightReport):
        """检查初始能量和力"""
        try:
            self.lmp.command("run 0")
            
            natoms = self.lmp.get_natoms()
            pe = self.lmp.get_thermo("pe")
            
            report.num_atoms = natoms
            report.initial_potential_energy = pe
            report.initial_energy_finite = not (np.isnan(pe) or np.isinf(pe))
            report.nan_detected = np.isnan(pe) or np.isinf(pe)
            
        except Exception as e:
            report.error_messages.append(f"Initial check failed: {e}")
    
    # ... (其他 phase 方法类似实现)
```

---

## 8. 模块 6: 受约束自动修复

### 8.1 职责

当物理约束检查器或沙箱预运行发现问题时，生成最小修复动作并应用到 MD Typed IR。

### 8.2 修复权限分级

```
A类：纯语法修复 — 自动执行
  - 调整命令顺序
  - 修正拼写错误
  
B类：不改变物理意义的结构修复 — 自动执行，记录 diff
  - 添加缺失的 thermo 输出
  - 添加缺失的 observable
  - 修正 atom_style 为兼容值

C类：改变数值稳定性的修复 — 生成候选，重新验证
  - 减小 timestep
  - 增加平衡时间
  - 降低应变率
  - 关闭非周期方向控压

D类：改变力场/系综/科学目标 — 禁止静默执行
  - 更换力场
  - 从 NVE 改为 NVT
  - 更换材料系统

E类：缺乏物理依据 — 阻断生成
  - 猜测缺失的 pair coefficient
  - 编造力场参数
```

### 8.3 实现

```python
# mdsynth/repair/engine.py

@dataclass
class RepairAction:
    """一次修复动作"""
    id: str
    permission: RepairPermission
    target_path: str          # IR 路径
    action_type: str          # "set_value", "multiply", "add_field", "remove_field"
    old_value: Any = None
    new_value: Any = None
    reason: str = ""
    confidence: str = "high"


class ConstrainedRepairEngine:
    """
    受约束修复引擎。
    
    原则:
    1. 永远修改 IR，不修改 LAMMPS 脚本
    2. 每次只做一个最小修改
    3. 修改后重新运行全部验证
    4. A/B 类自动执行，C 类生成候选，D/E 类请求干预
    """
    
    def __init__(self, max_auto_rounds: int = 3):
        self.max_auto_rounds = max_auto_rounds
        self.history: list[RepairAction] = []
    
    def generate_actions(
        self, ir: MDTypedIR, errors: list[Diagnostic]
    ) -> list[RepairAction]:
        """根据诊断结果生成修复动作"""
        actions = []
        
        for error in errors:
            if error.repair_permission in (RepairPermission.A, RepairPermission.B):
                # 自动修复
                action = self._auto_repair(ir, error)
                if action:
                    actions.append(action)
            elif error.repair_permission == RepairPermission.C:
                # 生成候选，记录
                action = self._candidate_repair(ir, error)
                if action:
                    action.confidence = "medium"
                    actions.append(action)
            elif error.repair_permission in (RepairPermission.D, RepairPermission.E):
                # 不自动修复，上报
                actions.append(RepairAction(
                    id=f"blocked_{error.rule_id}",
                    permission=error.repair_permission,
                    target_path=error.path,
                    action_type="blocked",
                    reason=error.message,
                    confidence="low",
                ))
        
        return actions
    
    def apply_actions(self, ir: MDTypedIR, actions: list[RepairAction]) -> MDTypedIR:
        """将修复动作应用到 IR"""
        for action in actions:
            if action.action_type == "blocked":
                continue
            if action.permission in (RepairPermission.A, RepairPermission.B, RepairPermission.C):
                ir = self._apply_action(ir, action)
                self.history.append(action)
        
        return ir
    
    def _apply_action(self, ir: MDTypedIR, action: RepairAction) -> MDTypedIR:
        """将单个修复动作应用到 IR"""
        # 使用 path 字符串定位 IR 中的字段
        # 例如 "stages[0].duration.value"
        parts = action.target_path.split(".")
        obj = ir
        for part in parts[:-1]:
            if "[" in part:
                name, idx = part.split("[")
                idx = int(idx.rstrip("]"))
                obj = getattr(obj, name)[idx]
            else:
                obj = getattr(obj, part)
        
        if action.action_type == "set_value":
            setattr(obj, parts[-1], action.new_value)
        elif action.action_type == "multiply":
            current = getattr(obj, parts[-1])
            setattr(obj, parts[-1], current * action.new_value)
        
        return ir
```

### 8.4 错误分类器

```python
# mdsynth/repair/classifier.py

# LAMMPS 常见错误 → IR 问题映射

LAMMPS_ERROR_MAP = {
    "All pair coeffs are not set": {
        "ir_path": "force_field",
        "rule_id": "FORCE_FIELD_COVERAGE_INCOMPLETE",
        "repair_permission": "D",
    },
    "Lost atoms": {
        "ir_path": "protocol.stages[*].timestep",
        "rule_id": "TIMESTEP_TOO_LARGE",
        "repair_permission": "C",
        "suggested_action": "multiply",
        "suggested_factor": 0.5,
    },
    "Bond atoms missing": {
        "ir_path": "force_field.atom_style",
        "rule_id": "ATOM_STYLE_INCOMPATIBLE",
        "repair_permission": "D",
    },
    "Non-numeric atom coords": {
        "ir_path": "system.structure",
        "rule_id": "STRUCTURE_CORRUPTED",
        "repair_permission": "E",
    },
    "Unknown command": {
        "ir_path": "lammps_script",
        "rule_id": "UNKNOWN_COMMAND",
        "repair_permission": "A",
    },
}
```

---

## 9. 模块 7: 脚本 + 证据包 + 可复现包

### 9.1 输出目录结构

```
simulation_package/
├── in.main.lammps           # 主 LAMMPS 输入脚本
├── system.data              # 结构数据文件（如果适用）
├── forcefield/              # 势函数文件
│   └── Cu_u3.eam
├── md_ir.yaml               # MD Typed IR（序列化）
├── intent_spec.yaml         # Scientific Intent Spec
├── forcefield_manifest.yaml # 力场声明
├── validation_report.json   # 验证报告
├── preflight_report.json    # 沙箱预运行报告
├── semantic_diff.md         # 修复历史（如有）
├── provenance.lock          # 溯源锁文件
├── assumptions.md           # 假设和风险说明
├── run.sh                   # 运行脚本
├── environment.yml          # 环境信息
└── README.md                # 人类可读说明
```

### 9.2 证据包内容

```python
# mdsynth/evidence/builder.py

@dataclass
class ValidationReport:
    script_compiles: bool = False
    runtime_compatible: bool = False
    force_field_provenance: str = "unverified"
    static_checks: dict = field(default_factory=lambda: {
        "passed": 0,
        "warnings": 0,
        "failed": 0,
    })
    preflight: dict = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    repair_history: list[dict] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)


class EvidencePackageBuilder:
    """
    组装最终输出包。
    
    将所有中间产物（Intent、IR、验证报告、预运行报告、修复历史）
    打包为一个可复现的目录结构。
    """
    
    def build(
        self,
        intent_spec: ScientificIntentSpec,
        md_ir: MDTypedIR,
        diagnostics: list[Diagnostic],
        lammps_script: str,
        preflight_report: PreflightReport,
        repair_history: list[RepairAction],
    ) -> MDSynthOutput:
        ...
```

### 9.3 provenance.lock 格式

```yaml
# provenance.lock
generated_by: mdsynth v0.1.0
generated_at: "2026-06-13T15:30:00Z"
user_request: "计算铜在 300K 下的拉伸模量，沿 x 方向拉伸"
target_lammps:
  version: "20260330"
  packages:
    - KSPACE
    - MANYBODY
  mpi: true
assumptions:
  - "使用 EAM 势函数 Cu_u3.eam"
  - "fcc 晶格常数 3.615 Å"
  - "晶胞复制 10×10×10"
repair_actions: []
force_field_provenance:
  family: eam
  source: builtin
  verification: verified_for_bulk_copper
limitations:
  - "MD 应变率 (1.0e8 1/s) 远高于实验应变率"
  - "系统尺寸 (4000 atoms) 可能有尺寸效应"
```

---

## 10. MVP 任务族定义

### 10.1 完整支持的 5 个任务

| # | 任务类型 | 需要的最小参数 | 生成内容 |
|---|----------|---------------|----------|
| 1 | `structure_relaxation` | material | 能量最小化脚本 |
| 2 | `equilibration_nvt` | material, temperature | 最小化 + NVT 平衡 |
| 3 | `equilibration_npt` | material, temperature | 最小化 + NPT 平衡 |
| 4 | `uniaxial_tension` | material, temperature, axis | 最小化 + NPT 平衡 + 单轴拉伸 |
| 5 | `thermal_expansion` | material | 多温度 NPT 模拟 |

### 10.2 支持的材料（内置数据）

| 材料 | 晶格 | 默认势函数 | 势函数文件 | 原子质量 |
|------|------|-----------|-----------|---------|
| Cu (copper) | fcc, 3.615 Å | EAM | Cu_u3.eam | 63.546 |
| Al (aluminum) | fcc, 4.05 Å | EAM | Al_mm.eam | 26.982 |
| Fe (iron) | bcc, 2.866 Å | EAM | Fe_mm.eam | 55.845 |
| Au (gold) | fcc, 4.078 Å | EAM | Au_u3.eam | 196.967 |
| W (tungsten) | bcc, 3.165 Å | EAM | W_mm.eam | 183.84 |
| Ni (nickel) | fcc, 3.52 Å | EAM | Ni_u3.eam | 58.693 |

---

## 11. LLM 调用规范

### 11.1 Prompt 管理

所有 prompt 模板集中在 `mdsynth/llm/prompts/` 目录。

```python
# mdsynth/llm/base.py

from abc import ABC, abstractmethod

class LLMBackend(ABC):
    """LLM 后端抽象基类"""
    
    @abstractmethod
    def generate_structured(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        output_schema: dict
    ) -> dict:
        """生成结构化输出（JSON）"""
        ...

# mdsynth/llm/openai_backend.py

import openai

class OpenAIBackend(LLMBackend):
    def __init__(self, model: str = "gpt-4o"):
        self.client = openai.OpenAI()
        self.model = model
    
    def generate_structured(self, system_prompt, user_prompt, output_schema):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "schema": output_schema,
                    "strict": True,
                }
            },
            temperature=0.1,  # 低温度以保证一致性
        )
        return json.loads(response.choices[0].message.content)
```

### 11.2 LLM 调用的两个场景

**场景 1: Intent Extraction**

```
System: 你是一个分子动力学科学意图提取器。
        从用户的自然语言中提取结构化的研究意图。
        只输出 JSON，不要解释。
        
        支持的任务类型：
        - structure_relaxation
        - equilibration_nvt
        - equilibration_npt
        - uniaxial_tension
        - thermal_expansion
        
        支持的材料：Cu, Al, Fe, Au, W, Ni (及其中文名/英文名)
        
        规则：
        - temperature 单位始终是 K
        - pressure 单位始终是 bar 或 atm
        - 只标记用户明确提供的信息为 known
        - 用户未提及的标记为 unknown，不要猜测

User: {user_request}

Output schema: {ScientificIntentSpec 的 JSON Schema}
```

**场景 2: IR Planning (仅在处理歧义时调用)**

大多数 IR Planning 通过确定性规则完成，LLM 仅用于：
- 用户提供了自定义需求但任务族无法精确匹配时
- 需要从多种候选方案中选择时

### 11.3 LLM 调用点总结

| 调用点 | 模块 | 频率 | 温度 |
|--------|------|------|------|
| 意图提取 | intent/extractor.py | 每次运行 1 次 | 0.1 |
| IR 规划辅助 | ir/planner.py | 仅在规则无法覆盖时 | 0.2 |
| 修复建议 | repair/engine.py | 仅在 C/D 类修复时 | 0.3 |

---

## 12. 错误处理与日志

### 12.1 异常体系

```python
class MDSynthError(Exception):
    """基础异常"""
    pass

class IntentExtractionError(MDSynthError):
    """意图提取失败"""
    pass

class UnsupportedTaskError(MDSynthError):
    """不支持的任务类型"""
    pass

class IRPlanningError(MDSynthError):
    """IR 规划失败"""
    pass

class ValidationError(MDSynthError):
    """IR 验证失败（阻塞性错误）"""
    pass

class CompilerError(MDSynthError):
    """LAMMPS 编译失败"""
    pass

class SandboxError(MDSynthError):
    """沙箱运行失败"""
    pass

class RepairExhaustedError(MDSynthError):
    """修复次数耗尽"""
    pass

class ForceFieldNotFoundError(MDSynthError):
    """找不到势函数文件"""
    pass
```

### 12.2 结构化日志

```python
# mdsynth/utils/logging.py

import structlog

logger = structlog.get_logger()

# 使用方式:
logger.info("pipeline_started", user_request=user_request)
logger.info("intent_extracted", task_type=spec.task_type, confidence=spec.extraction_confidence)
logger.info("ir_planned", num_stages=len(md_ir.stages))
logger.info("validation_complete", errors=n_errors, warnings=n_warnings)
logger.info("compilation_complete", script_lines=len(script.split("\n")))
logger.info("preflight_complete", passed=report.passed)
logger.info("repair_applied", actions=len(actions))
```

---

## 13. 测试策略

### 13.1 测试层次

```
unit/                          # 单元测试
├── test_intent_extractor.py   # 使用 mock LLM
├── test_ir_planner.py         # 验证 IR 生成正确性
├── test_validator_rules.py    # 每条规则独立测试
├── test_compiler_emitters.py  # 每个代码块测试
├── test_sandbox_runner.py     # mock LAMMPS 的测试
└── test_repair_engine.py      # 修复策略测试

integration/                   # 集成测试
├── test_pipeline_e2e.py       # 端到端测试 (mock LLM + mock LAMMPS)
├── test_copper_npt.py         # Cu NPT 完整流程
├── test_aluminum_tension.py   # Al 拉伸完整流程
└── test_thermal_expansion.py  # 热膨胀完整流程

fixtures/                      # 测试夹具
├── intent_copper_tension.yaml
├── ir_copper_npt.yaml
└── expected_scripts/          # 期望的 LAMMPS 脚本
    ├── copper_npt.in
    ├── aluminum_tension.in
    └── iron_thermal_expansion.in
```

### 13.2 关键测试用例

```python
# 每个规则至少 2 个测试: 应该通过 + 应该失败

class TestBarostatBoundaryRule:
    def test_npt_on_periodic_boundary_passes(self):
        """全周期边界 + NPT = 通过"""
        ...
    
    def test_npt_on_nonperiodic_z_fails(self):
        """z 非周期 + NPT 对 z 控压 = ERROR"""
        ...
    
    def test_nvt_no_pressure_control_ok(self):
        """NVT 不需要 pressure control"""
        ...

class TestRequiredObservablesRule:
    def test_tension_missing_stress_fails(self):
        """拉伸任务缺 stress_tensor = ERROR"""
        ...
    
    def test_tension_all_observables_passes(self):
        """拉伸任务有所有必需的 observables = 通过"""
        ...
```

### 13.3 集成测试列表

1. **copper_npt_equilibration**: "铜在 300K 0bar 下 NPT 平衡 200ps" → 验证生成脚本可运行
2. **copper_uniaxial_tension**: "铜在 300K 下沿 x 拉伸" → 验证包含 deform 命令
3. **aluminum_thermal_expansion**: "铝的热膨胀系数" → 验证多温度点
4. **iron_nvt_equilibration**: "铁在 500K NVT 平衡" → 验证 NVT 正确生成
5. **error_missing_temperature**: "铜的拉伸" → 验证缺失温度时报错
6. **error_nonperiodic_barostat**: 薄膜 z 非周期 + NPT z 控压 → 验证检查器拦截

---

## 14. 项目目录结构

```
mdsynth/
├── __init__.py                  # 包入口，导出 MDSynthPipeline
├── pipeline.py                  # 主流水线
├── config.py                    # 全局配置
│
├── intent/                      # 科学意图提取
│   ├── __init__.py
│   ├── spec.py                  # ScientificIntentSpec 数据类
│   ├── extractor.py             # 基于 LLM 的提取器
│   └── taxonomy.py              # 任务和材料分类
│
├── ir/                          # MD Typed IR
│   ├── __init__.py
│   ├── md_ir.py                 # 完整 IR 数据类
│   ├── planner.py               # Intent→IR 规划器
│   ├── task_families.py         # 任务族定义
│   └── defaults.py              # 默认值策略
│
├── validator/                   # 物理约束检查器
│   ├── __init__.py
│   ├── engine.py                # PhysicsValidator 引擎
│   ├── diagnostic.py            # Diagnostic 类型
│   ├── rules/
│   │   ├── __init__.py
│   │   ├── completeness.py      # 规则 1-3
│   │   ├── boundary.py          # 规则 5, 9
│   │   ├── force_field.py       # 规则 3, 4
│   │   ├── ensemble.py          # 规则 6, 7
│   │   ├── observables.py       # 规则 10, 11
│   │   └── numerics.py          # 规则 15 + WARNING 规则
│   └── repair_hints.py          # 修复提示
│
├── backend/                     # LAMMPS 确定性编译器
│   ├── __init__.py
│   ├── compiler.py              # 主编译器
│   ├── lowered_ir.py            # LAMMPSLoweredIR
│   ├── dependency_graph.py      # 依赖 DAG + 拓扑排序
│   ├── units.py                 # 单位系统
│   └── emitters/
│       ├── __init__.py
│       ├── header.py
│       ├── system.py
│       ├── forcefield.py
│       ├── neighbor.py
│       ├── protocol.py
│       └── output.py
│
├── sandbox/                     # 沙箱预运行
│   ├── __init__.py
│   ├── runner.py                # SandboxPreflightRunner
│   ├── diagnostics.py           # 数值诊断器
│   └── report.py                # PreflightReport
│
├── repair/                      # 自动修复
│   ├── __init__.py
│   ├── engine.py                # ConstrainedRepairEngine
│   ├── classifier.py            # 错误分类器
│   ├── strategies.py            # 修复策略库
│   └── permissions.py           # 权限分级
│
├── evidence/                    # 证据包
│   ├── __init__.py
│   ├── builder.py               # EvidencePackageBuilder
│   ├── manifest.py              # ProvenanceManifest
│   └── report_templates.py      # 报告模板
│
├── knowledge/                   # 确定性知识库
│   ├── __init__.py
│   ├── lammps_commands.py       # 命令本体
│   ├── force_fields.py          # 力场注册表
│   ├── materials.py             # 材料-势函数映射
│   └── heuristics.py            # 经验规则
│
├── llm/                         # LLM 抽象层
│   ├── __init__.py
│   ├── base.py                  # LLMBackend 基类
│   ├── openai_backend.py        # OpenAI 实现
│   └── prompts/
│       ├── __init__.py
│       ├── intent_extraction.py  # 意图提取 prompt
│       └── ir_planning.py        # IR 规划 prompt
│
└── utils/                       # 工具
    ├── __init__.py
    ├── quantities.py            # Quantity / QuantityRange / Dimension
    ├── serialization.py         # YAML/JSON 序列化
    └── logging.py               # 结构化日志
```

### 14.1 依赖

```toml
# pyproject.toml
[project]
name = "mdsynth"
version = "0.1.0"
description = "AI-native MD simulation experiment compiler for LAMMPS"
requires-python = ">=3.10"
dependencies = [
    "openai>=1.0.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "structlog>=23.0.0",
    "numpy>=1.24.0",
    # LAMMPS Python module 需要单独安装，
    # 不在 PyPI 依赖中，由运行环境提供
]

[project.optional-dependencies]
test = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
]
```

---

## 15. 实现顺序

### Phase 1: 基础数据层 (最先实现)

**目标**: 所有模块依赖的核心数据结构就位

| 步骤 | 文件 | 内容 |
|------|------|------|
| 1.1 | `utils/quantities.py` | `Quantity`, `QuantityRange`, `Dimension`, `BoundaryType` |
| 1.2 | `utils/serialization.py` | YAML/JSON 序列化工具 |
| 1.3 | `utils/logging.py` | 结构化日志配置 |
| 1.4 | `intent/spec.py` | `ScientificIntentSpec` 及所有子类型 |
| 1.5 | `intent/taxonomy.py` | 任务分类法 + 材料表 |
| 1.6 | `ir/md_ir.py` | `MDTypedIR` 及所有子类型 (SystemIR, Stage, Ensemble, etc.) |
| 1.7 | `knowledge/materials.py` | 6 种金属的材料数据 |
| 1.8 | `knowledge/force_fields.py` | EAM 势函数注册表 |
| 1.9 | `knowledge/heuristics.py` | timestep 建议、默认值表 |

### Phase 2: 验证与编译 (核心确定性子系统)

**目标**: 不依赖 LLM 的核心逻辑跑通

| 步骤 | 文件 | 内容 |
|------|------|------|
| 2.1 | `validator/diagnostic.py` | `Diagnostic`, `Severity`, `RepairPermission` |
| 2.2 | `validator/engine.py` | `PhysicsValidator` 引擎 + `Rule` Protocol |
| 2.3 | `validator/rules/completeness.py` | 规则 1-3 |
| 2.4 | `validator/rules/boundary.py` | 规则 5, 9 |
| 2.5 | `validator/rules/force_field.py` | 规则 3, 4 |
| 2.6 | `validator/rules/ensemble.py` | 规则 6, 7 |
| 2.7 | `validator/rules/observables.py` | 规则 10, 11, 12, 13, 14 |
| 2.8 | `validator/rules/numerics.py` | 规则 15 + WARNING 规则 |
| 2.9 | `backend/lowered_ir.py` | `LAMMPSLoweredIR`, `LAMMPSCommand` |
| 2.10 | `backend/dependency_graph.py` | 依赖 DAG 定义 + 拓扑排序 |
| 2.11 | `backend/units.py` | metal units 单位处理 |
| 2.12 | `backend/emitters/header.py` | units, atom_style, boundary |
| 2.13 | `backend/emitters/system.py` | lattice, region, create_box, create_atoms |
| 2.14 | `backend/emitters/forcefield.py` | mass, pair_style, pair_coeff |
| 2.15 | `backend/emitters/neighbor.py` | neighbor, neigh_modify |
| 2.16 | `backend/emitters/protocol.py` | minimize, velocity, fix, run, unfix |
| 2.17 | `backend/emitters/output.py` | thermo, thermo_style, dump, restart |
| 2.18 | `backend/compiler.py` | 主编译器（组装所有 emitter） |

### Phase 3: LLM 集成

**目标**: 端到端流水线可运行

| 步骤 | 文件 | 内容 |
|------|------|------|
| 3.1 | `llm/base.py` | `LLMBackend` 抽象基类 |
| 3.2 | `llm/openai_backend.py` | OpenAI GPT-4o 实现 |
| 3.3 | `llm/prompts/intent_extraction.py` | Intent extraction prompt 模板 |
| 3.4 | `intent/extractor.py` | `ScientificIntentExtractor` |
| 3.5 | `ir/task_families.py` | 5 个任务族定义 |
| 3.6 | `ir/defaults.py` | 默认值策略 |
| 3.7 | `ir/planner.py` | `MDTypedIRPlanner` |

### Phase 4: 沙箱与修复

**目标**: 自动验证和修复闭环

| 步骤 | 文件 | 内容 |
|------|------|------|
| 4.1 | `sandbox/report.py` | `PreflightReport` |
| 4.2 | `sandbox/diagnostics.py` | 数值诊断函数 |
| 4.3 | `sandbox/runner.py` | `SandboxPreflightRunner` (LAMMPS Python API) |
| 4.4 | `repair/permissions.py` | 修复权限分级 |
| 4.5 | `repair/classifier.py` | 错误分类器 + LAMMPS 错误映射 |
| 4.6 | `repair/strategies.py` | 修复策略库 |
| 4.7 | `repair/engine.py` | `ConstrainedRepairEngine` |

### Phase 5: 证据包与流水线

**目标**: 完整可交付系统

| 步骤 | 文件 | 内容 |
|------|------|------|
| 5.1 | `evidence/report_templates.py` | 报告模板 |
| 5.2 | `evidence/manifest.py` | `ProvenanceManifest` |
| 5.3 | `evidence/builder.py` | `EvidencePackageBuilder` |
| 5.4 | `config.py` | 全局配置 |
| 5.5 | `pipeline.py` | `MDSynthPipeline`（主入口） |
| 5.6 | `__init__.py` | 公开 API 导出 |

### Phase 6: 测试

**目标**: 验证系统正确性

| 步骤 | 内容 |
|------|------|
| 6.1 | 单元测试（每模块 ≥80% 覆盖率） |
| 6.2 | 集成测试（5 个任务族的 E2E 测试） |
| 6.3 | 边界条件测试（无效输入、极端值） |
| 6.4 | 回归测试夹具生成 |

---

## 附录 A: 术语对照

| 术语 | 英文 | 定义 |
|------|------|------|
| 科学意图说明书 | Scientific Intent Spec | 从自然语言提取的结构化研究目标 |
| 类型化分子动力学中间表示 | MD Typed IR | 带类型系统的模拟实验设计图 |
| 物理约束检查器 | PhysicsConstraintValidator | 检查 IR 物理/逻辑正确性的规则引擎 |
| LAMMPS 确定性编译器 | LAMMPSBackendCompiler | 将 IR 编译为 LAMMPS 脚本的纯代码生成器 |
| LAMMPS Lowered IR | LAMMPSLoweredIR | 接近 LAMMPS 命令的中间表示 |
| 沙箱预运行 | SandboxPreflight | 在隔离环境用小规模运行验证脚本 |
| 受约束自动修复 | ConstrainedRepairEngine | 按权限分级修改 IR 的修复系统 |
| 证据包 | EvidencePackage | 脚本 + 溯源 + 验证报告 + 假设 + 风险 |
| 诊断 | Diagnostic | 单条检查结果（含严重程度和修复建议） |

## 附录 B: 关键的 "不要做"

1. **不要让 LLM 直接输出 LAMMPS 命令** — LLM 只输出结构化方案，代码生成由编译器完成
2. **不要解析任意 LAMMPS 脚本** — 只生成自己保证的规范脚本
3. **不要用 LLM 重写脚本进行修复** — 修复总是改 IR 然后重新编译
4. **不要静默编造势函数参数** — 缺失参数是 D/E 类阻断
5. **不要把 LAMMPS 命令作为用户接口** — 用户只需要说科学目标
6. **不要覆盖 100% LAMMPS 命令** — MVP 只覆盖金属 EAM 体系的必要命令子集
7. **不要忽略 provenance** — 每个参数都要标记来源（用户/默认/推断/修复）
8. **不要在未经沙箱预运行的情况下声称脚本可用** — 每个脚本必须至少在沙箱中跑过 0-step + 短 NVE

---

*本文档为 MDSynth v0.1.0 的完整开发规格说明书。Claude Code 应严格按照本文档的模块划分、数据结构、接口定义和实现顺序进行开发。任何与本文档冲突的实现决策，应优先以本文档为准。*
