# MDSynth v0.1.0

AI-native MD simulation experiment compiler for LAMMPS.

将自然语言研究目标自动生成为可运行、物理可信、可复现的 LAMMPS 输入脚本及完整证据包。

---

## 快速开始

### 安装

```bash
pip install -e .
```

依赖: `openai`, `pyyaml`, `structlog`, `numpy`（详见 `pyproject.toml`）。

### 5 行代码运行

```python
from mdsynth import MDSynthPipeline
from mdsynth.llm.base import MockLLMBackend

pipeline = MDSynthPipeline(llm_backend=MockLLMBackend())
output = pipeline.run("铜在300K下NPT平衡200ps")
pipeline.save(output, "./output/")
```

### 使用真实 GPT-4o

```python
from mdsynth import MDSynthPipeline
from mdsynth.llm.openai_backend import OpenAIBackend

llm = OpenAIBackend(model="gpt-4o", api_key="sk-xxx")
pipeline = MDSynthPipeline(llm_backend=llm)
output = pipeline.run("计算铜在300K下的拉伸模量，沿x方向拉伸")
pipeline.save(output, "./output/")

# 查看生成的 LAMMPS 脚本
print(output.lammps_script)
```

---

## MVP 支持范围

### 5 种任务类型

| 任务 | 中文 | 最少参数 | 生成内容 |
|------|------|----------|----------|
| `structure_relaxation` | 结构弛豫 | 材料 | 能量最小化脚本 |
| `equilibration_nvt` | NVT 平衡 | 材料 + 温度 | 最小化 + NVT |
| `equilibration_npt` | NPT 平衡 | 材料 + 温度 | 最小化 + NPT |
| `uniaxial_tension` | 单轴拉伸 | 材料 + 温度 + 方向 | 最小化 + NPT平衡 + 变形 |
| `thermal_expansion` | 热膨胀系数 | 材料 | 多温度 NPT |

### 6 种内置材料

| 材料 | 晶格 | 晶格常数 (Å) | 默认势函数 |
|------|------|-------------|-----------|
| Cu | fcc | 3.615 | Cu_u3.eam |
| Al | fcc | 4.05 | Al_mm.eam |
| Fe | bcc | 2.866 | Fe_mm.eam |
| Au | fcc | 4.078 | Au_u3.eam |
| W | bcc | 3.165 | W_mm.eam |
| Ni | fcc | 3.52 | Ni_u3.eam |

---

## 输入示例

支持**中文**和**英文**输入：

```python
# 中文
"铜在300K下NPT平衡200ps"
"计算铝在500K下的拉伸模量，沿x方向拉伸"
"铁的热膨胀系数"
"金的结构弛豫"

# 英文
"copper NPT equilibration at 300K, 0 bar for 200ps"
"uniaxial tension of aluminum at 500K along x"
"thermal expansion of tungsten"
"structure relaxation of nickel"
```

---

## 输出包结构

运行后 `./output/` 目录包含：

```
output/
├── in.main.lammps           # 主 LAMMPS 输入脚本
├── md_ir.yaml               # MD Typed IR（可机器读取）
├── intent_spec.yaml         # Scientific Intent Spec
├── validation_report.json   # 验证报告
├── preflight_report.json    # 沙箱预运行报告
├── provenance.lock          # 溯源锁文件
├── assumptions.md           # 假设和风险说明
├── semantic_diff.md         # 修复历史（如有）
├── forcefield_manifest.yaml # 力场声明
└── README.md                # 人类可读说明
```

---

## 架构概览

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
│  MDTypedIRPlanner               │  ← 规则引擎
│  输出: MDTypedIR                │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  PhysicsConstraintValidator     │  ← 纯规则引擎 (15条规则)
│  输出: list[Diagnostic]         │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  LAMMPSBackendCompiler          │  ← 确定性代码生成
│  输出: LAMMPS 输入脚本          │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  SandboxPreflightRunner         │  ← LAMMPS Python API
│  输出: PreflightReport          │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  EvidencePackageBuilder         │  ← 纯组装器
│  输出: 脚本 + 证据包            │
└─────────────────────────────────┘
```

---

## 运行测试

```bash
# 全部测试
pytest tests/ -v

# 仅单元测试
pytest tests/unit/ -v

# 含覆盖率
pytest tests/ --cov=mdsynth -v
```

---

## 设计原则

1. **LLM 只负责理解意图** — 不直接生成 LAMMPS 命令
2. **所有 LAMMPS 知识都是确定性的** — 命令模板、参数约束由代码保证
3. **先检查后生成** — 物理冲突在 IR 层发现
4. **修复改 IR，不改脚本** — 自动修复总是修改 IR 然后重新编译
5. **每个决策都有证据** — 脚本携带来源、假设、验证结果
