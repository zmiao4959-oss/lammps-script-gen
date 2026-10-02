# MDSynth 学习指南

> 目标：从零开始，循序渐进地读懂这个由 AI 生成的项目

---

## 核心思想：顺着数据流读

这个项目本质上是一条**数据流水线**。你不需要一次性读懂所有代码，只需要顺着数据流动的方向，一段一段地理解：

```
用户中文输入
    │
    ▼
┌──────────────────────┐
│ ① 提取科学意图       │  ← LLM 调用（唯一一次）
│   → IntentSpec       │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ ② 规划实验设计       │  ← 纯规则引擎
│   → MDTypedIR        │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ ③ 物理约束检查       │  ← 15 条规则
│   → Diagnostic[]     │  ← 出错了进入修复
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ ④ 编译为 LAMMPS 脚本 │  ← 纯代码生成
│   → in.main.lammps   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ ⑤ 沙箱测试 + 打包    │  ← 运行 + 证据
│   → 输出目录         │
└──────────────────────┘
```

**每读完一步，你应该能回答："这一步的输入是什么？输出是什么？谁调用了它？"**

---

## 第一阶段：跑起来（30 分钟）

**目标**：不看代码，先看它怎么工作

### 1.1 用 Mock 模式运行

```python
# 创建 run_mock.py
from mdsynth import MDSynthPipeline
from mdsynth.llm.base import MockLLMBackend

pipeline = MDSynthPipeline(llm_backend=MockLLMBackend())
output = pipeline.run("铜在300K下NPT平衡200ps")
pipeline.save(output, "./output/")

# 打印生成的 LAMMPS 脚本
print(output.lammps_script)
```

### 1.2 查看输出

打开 `./output/` 目录，浏览这些文件：

| 文件 | 关注什么 |
|------|----------|
| `in.main.lammps` | 生成的 LAMMPS 脚本——这是最终产物 |
| `intent_spec.yaml` | 系统从你的中文里提取了什么 |
| `md_ir.yaml` | 中间表示——这是整个系统的核心 |
| `provenance.lock` | 溯源：每个决策的来源 |
| `validation_report.json` | 哪些检查通过了，哪些没通过 |

### 1.3 换个输入试试

```python
output = pipeline.run("铁在500K下x方向拉伸")
pipeline.save(output, "./output_iron/")
```

**到这步你理解了**：用户输入 → 系统自动输出完整的 LAMMPS 脚本 + 证据包。

---

## 第二阶段：数据结构（1 小时）

**目标**：读懂系统最核心的两个数据类——它们是流水线的"语言"

### 2.1 ScientificIntentSpec — 用户想要什么

**阅读文件**：`mdsynth/intent/spec.py`

关键问题：
- `task_type` 有哪些可能的值？（看 `taxonomy.py`）
- `temperature` 为什么不是简单的数字，而是 `QuantitySpec`？
- `FieldStatus` 的 KNOWN / UNKNOWN / DEFAULTED / INFERRED 分别什么意思？
- `missing_blocking` 和 `missing_defaultable` 的区别是什么？

### 2.2 MDTypedIR — 实验怎么做

**阅读文件**：`mdsynth/ir/md_ir.py`

这是整个项目**最重要的文件**。按顺序读：

| 读的顺序 | 数据类 | 它表达什么 |
|----------|--------|-----------|
| 1 | `SystemIR` | 模拟什么体系（材料、结构） |
| 2 | `ForceFieldIR` | 用什么力场 |
| 3 | `CellIR` | 盒子边界条件 |
| 4 | `ProtocolStage` | 模拟分几个阶段 |
| 5 | `EnsembleSpec` | 每个阶段的系综（NVT/NPT） |
| 6 | `DeformationSpec` | 变形阶段怎么拉 |
| 7 | `MDTypedIR` | 顶层——把所有东西装在一起 |

关键问题：
- `MDTypedIR` 里有没有任何 LAMMPS 命令？（答案：没有——这就是设计原则 #1）
- `Quantity` 和普通的 `float` 有什么区别？（多了 unit、dimension、source）

### 2.3 对应关系验证

打开 `./output/md_ir.yaml`，对照上面读的数据类，找到对应的字段。你会发现 YAML 里的结构和你刚读的 dataclass 一一对应。

**到这步你理解了**：系统用两套"语言"描述一个 MD 模拟——IntentSpec（用户想干什么）和 MDTypedIR（实验怎么做），两者都不包含 LAMMPS 命令。

---

## 第三阶段：从 Intent 到 IR（1 小时）

**目标**：看懂规则引擎怎么把"用户想干什么"变成"实验怎么做"

### 3.1 任务族 — 每种任务的标准配方

**阅读文件**：`mdsynth/ir/task_families.py`

关键问题：
- 5 种任务类型各自的 `default_stages` 是什么？
- `structure_relaxation` 要几个阶段？`uniaxial_tension` 呢？
- `required_observables` 和 `required_analysis` 的作用是什么？

### 3.2 规划器 — 核心转换逻辑

**阅读文件**：`mdsynth/ir/planner.py`

这是项目里**最长的文件**，但结构很清晰。按 `plan()` 方法里的步骤顺序读：

```
plan()
  ├── _build_system()          ← 从材料名→晶格、晶胞复制
  ├── _build_force_field()     ← 从材料名→EAM 势函数文件
  ├── _build_cell()            ← 边界条件（默认周期性）
  ├── _build_stages()          ← 任务族→协议阶段列表
  │     ├── _build_minimization_stage()
  │     ├── _build_equilibration_stage()
  │     ├── _build_deformation_stage()
  │     └── _build_temperature_sweep_stages()
  ├── _determine_timestep()    ← 默认 0.001 ps (1 fs)
  ├── _build_observables()     ← 任务族要求跟踪什么量
  └── _build_outputs()         ← thermo/dump 输出配置
```

**建议读法**：不要逐行读。先看 `_build_stages()` 里 `energy_minimization` 分支，它最短。理解了这个模式后，其他分支同理。

### 3.3 默认值来自哪

**阅读文件**：`mdsynth/ir/defaults.py`

注意 `METAL_DEFAULTS` 字典——这些数字是整个系统"为什么不乱"的基础。

**到这步你理解了**：Intent → IR 的转换大部分是查表和填充默认值，不是 LLM 做的。LLM 只在第一步提取 Intent 时用了一次。

---

## 第四阶段：验证器 — 15 条物理规则（1 小时）

**目标**：理解系统怎么在运行前拦截错误

### 4.1 诊断和严重程度

**阅读文件**：`mdsynth/validator/diagnostic.py`

关注 `Severity` 的三个级别：ERROR（阻断）、WARNING（允许但记录）、INFO（记录默认值）。

### 4.2 验证引擎

**阅读文件**：`mdsynth/validator/engine.py`

`validate()` 方法做的事情：遍历所有规则，每个规则返回诊断列表，合并返回。非常简单。

### 4.3 读两条规则

**先读最简单的**：`mdsynth/validator/rules/completeness.py`

`CheckStructureRequired` 只有 5 行逻辑——如果 `ir.system.structure is None`，报错。这就是一条规则的全部内容。

**再读一条中等的**：`mdsynth/validator/rules/boundary.py`

`CheckBarostatBoundaryCompatibility`：遍历所有 stage，如果 NPT 的控压方向是 non-periodic，报错。这是物理知识（边界条件约束）编码为代码的典型例子。

### 4.4 剩下的规则看一眼标题就够了

| 文件 | 规则 |
|------|------|
| `force_field.py` | 力场必须覆盖所有元素、atom_style 兼容 |
| `ensemble.py` | 不能双积分、固定原子不控温 |
| `observables.py` | 必需观测量不缺、分析输入存在 |
| `numerics.py` | timestep 合理、系统尺寸够、应变率不超标 |

**到这步你理解了**：验证器 = 15 条独立的规则，每条都是"如果 IR 里的某个条件不满足，就报 Diagnostic"。Diagnostic 里附带了修复建议和权限级别。

---

## 第五阶段：编译器 — IR 变成 LAMMPS 脚本（1 小时）

**目标**：看懂纯代码生成的最后一步

### 5.1 Lowered IR

**阅读文件**：`mdsynth/backend/lowered_ir.py`

`LAMMPSCommand` 就是一个 LAMMPS 命令行的抽象：`kind` + `args` + `comment`。

### 5.2 主编译器

**阅读文件**：`mdsynth/backend/compiler.py`

`compile()` 方法只有 20 行：调用各个 emitter → 拓扑排序 → 渲染为文本。

### 5.3 Emitters — 只读一个

**先读最简单的**：`mdsynth/backend/emitters/header.py`

输入 `MDTypedIR`，输出 `[LAMMPSCommand("clear"), LAMMPSCommand("units", ["metal"]), ...]`。就是把 IR 字段翻译成 LAMMPS 命令。

**再读协议 emitter**（最复杂但最重要）：`mdsynth/backend/emitters/protocol.py`

关注 `_emit_ensemble_fix()` 函数——它是怎么根据 `EnsembleType.NVT / NPT` 生成不同的 `fix` 命令的。

### 5.4 命令依赖 DAG

**阅读文件**：`mdsynth/backend/dependency_graph.py`

`COMMAND_DEPENDENCIES` 字典定义了 LAMMPS 命令之间的先后关系（比如 `pair_coeff` 必须在 `pair_style` 之后）。`topological_sort()` 用 Kahn's 算法自动排序。

**到这步你理解了**：编译器 = 6 个 emitter 各自生成一段命令 → 拓扑排序 → 拼成完整脚本。全程没有任何 if-else 的歧义判断。

---

## 第六阶段：修复引擎（30 分钟）

**目标**：系统怎么在验证失败时自动修复

### 6.1 权限分级

**阅读文件**：`mdsynth/repair/permissions.py`

5 个级别：
- **A**：纯语法修复，自动执行
- **B**：结构修复（不改物理），自动但记录 diff
- **C**：数值修复（改 timestep 等），生成候选
- **D**：物理变更（换力场等），禁止自动执行
- **E**：无物理依据，阻断生成

### 6.2 修复引擎

**阅读文件**：`mdsynth/repair/engine.py`

关键方法：`generate_actions()` → 从 Diagnostic 列表生成 RepairAction 列表 → `apply_actions()` 修改 IR。

重点看 `_apply_action()`：它根据 `action_type` 分发到具体策略。

### 6.3 修复策略

**阅读文件**：`mdsynth/repair/strategies.py`

5 个核心函数：`apply_set_value`、`apply_multiply`、`apply_add_observables`、`apply_disable_pressure_control`、`apply_add_temperature_points`。

每个都做同样的事：深拷贝 IR → 改一个字段 → 返回新 IR。

**到这步你理解了**：修复 = 根据 Diagnostic 的 repair_permission 级别决定能不能自动修 → 调用对应的 strategy 函数 → 修改 IR → 重新验证。

---

## 第七阶段：主流水线串联（30 分钟）

**目标**：把前六步串起来

### 7.1 Pipeline.run()

**阅读文件**：`mdsynth/pipeline.py`

这是**整个项目的入口**。`run()` 方法的 7 个 Step 就是数据流的所有阶段。你已经分别理解了每一段，现在把它们串起来：

```
Step 1: intent_spec = extractor.extract(user_request)       ← 第二阶段
Step 2: md_ir = planner.plan(intent_spec)                    ← 第三阶段
Step 3-4: diagnostics = validator.validate(md_ir)             ← 第四阶段
          → repair.generate_actions() → repair.apply_actions() ← 第六阶段
          → 循环直到无 ERROR 或超过 3 轮
Step 5: script, lowered_ir = compiler.compile(md_ir)          ← 第五阶段
Step 6: preflight_report = sandbox.run(script, md_ir)
Step 7: output = evidence_builder.build(...)                  ← 打包
```

### 7.2 最终测试

```python
# 运行所有测试
pytest tests/ -v

# 只看一个端到端测试，在脑子里走一遍流程
# 读取 tests/integration/test_pipeline_e2e.py 的 test_copper_npt_e2e
```

**到这步你理解了整个项目。**

---

## 附录：关键概念速查

| 概念 | 一句话解释 | 在哪 |
|------|-----------|------|
| `Quantity` | 带单位的数值（不仅仅是 float） | `utils/quantities.py` |
| `QuantitySpec` | 带状态的数值（known/unknown/defaulted） | `intent/spec.py` |
| `FieldStatus` | 字段来源：用户说的/默认的/推断的 | `intent/spec.py` |
| `Diagnostic` | 一条检查结果（哪条规则、什么严重程度、怎么修） | `validator/diagnostic.py` |
| `RepairPermission` | 修复能不能自动执行（A/B/C/D/E） | `repair/permissions.py` |
| `StageType` | 阶段类型：最小化/平衡/生产/变形 | `ir/md_ir.py` |
| `EnsembleType` | 系综：NVE/NVT/NPT | `ir/md_ir.py` |
| `LAMMPSCommand` | 一条 LAMMPS 命令的抽象表示 | `backend/lowered_ir.py` |
| `TaskFamily` | 一种任务的标准配方（几阶段、观测什么） | `ir/task_families.py` |

---

## 学习检查清单

完成每一步后，检查自己能不能回答：

- [ ] 第二阶段：`MDTypedIR` 和 `ScientificIntentSpec` 的区别？
- [ ] 第三阶段：为什么 Planner 不需要 LLM？
- [ ] 第四阶段：一条验证规则的结构是什么？（输入、检查逻辑、输出）
- [ ] 第五阶段：Emitter 的输入输出类型是什么？
- [ ] 第六阶段：C 类和 D 类修复的本质区别是什么？
- [ ] 第七阶段：`pipeline.run()` 的 7 个步骤分别调用了哪个类的哪个方法？

---

> 总预计时间：**约 5 小时**，分 7 个阶段。建议分 2-3 天完成，每读完一个阶段就跑一下对应的单元测试来验证理解。
