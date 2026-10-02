# MDSynth 项目上下文

AI 驱动的 LAMMPS 分子动力学模拟脚本自动生成器。详见 README.md。

## 关键文件

- `mdsynth/pipeline.py` — 主编排器，7 步流水线
- `mdsynth/intent/` — 自然语言 → 意图提取
- `mdsynth/ir/` — 中间表示层
- `mdsynth/validator/` — 物理约束检查（15 条规则）
- `mdsynth/backend/compiler.py` — IR → LAMMPS 脚本编译器
- `mdsynth/knowledge/` — 材料、势函数、命令模板知识库
- `SPECIFICATION.md` — 完整开发规格说明书

## 技术栈

Python 3.10+, pydantic v2, openai, pyyaml, numpy, structlog
