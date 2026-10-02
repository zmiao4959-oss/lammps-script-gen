from mdsynth import MDSynthPipeline
from mdsynth.llm.base import MockLLMBackend

pipeline = MDSynthPipeline(llm_backend=MockLLMBackend())
# output = pipeline.run("铜在300K下NPT平衡200ps")
# pipeline.save(output, "./output/")
output = pipeline.run("铁在500K下x方向拉伸")
pipeline.save(output, "./output_iron/")

# 打印生成的 LAMMPS 脚本
print(output.lammps_script)