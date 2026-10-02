from dotenv import load_dotenv
import os
# 加载.env文件中的环境变量
load_dotenv()
# from mdsynth import MDSynthPipeline
# from mdsynth.llm.base import MockLLMBackend

# pipeline = MDSynthPipeline(llm_backend=MockLLMBackend())
# output = pipeline.run("铜在300K下NPT平衡200ps")
# pipeline.save(output, "./output/")
from mdsynth import MDSynthPipeline
from mdsynth.llm.openai_backend import OpenAIBackend

llm = OpenAIBackend(model="deepseek-v4-pro", api_key=os.getenv("DEEPSEEK_API_KEY"),base_url="https://api.deepseek.com")
pipeline = MDSynthPipeline(llm_backend=llm)
pipeline.config.lammps_executable = "C:/Users/35059/app/lammps/LAMMPS 64-bit 22Jul2025-MSMPI/bin/lmp.exe"
output = pipeline.run("模拟Cu的NPT平衡过程之后20%的拉伸过程，温度为300K，拉伸速率为1e-5/ps，拉伸方向为x轴，模拟时间为200ps")
pipeline.save(output, "./output/")

# 查看生成的 LAMMPS 脚本
print(output.lammps_script)
