#!/bin/sh
### General options
#BSUB -J Run-3-large-datasets-NN-thresh-0.05               # 作业名称
#BSUB -n 4                     # 请求的核心数（根据您的需求调整）
#BSUB -R "span[hosts=1]"       # 所有核心必须在同一节点上
#BSUB -R "rusage[mem=4GB]"     # 每核心需要的内存（根据您的需求调整）
#BSUB -M 4GB                   # 每进程的最大内存限制
#BSUB -W 24:00                 # 最大运行时间为 24 小时
#BSUB -o Output_%J.out         # 标准输出文件
#BSUB -e Output_%J.err         # 标准错误文件

# # 加载 Conda 模块
# module load anaconda3

# 初始化 Conda（如果尚未初始化）
source ~/.bashrc

# 激活 Conda 环境
conda activate rai_fairness

# 运行 Python 脚本
python run.py