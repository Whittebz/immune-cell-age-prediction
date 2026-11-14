# 复现说明

## 环境与运行顺序

使用Python 3.12和[固定依赖](../requirements.txt)。在仓库根目录安装环境、获取[数据](DATA.md)，再按[README](../README.md)建立共同队列、准备各细胞输入、训练和验证。

CPU线性代数库的线程数可按机器资源限制：

```bash
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
```

`cell_pipeline.py cohort`基于源文件元数据生成共同供者名单与固定划分。`prepare`生成各细胞的供者级矩阵，`train`执行共同队列训练，`validate`核对数据、协议、划分和保存产物。

## 本地产物

| 路径 | 用途 |
| --- | --- |
| `dataset/onek1k.h5ad` | 原始输入 |
| `data/processed/<CELL>/` | 供者级表达与元数据 |
| `data/shared/` | 共同供者名单、分折与协议 |
| `data/models/<CELL>/common_v1/` | 主分析模型与逐供者预测 |
| `results/cells/<CELL>/common_v1/` | 主分析指标、图表与验证结果 |
| `results/cells/<CELL>/pool_en_common_v1/` | pool分组Elastic Net补充结果 |

这些目录在运行时生成，不作为公开仓库内容提交。程序保留已有输出，重复实验应先查看对应脚本的`--help`并选择新运行目录。

## CD8最大可用队列与特征解释

下列独立入口使用CD8最大可用队列，与共同505人主基准分别记录：

```bash
python src/train_cd8.py
python src/validate_cd8_training.py
python src/interpret_cd8.py
```

`interpret_cd8.py`读取[文献登记表](CD8_BIOLOGY_REFERENCES.json)，生成代表基因描述统计和图表。模型系数、同队列年龄相关性和外部文献证据分别解释。

## 方法学检查与探索性调优

```bash
python src/audit_cohort_selection.py
python src/audit_cd8_features.py
python src/audit_oof_diagnostics.py
python src/tune_cd8_common.py --help
python src/validate_cd8_tuning.py --help
```

各入口依赖已完成的相应预处理和模型输出。调优配置见[cd8_tuning_v2.json](../config/cd8_tuning_v2.json)，包含固定特征数、正则化网格、训练折年龄权重和比较规则。v2在查看先前结果后开展，作为探索性敏感性分析单独报告。

## 验证

```bash
python -m unittest discover -s src -p 'test_*.py' -v
```

单元及合成集成测试检查训练折内预处理、固定分折、预测重建、协议校验和统计计算。依赖真实数据的测试在缺少本地输入时跳过；完整真实数据复现还需执行相应`validate`入口。

评估采用外层折外预测；全数据重拟合模型用于数值演示。pool分组检验衡量同队列内批次泛化，跨队列应用需独立外部验证。
