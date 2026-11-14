# Immune Cell Age Prediction

基于 OneK1K 外周血单细胞转录组的供者实际年龄预测。项目比较 CD4 T、CD8 T、NK、B 和单核细胞，提供供者级表达聚合、嵌套交叉验证、基线比较、特征解释与方法学检查代码。

## 方法

- 按供者与细胞类型聚合计数，采用 log1p CPM 归一化。
- 五类细胞主比较使用共同505名供者和相同交叉验证划分。
- 外层5折评估，内层3折选参；方差特征筛选和标准化在对应训练折中拟合。
- 比较中位数基线、Ridge 和 Elastic Net，以 MAE（年）和 R² 评价。
- 通过按技术 pool 分组的嵌套验证检查同队列内的批次泛化。
- CD8最大可用977人分析及调优v2分别作为补充分析。

## 安装

使用 Python 3.12。依赖版本列于 [requirements.txt](requirements.txt)。

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## 运行

先按[数据说明](docs/DATA.md)获取输入，在仓库根目录执行：

```bash
# 单独处理并验证CD8数据，作为共同队列流程的CD8输入。
python src/preprocess_cd8.py
python src/validate_cd8.py

# 建立五类细胞共同供者名单和固定划分。
python src/cell_pipeline.py cohort

# 对每类细胞准备输入、训练和验证。
for cell in CD8 CD4 B NK MONO; do
  python src/cell_pipeline.py prepare --cell "$cell"
  python src/cell_pipeline.py train --cell "$cell"
  python src/cell_pipeline.py validate --cell "$cell"
done

# 补充各细胞的共同队列pool分组Elastic Net验证。
for cell in CD8 CD4 B NK MONO; do
  python src/run_cell_supplement.py pool-en --cell "$cell" --cohort common
done
```

程序将数据和模型保存在本地，已有实验目录会触发覆盖保护。详细入口、补充分析及验证方式见[复现说明](docs/REPRODUCING.md)。

## 测试

```bash
python -m unittest discover -s src -p 'test_*.py' -v
```

测试包含合成数据单元测试和集成检查；依赖本地真实数据的测试在缺少输入时跳过。

## 目录

| 路径 | 内容 |
| --- | --- |
| `src/` | 数据处理、模型、解释、审查及测试代码 |
| `config/` | 细胞注释、质控、固定训练与探索性调优配置 |
| `docs/DATA.md` | 数据来源、下载与输入约定 |
| `docs/REPRODUCING.md` | 运行顺序、产物路径及补充实验 |
| `docs/CD8_BIOLOGY_REFERENCES.json` | 特征解释脚本使用的文献登记表 |

公开仓库收录代码、运行配置和必要文档。原始数据、供者级派生数据、模型文件、分析输出、PPT、讲稿和交流记录保存在仓库外。

研究目标为实际年龄回归。基因解释基于统计关联与文献证据，外部泛化和机制作用需要独立验证。
