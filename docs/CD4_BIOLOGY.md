# CD4 T模型的特征解释：共同505人

## 分析对象

本解释使用五类细胞主比较中的同一批505名供者，以及已完成的CD4 T供者级Elastic Net模型。CD4 T供者CV MAE为7.851年、R²为0.634；pool分组Elastic Net MAE为7.807年。

从87个符合原稳定规则的候选中，选取LRRN3、CD70和GZMB作为有原始文献依据的代表。三者在供者CV和pool分组模型中均为5/5折非零且方向一致。选择在模型训练完成后进行，用于解释已有结果；完整模型包含更多特征。

## 模型与同队列观察

| 基因 | 供者CV平均标准化系数 | 供者CV非零折数 | Pool CV非零折数 | 年龄Spearman相关 | 调整亚群比例后 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRRN3 | −1.246 | 5/5 | 5/5 | −0.552 | −0.484 |
| CD70 | +0.608 | 5/5 | 5/5 | +0.290 | +0.199 |
| GZMB | +0.366 | 5/5 | 5/5 | +0.336 | +0.240 |

模型系数表示其他特征固定时，标准化表达变化对应的预测年龄变化。相关性使用同一505人作描述性计算。组成调整对年龄和表达取秩，分别回归于截距及五个秩变换的细胞比例，再计算残差相关；CD4 TEM为省略的参照组成。亚群包括CD4 Naive、TCM、TEM、CTL、Proliferating和Treg。

## 文献解释

- **LRRN3：年龄相关表达。** Peters等在人类全血研究的原文Table 1中报告LRRN3随年龄降低，方向与本模型负系数及负相关一致。该研究提供血液表达背景。
- **CD70：共刺激与分化状态。** Liu等在人类CD4+CD28−与CD4+CD28+亚群中观察到CD70甲基化与表达差异。研究招募包含类风湿关节炎与老年非RA供者，RNA比较为5名供者，提供CD4共刺激及分化状态背景。
- **GZMB：细胞毒性程序。** Hashimoto等在超百岁人群中观察到表达GZMB等细胞毒性基因的CD4 T细胞。该文支持GZMB的细胞状态解释；其普通成人参考数据未显示显著的CD4 CTL年龄组差异，因此不以该文证明一般成人随年龄单调变化。

三项关联在组成调整后均减弱，表明亚群结构参与聚合表达的年龄相关信号。剩余关联属于条件统计关联。上述证据解释CD4模型使用的特征；它们不单独确定CD4优于其他细胞类型的原因。当前观察来自OneK1K，分子解释限于基因级RNA和所引用研究的测量层次。

## 原始研究

1. Peters MJ, Joehanes R, Pilling LC, et al. *The transcriptional landscape of age in human peripheral blood.* Nature Communications 6, 8570 (2015). DOI：[10.1038/ncomms9570](https://doi.org/10.1038/ncomms9570)。核对原文Table 1。
2. Liu Y, Chen Y, Richardson B. *Decreased DNA methyltransferase levels contribute to abnormal gene expression in 'senescent' CD4(+)CD28(-) T cells.* Clinical Immunology 132(2), 257–265 (2009). DOI：[10.1016/j.clim.2009.03.529](https://doi.org/10.1016/j.clim.2009.03.529)。核对原文Results与Figure 1C。
3. Hashimoto K, Kouno T, Ikawa T, et al. *Single-cell transcriptomics reveals expansion of cytotoxic CD4 T cells in supercentenarians.* PNAS 116(48), 24242–24251 (2019). DOI：[10.1073/pnas.1907883116](https://doi.org/10.1073/pnas.1907883116)。核对原文Results、Figure 4和Figure 6。

逐篇证据范围见[文献登记表](CD4_BIOLOGY_REFERENCES.json)。

## 复现

完成CD4共同队列训练与pool分组补充后，在仓库根目录执行：

```bash
python src/interpret_cd4.py
python -m unittest discover -s src -p 'test_interpret_cd4.py' -v
```

脚本读取现有数据和模型汇总，按分析名单对齐505人，核对输入校验和、双协议稳定性和比例完整性，生成队列级表格及图像。输出位于本地`results/cd4_biology_common505_verified/`，已有目录受覆盖保护。原模型、预测和指标保持不变。
