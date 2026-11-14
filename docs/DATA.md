# 数据来源与输入

## 来源

- OneK1K 原研究：[Science, DOI: 10.1126/science.abf3041](https://doi.org/10.1126/science.abf3041)。
- [CELLxGENE 数据集合](https://cellxgene.cziscience.com/collections/dde06e0f-ab3b-46be-96a2-a8082383c4a1)。
- [本项目使用的 H5AD 文件](https://datasets.cellxgene.cziscience.com/1e44db10-b572-46cc-adae-dcc7acd44ca6.h5ad)。

该版本包含1,248,980个细胞、35,528个基因和981个供者ID。原研究与平台文件的供者数量描述存在差异，分析按实际文件中的981个ID统计。

## 获取

在仓库根目录执行，下载前确认上游数据使用条件：

```bash
mkdir -p dataset
curl --fail --location --retry 3 \
  'https://datasets.cellxgene.cziscience.com/1e44db10-b572-46cc-adae-dcc7acd44ca6.h5ad' \
  --output dataset/onek1k.h5ad
sha256sum dataset/onek1k.h5ad
```

固定配置要求：

- 文件大小：4,434,273,970字节。
- SHA256：`14c5e9e70007a5e7dc047f7b29405773305e1a7b9791d0789af70649cf6f46ec`。

上述校验值来自项目实际使用的文件，用于锁定输入版本。

## 输入字段与处理

H5AD需包含供者ID、连续年龄、性别、技术pool、`predicted.celltype.l2`注释及细胞质控字段。脚本验证供者内元数据的一致性，按供者与细胞类型聚合计数，并生成log1p CPM表达矩阵。

固定质控阈值为每细胞至少200个检测基因、线粒体比例至多20%、每名供者每类细胞至少30个合格细胞。五类细胞的注释集合详见[配置](../config/cell_types.json)。

## 分析队列

- 主分析：五类细胞均通过质控的共同505人，统一内外层划分。
- CD8补充分析：最大可用977人，用于补充预测和特征解释。
- 调优v2：共同505人中的探索性分析，主比较仍采用冻结v1。

原始输入、供者名单、年龄、表达矩阵和模型输出均在运行者本地生成，由忽略规则排除。
