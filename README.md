# CSI300 产业链知识图谱与行业嵌入

本项目是一套**数据 + 实验脚本**集合，用于为金融文本分析模型构造「公司关联行业」的语义特征。核心流程：

1. 从巨潮资讯（cninfo）获取 A 股上市公司所属产业链数据；
2. 构建「行业/产品/服务 → 上游」知识图谱；
3. 使用 [OpenKE](https://github.com/thunlp/OpenKE) 的 **TransH** 在图上训练行业节点向量；
4. 将每家公司的行业向量和行业名称 token 组织成下游模型可直接加载的 pickle 特征。

> 仓库中的路径（如 `coling2018/...`）表明它来自一个 COLING 2018 金融文本分析项目的配套数据生成流程。

## 数据流

```
cninfo API
    │
    ▼
industries_df.txt ──────────────► entity2id.txt / relation2id.txt / train2id.txt
    │                                              │
    │                                              ▼
    │                                        OpenKE TransH
    │                                              │
    ▼                                              ▼
上市公司信息 ──► name_to_industries.dict ──► industry_embeddings.dict
                        │                          │
                        └──────────┬───────────────┘
                                   ▼
                            industry_gen.py
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
          embedding&map/w2v/            embedding&map/bert/
      （LTP 分词 + 词表）              （FinBERT tokenizer）
```

## 目录结构

```
.
├── CSI300_industry_graph.ipynb   # 主流程：API 调用、图谱构建、嵌入导出、IndustryEncoder 草稿
├── train_demo.ipynb              # OpenKE 训练脚本（TransE demo + TransH 产业链图训练）
├── industry_gen.py               # 按公司生成下游模型使用的特征 pickle
├── industries_df.txt             # 原始公司-产业链关系（CSV 格式，约 22 万行）
├── entity2id.txt                 # 图谱节点：行业/产品/服务名称 → id
├── relation2id.txt               # 图谱关系：仅「上游」一种
├── train2id.txt                  # 图谱三元组
└── embedding&map/
    ├── industry_embeddings.dict  # TransH 行业向量（6527 × 200）
    ├── name_to_industries.dict   # 公司名 → 行业列表
    ├── w2v/                      # w2v 方案产物（LTP 分词，序列长度 12）
    └── bert/                     # bert 方案产物（FinBERT tokenizer，序列长度 22）
```

## 数据说明

| 文件                         | 内容                                                                                                                                | 规模                         |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| `industries_df.txt`        | 公司所属产业链明细，字段：`UDINDUSTRYNAME / POSITION / CODE / UDINDUSTRYCODE / INDUSTRYCODE / INDUSTRYNAME / NAME / COMPANY_CODE` | 221,983 行                   |
| `entity2id.txt`            | 行业节点编号，格式 `名称\tid`                                                                                                     | 6,527 个节点                 |
| `relation2id.txt`          | 关系定义，仅 `上游 → 0`                                                                                                          | 1 种关系                     |
| `train2id.txt`             | 三元组 `head\ttail\trelation`                                                                                                     | 28,388 条边                  |
| `name_to_industries.dict`  | 上市公司名 → 所属行业列表                                                                                                          | 297 家公司，平均 13.5 个行业 |
| `industry_embeddings.dict` | 行业名称 → 200 维 TransH 向量                                                                                                      | 6,527 × 200                 |

**边的方向**：`head → tail` 表示「head 的上游是 tail」。例如 `智慧工厂 → 物联网服务`。

`name_to_industries.dict` 中的 297 家是产业链数据与下游数据集目标实体的交集（因此少于 CSI300 的 300 家）。其中 47 家公司的行业列表包含 `nan`。

> 注：`*.dict` 文件实际内容是 Python 字典字面量，可用 `eval` / `ast.literal_eval` 读取；`nan` 需要自行处理。

## 环境依赖

- Python 3.8+
- `requests`、`pandas`、`numpy`、`tqdm`、`torch`、`transformers`
- [OpenKE](https://github.com/thunlp/OpenKE)（需单独安装，仓库未内置）
- LTP（代码使用 `init_dict / add_words / seg` 接口，约 v3 版本）
- 外部资源（不随仓库提供，需自行准备）：
  - 巨潮资讯开放平台账号及 `p_chain_by_scode` 接口权限
  - FinBERT 预训练权重（`FinBERT_L-12_H-768_A-12_pytorch`）
  - 下游数据集的 `TEXT.vocab.*.pkl` / `TARGET.vocab.*.pkl`
  - 经济领域词典 `Econ_Dict.txt`

## 使用步骤

### 1. 配置 cninfo 凭据

凭据从环境变量读取，不要写进代码：

```bash
export CNINFO_ACCESS_KEY=your_key
export CNINFO_ACCESS_SECRET=your_secret
# 测试环境（可选）
export CNINFO_TEST_ACCESS_KEY=...
export CNINFO_TEST_ACCESS_SECRET=...
```

可选覆盖服务地址：`CNINFO_BASE_URL` / `CNINFO_TEST_BASE_URL`。

> 注意：仓库中原有的 cninfo 账号已被平台停用（`用户被禁用，不允许生成TOKEN`），测试环境也已下线。抓取数据前需要重新申请账号。如果 kernel 已经启动，在 notebook 内用 `%env CNINFO_ACCESS_KEY=...` 或重启 kernel。

### 2. 构建知识图谱

打开 `CSI300_industry_graph.ipynb`，依次运行：

- 调用 `CninfoAPI` 获取/补充产业链数据；
- 读取 `industries_df.txt`，为行业节点和边编号；
- 导出 `entity2id.txt`、`relation2id.txt`、`train2id.txt`。

输出路径目前是绝对路径（`/home/chenhaoran/...`），在不同机器上运行前需修改。

### 3. 训练 TransH

在 `train_demo.ipynb` 中：

- 将 `TrainDataLoader` 的 `in_path` 指向本仓库根目录；
- 运行 TransH 训练（`dim=200`，`NegativeSampling + MarginLoss`）；
- 保存 checkpoint，并导出 `ent_embeddings.weight`。

notebook 前半部分的 TransE/FB15K237 是 OpenKE 的示例，与产业链图无关。

### 4. 导出行业向量并生成下游特征

- 在 notebook 中按 `entity2id.txt` 从 TransH 权重取出每个行业的向量，生成 `industry_embeddings.dict`；
- 生成公司名 → 行业列表的 `name_to_industries.dict`；
- 运行特征生成脚本：

```bash
python industry_gen.py --method w2v --save
# 或
python industry_gen.py --method bert --save
```

运行前需修改文件顶部的 `IND_DIR_PATH`、`MODEL_DIR_PATH`、`LTP_DICT_PATH`，并确保外部 vocab / 模型文件存在。

## 下游特征产物

每个 pickle 均以 **目标实体 id（`targetId`）** 为键，覆盖 297 家公司：

| 文件                             | 内容                                             |
| -------------------------------- | ------------------------------------------------ |
| `targetId_to_node_embs.pkl`    | 公司 → 其各行业的 TransH 向量列表               |
| `targetId_to_token_idx.pkl`    | 公司 → 其各行业名称的 token id（已 padding）    |
| `targetId_to_len_inds.pkl`     | 公司 → 涉及行业数量                             |
| `targetId_inds_len_tokens.pkl` | 公司 → 每个行业名称的真实 token 数（用于 mask） |

`w2v/` 与 `bert/` 两套产物的维度不同（每个行业名分别 padding 到 12 / 22 个 token），其他结构一致。

notebook 末尾的 `IndustryEncoder` / `AttNet` / `SequentialRepr` 展示了预期用法：以新闻序列表示作为 query，对公司各行业向量做 attention 聚合。该部分是未完成的草稿（`forward` 尚未实现）。

## 已知限制

- 所有脚本使用硬编码绝对路径，直接运行前需调整；
- OpenKE、FinBERT、LTP 词典、下游 vocab 等依赖不在仓库内；
- 图谱规模较小且只有一种关系（上游），嵌入主要起「注入产业链结构信息」的作用；
- 数据为约 2022 年的快照，部分公司与行业字段存在 `nan`；
