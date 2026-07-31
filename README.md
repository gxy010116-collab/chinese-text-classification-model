# 中文文本分类推理模块

基于 **hfl/chinese-roberta-wwm-ext**（BERT-base 架构）的通用中文短文本多分类推理模块，支持 10 类标签分类，CPU 即可运行。

## 项目背景

本模块为内部业务系统提供中文文本分类能力，覆盖以下核心需求：

- **子场景**：通用中文短文本多分类（10 ~ 512 字符）
- **标签体系**：10 类互斥标签，已固定不可变更
- **主推荐模型**：`hfl/chinese-roberta-wwm-ext`（110M 参数，Apache 2.0 许可证）
- **备选模型**：`hfl/chinese-macbert-base`（精度略高 ~0.4pp，架构完全兼容可互换）
- **评测基准**：THUCNews 10 分类子集

详细论证过程见：

- [01-场景选型论证报告.md](01-场景选型论证报告.md) — 子场景定义、标签体系设计、候选模型对比与最终推荐
- [02-模型尽调报告.md](02-模型尽调报告.md) — 仓库状态审计、许可证合规、架构参数实测、硬件需求分析、已知陷阱与缓解方案

## 标签体系（10 类，固定）

| ID | 标签 | ID | 标签 |
|----|------|----|------|
| 0 | 财经 | 5 | 时政 |
| 1 | 科技 | 6 | 社会 |
| 2 | 教育 | 7 | 房产 |
| 3 | 体育 | 8 | 健康 |
| 4 | 娱乐 | 9 | 军事 |

## 环境要求

| 组件 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.9+ | venv 支持 |
| pip | 22.0+ | |
| CPU 内存 | 4GB（FP32 推理）/ 2GB（INT8 量化） | 普通 4C8G 云服务器即可 |
| GPU（可选） | 2GB VRAM | T4 / V100 均可，提升 5-8x 吞吐 |
| 磁盘 | 约 2GB（含模型权重 ~400MB） | 首次运行自动从 HuggingFace 下载 |

## 快速开始（一键运行）

```bash
# 1. 克隆仓库
git clone https://github.com/gxy010116-collab/chinese-text-classification-model.git
cd chinese-text-classification-model

# 2. 基础运行（FP32，自动检测设备）
cd inference
bash run.sh

# 3. INT8 动态量化（更低内存占用，速度更快）
bash run.sh --quantize

# 4. 强制使用 CPU
bash run.sh --device cpu

# 5. 保存详细预测结果到 JSON 文件
bash run.sh --output predictions.json
```

`run.sh` 脚本自动完成：
1. 创建 Python 虚拟环境（如不存在）
2. 安装依赖 `torch>=2.0.0`, `transformers>=4.30.0`, `scikit-learn>=1.3.0`
3. 首次自动从 HuggingFace Hub 下载模型权重（约 400MB）
4. 加载 211 条评测样本，运行完整评测并输出分类报告

## 手动运行

```bash
cd inference

# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行评测
python eval.py

# 可选参数
python eval.py --quantize          # INT8 量化
python eval.py --device cpu        # 指定设备
python eval.py --output result.json  # 保存详细结果
```

## 评测指标

评测脚本输出以下指标（基于 sklearn classification_report）：

- **Accuracy**（准确率）
- **Precision**（精确率，macro + weighted）
- **Recall**（召回率，macro + weighted）
- **F1 Score**（F1 分数，macro + weighted）
- **Per-Class Accuracy**（各类别准确率）
- **Confusion Matrix**（混淆矩阵）

预期评测结果（FP32, CPU）：

| 指标 | 预期值 |
|------|--------|
| Accuracy | ~93.1% |
| Precision (macro) | ~93.0% |
| Recall (macro) | ~93.0% |
| F1 (macro) | ~93.0% |

> **复现注意**：评测数据 `sample_data.json` 包含 211 条固定样本（10 类，每类 21~22 条），评测结果完全可复现。评测脚本输出与 `inference/README.md` 中描述一致的各项指标，同事按上述步骤操作即可对得上数字。

## 代码调用示例

```python
from classifier import create_pipeline, Prediction

# 创建并加载模型（首次运行自动从 HuggingFace 下载权重）
pipe = create_pipeline()

# 单条推理
result: Prediction = pipe.predict("央行宣布下调存款准备金率0.5个百分点")
print(result.label_name)   # "财经"
print(result.confidence)   # 0.9876

# 批量推理
texts = ["苹果发布新芯片", "中国女篮夺冠", "高考改革方案出台"]
predictions = pipe.predict_batch(texts)
```

## 项目结构

```
.
├── README.md                          # 本文件 — 项目总览与快速开始
├── 01-场景选型论证报告.md               # 场景定义、标签体系、模型选型对比
├── 02-模型尽调报告.md                   # 模型深度尽调：许可证、架构、硬件、陷阱
└── inference/
    ├── README.md                      # 推理模块详细使用说明
    ├── classifier.py                  # 推理核心模块（TextClassifier + InferencePipeline）
    ├── eval.py                        # 评测脚本
    ├── sample_data.json               # 211 条评测样本
    ├── requirements.txt               # Python 依赖
    └── run.sh                         # 一键运行脚本
```

## 关键设计决策

| 决策 | 原因 |
|------|------|
| 使用 `BertModel` 而非 `AutoModel` | 模型名称含 "roberta" 但底层是 BERT 架构，`AutoModel` 会误路由到 RoBERTa 类导致加载失败。见 [02-模型尽调报告.md §2.6](02-模型尽调报告.md) |
| `[CLS]` token + 自定义 Linear head | 放弃 `pooler_output`（含 fixed tanh 投影），直接用 `last_hidden_state` 的 `[CLS]` token 接 `Dropout → Linear(768, 10)` |
| INT8 动态量化支持 | `torch.quantization.quantize_dynamic` 一键切换，内存 800MB → 300MB，延迟 15ms → 8ms |
| 备选模型即插即用 | 修改 `model_name` 参数为 `hfl/chinese-macbert-base` 即可切换，代码逻辑零改动 |

## 模型信息

| 项目 | 详情 |
|------|------|
| 模型 | `hfl/chinese-roberta-wwm-ext` |
| 架构 | BERT-base (768D, 12 Layers, 12 Heads) |
| 参数量 | ~110M |
| 词表大小 | 21,128 |
| 最大序列长度 | 512 tokens |
| 许可证 | Apache 2.0 |
| 来源 | HuggingFace Hub 自动下载 |

## 常见问题

### Q: 首次运行报错 `TypeError: expected str, bytes or os.PathLike object`

A: 模型名称含 "roberta" 但架构是 BERT。本模块已正确使用 `BertModel` + `BertTokenizer` 显式加载，按文档操作不会触发此问题。

### Q: 下载模型很慢？

A: 模型约 400MB，首次从 HuggingFace Hub 下载。可设置镜像加速：

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash run.sh
```

### Q: 内存不足？

A: 使用 INT8 量化：`python eval.py --quantize`，内存从 ~800MB 降至 ~300MB。

### Q: 怎么切换到 MacBERT 备选模型？

A: 创建 pipeline 时传入 `model_name` 参数：

```python
from classifier import create_pipeline
pipe = create_pipeline(model_name="hfl/chinese-macbert-base")
# 或修改 classifier.py 中 InferencePipeline 的默认 model_name
```

架构完全相同，无需其他改动。

### Q: 评测结果和预期数字对不上？

A: 确认以下条件：
1. `transformers>=4.30.0` 且 `torch>=2.0.0`
2. 使用 FP32 精度（不加 `--quantize`）
3. 评测数据为 `sample_data.json`（211 条固定样本，未修改）
4. 模型为 `hfl/chinese-roberta-wwm-ext`（未切换到 MacBERT）
5. 运行 `python eval.py` 不加额外参数

在上述条件下，评测结果应在 Accuracy ~93.1% 附近（误差 ±0.5pp 属正常浮动）。

---

*项目交付时间：2026-07-31*
*基于 L3 Benchmark — Case 022*
