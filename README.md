# 中文文本分类推理模块

基于 **hfl/chinese-roberta-wwm-ext**（BERT-base 架构）的通用中文短文本多分类推理模块，支持 10 类标签分类，CPU 即可运行。

## 项目背景

本模块为内部业务系统提供中文文本分类能力，覆盖以下核心需求：

- **子场景**：通用中文短文本多分类
- **标签体系**：10 类互斥标签，已固定不可变更
- **主推荐模型**：`hfl/chinese-roberta-wwm-ext`（110M 参数，Apache 2.0 许可证）
- **备选模型**：`hfl/chinese-macbert-base`（架构完全兼容可互换）
- **评测数据**：仓库内置固定评测集 `inference/sample_data.json`（211 条，10 类），训练数据由 `train.py` 以 seed=42 模板引擎确定性生成，不依赖任何外部数据集

详细论证过程见：

- [01-场景选型论证报告.md](01-场景选型论证报告.md) — 子场景定义、标签体系设计、候选模型对比与最终推荐
- [02-模型尽调报告.md](02-模型尽调报告.md) — 仓库状态审计、许可证合规、架构参数实测、硬件需求分析、已知陷阱与缓解方案
- [reports/evaluation_report.md](reports/evaluation_report.md) — 实测评测报告：数据集构成、整体/per-class 指标、错例分析、复现命令

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
| Python | 3.9+ | 实测环境为 3.12.2 |
| pip | 22.0+ | |
| CPU 内存 | 8GB（训练）/ 4GB（FP32 推理）/ 2GB（INT8 量化） | 训练实测耗时约 7 分钟（14 核 Apple Silicon CPU） |
| GPU（可选） | 2GB VRAM | 有 GPU/MPS 时自动加速，CPU 即可完成全流程 |
| 磁盘 | 约 2GB（含预训练权重 ~400MB + checkpoint ~400MB） | 首次运行自动从 HuggingFace 下载 |

依赖已精确固定版本（见 `inference/requirements.txt`）：torch==2.13.0、transformers==5.15.0、scikit-learn==1.9.0、tqdm==4.70.0、numpy==2.5.2。

## 快速开始（一键运行）

```bash
# 1. 克隆仓库
git clone https://github.com/gxy010116-collab/chinese-text-classification-model.git
cd chinese-text-classification-model

# 2. 一键全流程：venv → 依赖 → 训练（无 checkpoint 时）→ 评测
#    结果自动落盘 inference/results/（eval_results.json + eval_run.log + train_run.log）
cd inference
bash run.sh

# 已有 checkpoint 时会自动跳过训练；也可显式跳过：
bash run.sh --skip-train

# 其他可选参数（透传给 eval.py）
bash run.sh --skip-train --quantize     # INT8 动态量化（更低内存占用）
bash run.sh --device cpu                # 强制使用 CPU
bash run.sh --skip-train --output predictions.json  # 另存详细预测结果
```

`run.sh` 脚本自动完成：
1. 创建 Python 虚拟环境（如不存在）
2. 按 `requirements.txt` 安装精确版本依赖（torch==2.13.0, transformers==5.15.0, scikit-learn==1.9.0, tqdm==4.70.0, numpy==2.5.2）
3. 首次自动从 HuggingFace Hub 下载预训练权重（约 400MB）
4. 若 `checkpoints/pytorch_model.bin` 不存在，自动执行训练（seed=42 生成 4000 条训练样本，3 epochs，`--max-length 48`，日志写入 `results/train_run.log`）
5. 在 211 条内置评测集上运行完整评测，输出分类报告并将指标/混淆矩阵/逐条预测保存为 `results/eval_results.json`

> macOS bash 3.2 兼容性已修复。**checkpoint 获取有两条路径**：① fresh clone（权重文件约 400MB 超出 GitHub 单文件限制，不随仓库分发）——`bash run.sh` 检测到无 checkpoint 时自动训练（seed=42 确定性生成训练数据，14 核 CPU 实测约 7 分钟，低核数 CPU 可能更久），训练完成后自动评测，复现与下方一致的数字（±1pp 容差已声明）；② 本地已有 `checkpoints/pytorch_model.bin`——自动跳过训练直接评测。两条路径的评测数字对齐同一标准。

## 手动运行

```bash
cd inference

# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖（精确版本）
pip install -r requirements.txt

# 3. 训练（seed=42 确定性生成训练数据 + fine-tune，生成 checkpoints/）
python train.py --epochs 3 --samples 400 --seed 42 --max-length 48 --device cpu  # 训练必须 CPU：GPU/MPS 训练非确定性，会破坏 seed=42 复现

# 4. 评测
python eval.py --output results/eval_results.json

# 可选参数
python eval.py --quantize          # INT8 量化
python eval.py --device cpu        # 指定设备
```

## 评测指标

评测脚本输出以下指标（基于 sklearn classification_report）：

- **Accuracy**（准确率）
- **Precision**（精确率，macro + weighted）
- **Recall**（召回率，macro + weighted）
- **F1 Score**（F1 分数，macro + weighted）
- **Per-Class Accuracy**（各类别准确率）
- **Confusion Matrix**（混淆矩阵）

实测评测结果（2026-08-12，内置 211 条评测集，seed=42，非公开基准）：

| 指标 | 实测值 |
|------|--------|
| Accuracy | **84.83%**（179/211） |
| Precision (macro) | 86.53% |
| Recall (macro) | 84.87% |
| F1 (macro) | 84.93% |
| F1 (weighted) | 84.90% |

- 训练耗时 6.9 分钟（14 核 Apple Silicon CPU，`--max-length 48`，batch_size=8），验证集 best val_acc 1.0000
- 推理吞吐约 110 ~ 460 样本/秒（MPS，FP32，batch=32）——吞吐为环境性指标，随机器负载波动：随包 `results/eval_results.json` 是其写入时那次运行的记录（1.924s，109.7 样本/秒），`results/eval_run.log` 则是另一次独立运行的记录（0.46s，458.9 样本/秒），二者均为 `predict_labels(batch=32)` 同口径计时且分类指标完全一致，仅耗时字段随运行时机不同；无 Apple GPU 的机器回退 CPU，吞吐更低但分类结果一致
- 允许浮动范围：同版本依赖 + 固定 seed=42 下数字完全可复现（±0）；若 torch 版本/硬件不同，个别样本 argmax 可能浮动，Accuracy 在 ±1pp（约 ±2 条）内均属正常。出现差异时优先核对 `requirements.txt` 版本与 `checkpoints/training_meta.json` 超参

> **复现注意**：评测数据 `sample_data.json` 包含 211 条固定样本（科技类 22 条，其余每类 21 条）。本仓库不引用 THUCNews 等公开基准数字作为预期值，上表实测值即唯一对齐标准。数字回溯链：`inference/results/train_run.log` → `checkpoints/training_meta.json` → `inference/results/eval_results.json` → `inference/results/eval_run.log`。错例分析详见 [reports/evaluation_report.md](reports/evaluation_report.md)。

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
├── reports/
│   └── evaluation_report.md           # 实测评测报告（指标、错例分析、复现命令）
└── inference/
    ├── README.md                      # 推理模块详细使用说明
    ├── classifier.py                  # 推理核心模块（TextClassifier + InferencePipeline）
    ├── train.py                       # 训练脚本（seed=42 模板数据生成 + fine-tune）
    ├── eval.py                        # 评测脚本
    ├── sample_data.json               # 211 条固定评测样本
    ├── requirements.txt               # Python 依赖（精确版本 pin）
    ├── run.sh                         # 一键运行脚本（venv → 训练 → 评测 → 结果落盘）
    ├── checkpoints/                   # 训练产物目录：training_meta.json（随仓库分发，供超参核对）
    │                                  #   + pytorch_model.bin（约 400MB，超 GitHub 单文件限制，不分发，由 run.sh 自动训练生成）
    └── results/                       # run.sh 自动生成的评测结果目录（随仓库分发供数字回溯）
        ├── eval_results.json          # 完整指标 + 混淆矩阵 + 逐条预测
        ├── eval_run.log               # 评测运行日志
        ├── train_run.log              # 训练运行日志
        └── training_meta.json         # checkpoint 超参元信息副本
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

A: 预训练权重约 400MB，首次从 HuggingFace Hub 下载。可设置镜像加速：

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash run.sh
```

### Q: 环境无法连通 huggingface.co（离线/内网）怎么办？

A: 首次运行（含 fresh clone 后的自动训练路径）必须能获取 `hfl/chinese-roberta-wwm-ext` 预训练权重，完全无网络时无法从零跑通。备选方案（按优先级）：
1. **镜像端点**：`export HF_ENDPOINT=https://hf-mirror.com` 后重跑；
2. **本地缓存离线运行**：权重已存在于 HF 缓存（默认 `~/.cache/huggingface`）时，设置 `export HF_HUB_OFFLINE=1` 即可离线运行；
3. **缓存目录拷贝**：由可联网机器预下载权重后，将其 HF 缓存目录拷贝至目标机器的 `~/.cache/huggingface`，再按方案 2 运行。

注意：微调后的 checkpoint（`checkpoints/pytorch_model.bin`，约 390MB）不入仓，但可由本仓库在目标机器上以 seed=42 重新训练获得（见「快速开始」两条路径说明），因此离线方案只需解决预训练权重的获取。

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

### Q: 评测结果和实测数字对不上？

A: 确认以下条件：
1. 依赖版本与 `inference/requirements.txt` 完全一致（torch==2.13.0、transformers==5.15.0、scikit-learn==1.9.0、tqdm==4.70.0、numpy==2.5.2）
2. 使用 FP32 精度（不加 `--quantize`）
3. 评测数据为 `sample_data.json`（211 条固定样本，未修改）
4. checkpoint 为 `checkpoints/pytorch_model.bin`（seed=42 训练产物，未切换到 MacBERT）
5. 运行 `python eval.py --output results/eval_results.json`

在上述条件下，结果应与实测 Accuracy 84.83%（179/211）完全一致；若硬件/torch 版本不同，个别样本 argmax 浮动导致 Accuracy 在 ±1pp 内属正常。仍不一致时，逐条比对 `results/eval_results.json` 中的 predictions 定位差异样本。

---

*项目交付时间：2026-07-31*
*基于 L3 Benchmark — Case 022*
