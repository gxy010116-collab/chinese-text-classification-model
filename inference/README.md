# 中文文本分类推理模块

## 概述

基于 **hfl/chinese-roberta-wwm-ext**（BERT-base 架构）的中文短文本 10 分类推理模块，包含自动训练和评测全流程。支持 CPU/GPU/MPS 推理，可选 INT8 动态量化。

**标签体系（固定10类，不可变更）：**

| ID | 标签 | ID | 标签 |
|----|------|----|------|
| 0 | 财经 | 5 | 时政 |
| 1 | 科技 | 6 | 社会 |
| 2 | 教育 | 7 | 房产 |
| 3 | 体育 | 8 | 健康 |
| 4 | 娱乐 | 9 | 军事 |

## 环境要求

| 组件 | 最低版本 |
|------|---------|
| Python | 3.9+ |
| pip | 22.0+ |
| CPU 内存 | 8GB（训练）/ 4GB（推理 FP32）/ 2GB（INT8 量化） |
| 磁盘 | ~2GB（模型权重 ~400MB + checkpoint ~400MB + venv） |
| GPU（可选） | 2GB VRAM（训练建议 6GB+，推理 2GB 即可） |

## 快速开始（一键运行）

```bash
# 全流程：安装依赖 → 自动训练 → 评测（首次需 30-60 分钟 CPU 训练）
bash run.sh

# 如果已有 checkpoint，跳过训练直接评测
bash run.sh --skip-train

# INT8 量化推理（更低内存占用）
bash run.sh --skip-train --quantize

# 强制使用 CPU
bash run.sh --device cpu

# 保存详细预测结果
bash run.sh --skip-train --output predictions.json
```

## 手动分步运行

```bash
# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 训练模型（生成训练数据 + fine-tune）
#    run.sh 使用 --max-length 48（评测/训练文本均短于 48 token）；多核 CPU 实测约 7 分钟
python train.py --epochs 3 --samples 400 --seed 42 --max-length 48 --device cpu

# 4. 评测
python eval.py
```

## 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 3 | 训练轮数 |
| `--samples` | 400 | 每类训练样本数（总计 ~4000，15% 做验证集） |
| `--batch-size` | 8 | 批大小（CPU 建议 4-8） |
| `--lr` | 2e-5 | 学习率 |
| `--max-length` | 128 | 最大 token 长度（缩短可加速训练） |
| `--seed` | 42 | 随机种子（固定以保证可复现） |
| `--device` | auto | 指定设备（cpu/cuda/mps） |

**训练数据**: 通过模板+关键词组合自动生成，覆盖 10 个类别的多样化中文短文本，不需要下载外部数据集。固定随机种子 (seed=42) 确保每次训练结果一致。

**预期训练时间**:
- CPU (4核): ~30-60 分钟（400 samples/class, 3 epochs）
- GPU (T4): ~5-10 分钟

**实测评测指标**（2026-08-12 在内置 211 条评测集 `sample_data.json` 上实测，非预期值）:

| 指标 | 实测值 |
|------|--------|
| Accuracy | **84.83%**（179/211） |
| Precision (macro) | 86.53% |
| Recall (macro) | 84.87% |
| F1 (macro) | 84.93% |
| F1 (weighted) | 84.90% |

实测数字与结果文件对应关系：
- 完整指标、per-class 明细、混淆矩阵、逐条预测 → `results/eval_results.json`
- 运行日志 → `results/eval_run.log`（评测）、`results/train_run.log`（训练）
- 训练超参与 checkpoint 元信息 → `results/training_meta.json`（与 `checkpoints/training_meta.json` 一致）

注意：本仓库不引用 THUCNews 等公开基准数字作为预期值；以上数字即本仓库在固定 seed=42 下的可复现实测值。分析详见 `reports/evaluation_report.md`。

## 评测指标

评测脚本输出：
- **Accuracy**（准确率）
- **Precision**（精确率，macro + weighted）
- **Recall**（召回率，macro + weighted）
- **F1 Score**（F1 分数，macro + weighted）
- **Per-Class Accuracy**（各类别准确率）
- **Confusion Matrix**（混淆矩阵）

评测数据 `sample_data.json` 包含 211 条固定测试样本（科技类 22 条，其余各类 21 条），覆盖全部 10 个标签类别。评测数字与训练 checkpoint 绑定，固定 seed=42 可完全复现。

## 文件说明

```
inference/
├── classifier.py        # 推理核心模块（TextClassifier + InferencePipeline）
│                         # 自动加载 checkpoints/ 下的微调权重
├── train.py             # 训练脚本（数据生成 + fine-tune + checkpoint 保存）
├── eval.py              # 评测脚本（accuracy / precision / recall / F1）
├── sample_data.json     # 固定评测数据（10类共211条，科技22+其余各21）
├── requirements.txt     # Python 依赖
├── run.sh               # 一键运行脚本
├── checkpoints/         # 训练后生成的微调权重目录
│   ├── pytorch_model.bin
│   └── training_meta.json
├── results/             # 评测输出目录（run.sh 自动生成）
│   ├── eval_results.json    # 完整指标 + 混淆矩阵 + 逐条预测
│   ├── eval_run.log         # 评测运行日志
│   ├── train_run.log        # 训练运行日志
│   └── training_meta.json   # checkpoint 元信息副本
└── README.md            # 本文件
```

## 代码调用示例

```python
from classifier import create_pipeline, Prediction

# 创建并加载模型（自动加载 fine-tune checkpoint）
pipe = create_pipeline()

# 切换到 MacBERT 备选模型
# pipe = create_pipeline(model_name="hfl/chinese-macbert-base")

# 单条推理
result: Prediction = pipe.predict("央行宣布下调存款准备金率0.5个百分点")
print(result.label_name)   # "财经"
print(result.confidence)   # 0.9876

# 批量推理
texts = ["苹果发布新芯片", "中国女篮夺冠", "高考改革方案出台"]
predictions = pipe.predict_batch(texts)
```

## 常见问题

### Q: 首次运行报错 `TypeError: expected str, bytes or os.PathLike object`
A: 模型名称含 "roberta" 但架构是 BERT。本模块已正确使用 `BertModel` + `BertTokenizer` 显式加载。自行编写加载代码时请使用 BERT 类而非 Auto 类。

### Q: 模型加载时出现 "Some weights were not used" 警告
A: 这是正常现象。BERT backbone 加载时会忽略 MLM 预训练任务的权重（如 `cls.predictions.*`），只使用 encoder 部分。不影响分类效果。

### Q: 下载模型很慢？
A: 模型约 400MB，首次从 HuggingFace Hub 下载。可设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`。

### Q: 内存不足？
A: 推理时使用 INT8 量化：`python eval.py --quantize`，内存从 ~800MB 降至 ~300MB。训练时内存不足可减少 `--samples 200 --batch-size 4`。

### Q: 评测数字和我同事跑出来的不一样？
A: 确保使用相同的 `sample_data.json` 与相同的依赖版本。checkpoint 权重文件（约 400MB）不随仓库分发——fresh clone 后 `bash run.sh` 会以固定 seed=42 自动重新训练（模板数据确定性生成，约 7 分钟），得到的 checkpoint 与评测数字应和本仓库声明一致（±1pp 容差）。若你本地有既有 checkpoint，请核对其 `training_meta.json` 超参（seed=42 / epochs=3 / samples=400 / max_length=48）。

### Q: 怎么切换到 MacBERT 备选模型？
A: 训练时 `python train.py --model-name hfl/chinese-macbert-base`，推理时 `create_pipeline(model_name="hfl/chinese-macbert-base")`。
