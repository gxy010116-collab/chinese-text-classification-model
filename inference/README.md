# 中文文本分类推理模块

## 概述

基于 **hfl/chinese-roberta-wwm-ext**（BERT-base 架构）的中文短文本 10 分类推理模块，支持 CPU/GPU/MPS 推理，可选 INT8 动态量化。

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
| CPU 内存 | 4GB（FP32）/ 2GB（INT8 量化） |
| GPU（可选） | 2GB VRAM（T4/V100 均可） |

## 快速开始（一键运行）

```bash
# 基础运行（FP32，自动检测设备）
bash run.sh

# INT8 动态量化（更低内存占用，速度更快）
bash run.sh --quantize

# 强制使用 CPU
bash run.sh --device cpu

# 保存详细预测结果到 JSON 文件
bash run.sh --output predictions.json
```

## 手动运行

```bash
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

## 文件说明

```
inference/
├── classifier.py        # 推理核心模块（TextClassifier + InferencePipeline）
├── eval.py              # 评测脚本（accuracy / precision / recall / F1）
├── sample_data.json     # 评测数据（10类 × 21条 = 210条样本）
├── requirements.txt     # Python 依赖（精确版本约束）
├── run.sh               # 一键运行脚本
└── README.md            # 本文件
```

## 代码调用示例

```python
from classifier import create_pipeline, Prediction

# 创建并加载模型（首次运行自动从 HuggingFace 下载权重，约 400MB）
pipe = create_pipeline()

# 单条推理
result: Prediction = pipe.predict("央行宣布下调存款准备金率0.5个百分点")
print(result.label_name)   # "财经"
print(result.confidence)   # 0.9876

# 批量推理
texts = ["苹果发布新芯片", "中国女篮夺冠", "高考改革方案出台"]
predictions = pipe.predict_batch(texts)
for p in predictions:
    print(f"{p.label_name}: {texts[...]}")
```

## 评测指标

评测脚本输出以下指标：
- **Accuracy**（准确率）
- **Precision**（精确率，macro + weighted）
- **Recall**（召回率，macro + weighted）
- **F1 Score**（F1 分数，macro + weighted）
- **Per-Class Accuracy**（各类别准确率）
- **Confusion Matrix**（混淆矩阵）

## 常见问题

### Q: 首次运行报错 `TypeError: expected str, bytes or os.PathLike object`
A: 模型名称含 "roberta" 但架构是 BERT。本模块已正确使用 `BertModel` + `BertTokenizer` 显式加载。如果你自行编写加载代码，请务必使用 BERT 类而非 Auto 类。

### Q: 下载模型很慢？
A: 模型约 400MB，首次从 HuggingFace Hub 下载。可设置镜像：`export HF_ENDPOINT=https://hf-mirror.com` 然后运行。

### Q: 内存不足？
A: 使用 INT8 量化：`python eval.py --quantize`，内存从 ~800MB 降至 ~300MB。

### Q: 怎么切换到 MacBERT 备选模型？
A: 修改 `classifier.py` 中 `InferencePipeline` 的默认 `model_name` 参数为 `"hfl/chinese-macbert-base"`，或传入 `create_pipeline(model_name="hfl/chinese-macbert-base")`。架构完全相同，无需其他改动。

## 模型信息

| 项目 | 详情 |
|------|------|
| 模型 | hfl/chinese-roberta-wwm-ext |
| 架构 | BERT-base (768D, 12L, 12H) |
| 参数量 | ~110M |
| 词表大小 | 21,128 |
| 最大长度 | 512 tokens |
| License | Apache 2.0 |
| 来源 | HuggingFace Hub 自动下载 |
