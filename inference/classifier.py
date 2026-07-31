"""
Chinese Text Classification Inference Module
=============================================
Model: hfl/chinese-roberta-wwm-ext (BERT-base architecture, NOT RoBERTa)
Labels: 10-class multi-class classification

Pitfall: despite the "roberta" in the model name, the underlying architecture
is BERT-base. MUST use BertModel + BertTokenizer explicitly. AutoModel /
AutoTokenizer will route to RoBERTa classes and fail.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# ---------------------------------------------------------------------------
# Label definitions (fixed per task #1 specification, do NOT modify)
# ---------------------------------------------------------------------------
LABEL_MAP: List[str] = [
    "财经",       # 0
    "科技",       # 1
    "教育",       # 2
    "体育",       # 3
    "娱乐",       # 4
    "时政",       # 5
    "社会",       # 6
    "房产",       # 7
    "健康",       # 8
    "军事",       # 9
]

NUM_LABELS: int = len(LABEL_MAP)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class TextClassifier(nn.Module):
    """BERT-base backbone + Linear(768, 10) classification head.

    Uses the [CLS] token hidden state (first token) as the sentence-level
    representation, following the standard BERT classification paradigm.
    """

    def __init__(
        self,
        model_name: str = "hfl/chinese-roberta-wwm-ext",
        num_labels: int = NUM_LABELS,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        # CRITICAL: use BertModel, not AutoModel (see module docstring)
        self.bert: BertModel = BertModel.from_pretrained(model_name)
        self.dropout: nn.Dropout = nn.Dropout(dropout_rate)
        self.classifier: nn.Linear = nn.Linear(
            self.bert.config.hidden_size, num_labels
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # pooler_output uses a trained linear+tanh projection;
        # we use raw [CLS] hidden state instead for the custom head.
        cls_hidden: torch.Tensor = outputs.last_hidden_state[:, 0, :]
        pooled: torch.Tensor = self.dropout(cls_hidden)
        logits: torch.Tensor = self.classifier(pooled)
        return logits


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    """Single prediction result."""

    label_id: int
    label_name: str
    confidence: float
    top_k: List[Tuple[int, str, float]]  # (label_id, label_name, prob)


class InferencePipeline:
    """End-to-end inference pipeline: tokenize → encode → predict → decode.

    Usage::

        pipe = InferencePipeline(quantize=True)
        pipe.load()
        result = pipe.predict("央行宣布下调存款准备金率")
        print(result.label_name)  # "财经"
    """

    def __init__(
        self,
        model_name: str = "hfl/chinese-roberta-wwm-ext",
        max_length: int = 512,
        device: Optional[str] = None,
        quantize: bool = False,
        quantize_dtype: torch.dtype = torch.qint8,
    ):
        self.model_name: str = model_name
        self.max_length: int = max_length
        self.device: torch.device = self._resolve_device(device)
        self.quantize: bool = quantize
        self.quantize_dtype: torch.dtype = quantize_dtype

        self.tokenizer: Optional[BertTokenizer] = None
        self.model: Optional[TextClassifier] = None
        self._loaded: bool = False

    # ----- device resolution -----

    @staticmethod
    def _resolve_device(preferred: Optional[str]) -> torch.device:
        if preferred:
            return torch.device(preferred)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ----- load -----

    def load(self) -> InferencePipeline:
        """Load tokenizer and model, optionally applying INT8 dynamic quantization."""
        if self._loaded:
            return self

        log.info("Loading tokenizer: %s", self.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        log.info("Loading backbone: %s", self.model_name)
        self.model = TextClassifier(
            model_name=self.model_name, num_labels=NUM_LABELS
        )

        if self.quantize:
            log.info(
                "Applying dynamic INT8 quantization "
                "(Linear layers → qint8, embedding stays fp32)"
            )
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear},
                dtype=self.quantize_dtype,
            )

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True

        # report memory footprint (approximate)
        param_count: int = sum(p.numel() for p in self.model.parameters())
        byte_per_param: int = 1 if self.quantize else 4
        footprint_mb: float = (param_count * byte_per_param) / (1024 * 1024)
        log.info(
            "Model loaded. params=%s device=%s quantized=%s footprint≈%.0fMB",
            param_count,
            self.device,
            self.quantize,
            footprint_mb,
        )
        return self

    # ----- predict -----

    @torch.no_grad()
    def predict(self, text: str) -> Prediction:
        """Classify a single text string.

        Returns a Prediction with the top-1 label and full top-k ranking.
        """
        self._ensure_loaded()
        encoding = self._encode([text])
        logits: torch.Tensor = self.model(**encoding)
        probs: torch.Tensor = torch.softmax(logits, dim=-1).squeeze(0)
        return self._decode(probs)

    @torch.no_grad()
    def predict_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[Prediction]:
        """Classify a list of texts in mini-batches."""
        self._ensure_loaded()
        results: List[Prediction] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoding = self._encode(batch_texts)
            logits: torch.Tensor = self.model(**encoding)
            probs: torch.Tensor = torch.softmax(logits, dim=-1)
            for j in range(probs.size(0)):
                results.append(self._decode(probs[j]))
        return results

    @torch.no_grad()
    def predict_labels(
        self, texts: List[str], batch_size: int = 32
    ) -> List[int]:
        """Return raw label IDs for a list of texts (for evaluation)."""
        self._ensure_loaded()
        label_ids: List[int] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoding = self._encode(batch_texts)
            logits: torch.Tensor = self.model(**encoding)
            preds: torch.Tensor = torch.argmax(logits, dim=-1)
            label_ids.extend(preds.cpu().tolist())
        return label_ids

    # ----- internals -----

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

    def _encode(
        self, texts: List[str]
    ) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in encoding.items()}

    @staticmethod
    def _decode(probs: torch.Tensor) -> Prediction:
        probs_cpu: torch.Tensor = probs.cpu()
        sorted_indices = torch.argsort(probs_cpu, descending=True)
        top_k: List[Tuple[int, str, float]] = [
            (int(idx), LABEL_MAP[int(idx)], round(float(probs_cpu[idx]), 6))
            for idx in sorted_indices
        ]
        best: int = int(sorted_indices[0])
        return Prediction(
            label_id=best,
            label_name=LABEL_MAP[best],
            confidence=round(float(probs_cpu[best]), 6),
            top_k=top_k,
        )

    # ----- utilities -----

    @property
    def labels(self) -> List[str]:
        return LABEL_MAP

    @property
    def num_labels(self) -> int:
        return NUM_LABELS


# ---------------------------------------------------------------------------
# Convenience top-level function
# ---------------------------------------------------------------------------

def create_pipeline(
    quantize: bool = False,
    device: Optional[str] = None,
) -> InferencePipeline:
    """Factory: create and load an InferencePipeline in one call."""
    pipe = InferencePipeline(quantize=quantize, device=device)
    pipe.load()
    return pipe


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pipe = create_pipeline()

    samples: List[str] = [
        "央行宣布下调存款准备金率0.5个百分点",
        "苹果发布新一代M4芯片，性能提升50%",
        "教育部发布2025年高考改革方案",
        "中国女篮获得亚洲杯冠军",
        "春节档电影票房突破80亿元",
        "国务院常务会议部署稳经济一揽子政策",
        "全国多地迎来入冬以来最强降雪",
        "一线城市二手房成交量环比上涨15%",
        "国家药监局批准首款国产mRNA疫苗上市",
        "新型驱逐舰正式入列海军",
    ]

    for text in samples:
        pred: Prediction = pipe.predict(text)
        print(f"[{pred.label_name}] (conf={pred.confidence:.4f})  {text}")
