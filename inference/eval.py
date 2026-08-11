#!/usr/bin/env python3
"""
Evaluation script for Chinese text classification module.

Usage:
    python eval.py                          # default: FP32 on auto device
    python eval.py --quantize               # INT8 dynamic quantization
    python eval.py --device cpu             # force CPU
    python eval.py --output results.json    # save full prediction details

Outputs:
    - Console: sklearn classification_report (accuracy, precision, recall, F1)
    - Optional JSON: per-sample prediction details
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE: Path = Path(__file__).resolve().parent
DATA_PATH: Path = HERE / "sample_data.json"
CLASSIFIER_PATH: Path = HERE / "classifier.py"

sys.path.insert(0, str(HERE))

from classifier import LABEL_MAP, NUM_LABELS, InferencePipeline

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def load_sample_data(path: Path) -> List[Dict[str, Any]]:
    """Load evaluation samples from JSON file.

    Expected format:
    {
        "label_map": {"0": "财经", ...},
        "samples": [
            {"text": "...", "label": 0},
            ...
        ]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    samples: List[Dict[str, Any]] = data.get("samples", [])
    if not samples:
        raise ValueError("sample_data.json contains no samples")

    return samples


def validate_data(samples: List[Dict[str, Any]]) -> Dict[int, int]:
    """Check sample coverage: every label must have >= 20 samples."""
    counts: Dict[int, int] = {}
    for s in samples:
        lid: int = s["label"]
        counts[lid] = counts.get(lid, 0) + 1

    issues: List[str] = []
    for lid in range(NUM_LABELS):
        n: int = counts.get(lid, 0)
        if n < 20:
            issues.append(f"label {lid} ({LABEL_MAP[lid]}) has only {n} samples (need >= 20)")

    if issues:
        log.warning("Data coverage issues:")
        for msg in issues:
            log.warning("  %s", msg)

    return counts


def run_evaluation(
    pipe: InferencePipeline,
    samples: List[Dict[str, Any]],
    output_path: Path | None = None,
) -> Dict[str, Any]:
    """Run full evaluation: predict all samples, compute metrics, optionally save details."""

    texts: List[str] = [s["text"] for s in samples]
    y_true: List[int] = [s["label"] for s in samples]

    n_total: int = len(texts)
    log.info("Running inference on %d samples...", n_total)
    t_start: float = time.perf_counter()

    y_pred: List[int] = pipe.predict_labels(texts, batch_size=32)

    elapsed: float = time.perf_counter() - t_start
    throughput: float = n_total / elapsed

    # ---- metrics ----
    acc: float = float(accuracy_score(y_true, y_pred))
    prec_macro: float = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro: float = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro: float = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted: float = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted: float = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted: float = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # ---- per-class stats ----
    cm: np.ndarray = confusion_matrix(y_true, y_pred, labels=list(range(NUM_LABELS)))
    per_class: List[Dict[str, Any]] = []
    for lid in range(NUM_LABELS):
        tp: int = int(cm[lid, lid])
        total_true: int = int(cm[lid, :].sum())
        total_pred: int = int(cm[:, lid].sum())
        per_class.append({
            "label_id": lid,
            "label_name": LABEL_MAP[lid],
            "total_true": total_true,
            "total_pred": total_pred,
            "correct": tp,
            "accuracy_per_class": round(tp / total_true, 4) if total_true > 0 else 0.0,
        })

    # ---- summary ----
    report: str = classification_report(
        y_true,
        y_pred,
        target_names=LABEL_MAP,
        zero_division=0,
    )

    results: Dict[str, Any] = {
        "model": pipe.model_name,
        "device": str(pipe.device),
        "quantized": pipe.quantize,
        "total_samples": n_total,
        "inference_time_seconds": round(elapsed, 3),
        "throughput_samples_per_second": round(throughput, 1),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision_macro": round(prec_macro, 4),
            "recall_macro": round(rec_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_weighted": round(f1_weighted, 4),
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    # ---- save detailed results if requested ----
    if output_path:
        detail_results: Dict[str, Any] = {**results}
        detail_results["predictions"] = [
            {
                "text": s["text"],
                "true_label_id": y_true[i],
                "true_label_name": LABEL_MAP[y_true[i]],
                "pred_label_id": y_pred[i],
                "pred_label_name": LABEL_MAP[y_pred[i]],
                "correct": y_true[i] == y_pred[i],
            }
            for i, s in enumerate(samples)
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(detail_results, f, ensure_ascii=False, indent=2)
        log.info("Detailed results saved to: %s", output_path)

    return results


def print_results(results: Dict[str, Any]) -> None:
    """Pretty-print evaluation results to console."""

    metrics: Dict[str, float] = results["metrics"]
    per_class: List[Dict[str, Any]] = results["per_class"]

    sep: str = "=" * 64
    print(f"\n{sep}")
    print(f"  中文文本分类评测结果")
    print(f"{sep}")
    print(f"  Model:      {results['model']}")
    print(f"  Device:     {results['device']}")
    print(f"  Quantized:  {results['quantized']}")
    print(f"  Samples:    {results['total_samples']}")
    print(f"  Time:       {results['inference_time_seconds']}s")
    print(f"  Throughput: {results['throughput_samples_per_second']} samples/s")
    print(f"{sep}")

    print(f"\n--- Overall Metrics ---")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
    print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"  Recall (weighted):    {metrics['recall_weighted']:.4f}")
    print(f"  F1 (weighted):        {metrics['f1_weighted']:.4f}")

    print(f"\n--- Per-Class Accuracy ---")
    header: str = f"  {'ID':<3} {'Label':<6} {'Total':<6} {'Correct':<8} {'Acc':<8}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for pc in per_class:
        print(
            f"  {pc['label_id']:<3} "
            f"{pc['label_name']:<6} "
            f"{pc['total_true']:<6} "
            f"{pc['correct']:<8} "
            f"{pc['accuracy_per_class']:<8.4f}"
        )

    print(f"\n--- Classification Report ---")
    print(results["classification_report"])

    print(f"{sep}")
    print(f"  评测完成。")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chinese text classification evaluation"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 dynamic quantization before evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (cpu, cuda, mps). Default: auto-detect.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save full prediction details to a JSON file.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Path to sample data JSON. Default: sample_data.json",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Path to fine-tuned checkpoint directory. Default: inference/checkpoints/",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # ---- load data ----
    log.info("Loading sample data from: %s", args.data)
    samples: List[Dict[str, Any]] = load_sample_data(args.data)
    label_counts: Dict[int, int] = validate_data(samples)

    log.info("Sample distribution:")
    for lid in range(NUM_LABELS):
        log.info("  %s (%s): %d", lid, LABEL_MAP[lid], label_counts.get(lid, 0))
    log.info("  TOTAL: %d", len(samples))

    # ---- load model ----
    pipe: InferencePipeline = InferencePipeline(
        quantize=args.quantize,
        device=args.device,
        checkpoint_dir=str(args.checkpoint_dir) if args.checkpoint_dir else None,
    )
    pipe.load()

    # ---- evaluate ----
    results: Dict[str, Any] = run_evaluation(
        pipe,
        samples,
        output_path=args.output,
    )
    print_results(results)


if __name__ == "__main__":
    main()
