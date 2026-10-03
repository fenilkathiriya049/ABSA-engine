from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn_crfsuite import metrics as crf_metrics


def evaluate_ate(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict[str, Any]:
    """
    Evaluates sequence labeling performance for Aspect Term Extraction.
    Excludes the 'O' (Outside) tag to calculate meaningful span scores.
    """
    labels = ["B-ASP", "I-ASP"]
    
    # Calculate macro and micro F1 across active aspect spans
    precision = crf_metrics.flat_precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    recall = crf_metrics.flat_recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    f1 = crf_metrics.flat_f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    
    report = crf_metrics.flat_classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0)

    return {
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "report": report
    }


def evaluate_asc(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Evaluates classification performance for Aspect Sentiment Classification.
    """
    labels = ["positive", "negative", "neutral"]
    existing_labels = sorted(list(set(y_true) | set(y_pred)))
    target_labels = [lbl for lbl in labels if lbl in existing_labels] or existing_labels

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=target_labels, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=target_labels)

    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in target_labels], columns=[f"pred_{l}" for l in target_labels])

    return {
        "accuracy": float(acc),
        "report": report,
        "confusion_matrix": cm_df
    }


def run_evaluation_suite(
    ate_model_path: str = "artifacts/crf_ate_model.joblib",
    asc_pipeline_path: str = "artifacts/logreg_asc_pipeline.joblib",
    ate_data_path: str = "data/processed/ate_train.jsonl",
    asc_data_path: str = "data/processed/asc_train.csv"
) -> None:
    """
    Loads saved artifacts and runs benchmark evaluations on processed datasets.
    """
    from pathlib import Path
    import joblib
    from src.models.ate_crf import load_ate_dataset
    from src.models.asc_classifier import load_asc_dataset

    print("\n" + "=" * 50)
    print("      EVALUATING ASPECT TERM EXTRACTION (ATE)")
    print("=" * 50)

    if Path(ate_model_path).exists() and Path(ate_data_path).exists():
        crf_model = joblib.load(ate_model_path)
        X_ate, y_ate_true = load_ate_dataset(Path(ate_data_path))
        y_ate_pred = crf_model.predict(X_ate)
        
        ate_results = evaluate_ate(y_ate_true, y_ate_pred)
        print(f"ATE Macro Precision: {ate_results['macro_precision']:.4f}")
        print(f"ATE Macro Recall:    {ate_results['macro_recall']:.4f}")
        print(f"ATE Macro F1 Score:  {ate_results['macro_f1']:.4f}")
        print("\nSpan Classification Report:\n")
        print(ate_results["report"])
    else:
        print("[SKIP] ATE model artifact or dataset missing.")

    print("\n" + "=" * 50)
    print("   EVALUATING ASPECT SENTIMENT CLASSIFICATION (ASC)")
    print("=" * 50)

    if Path(asc_pipeline_path).exists() and Path(asc_data_path).exists():
        asc_pipeline = joblib.load(asc_pipeline_path)
        X_asc, y_asc_true = load_asc_dataset(Path(asc_data_path))
        y_asc_pred = asc_pipeline.predict(X_asc)

        asc_results = evaluate_asc(y_asc_true, y_asc_pred)
        print(f"ASC Accuracy: {asc_results['accuracy']:.4f}")
        print("\nPolarity Classification Report:\n")
        print(asc_results["report"])
        print("\nConfusion Matrix:")
        print(asc_results["confusion_matrix"])
    else:
        print("[SKIP] ASC pipeline artifact or dataset missing.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_evaluation_suite()