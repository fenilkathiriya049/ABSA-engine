import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import joblib
import yaml
import sklearn_crfsuite

from src.features.sequence_features import extract_crf_features_from_tokens


def load_ate_dataset(jsonl_path: Path) -> Tuple[List[List[Dict[str, Any]]], List[List[str]]]:
    """
    Reads JSONL processed data and converts token lists into CRF feature dictionaries.
    """
    X = []
    y = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            tokens = item["tokens"]
            tags = item["bio_tags"]

            features = extract_crf_features_from_tokens(tokens)
            X.append(features)
            y.append(tags)

    return X, y


def train_ate_model(config_path: str = "configs/config.yaml") -> sklearn_crfsuite.CRF:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = Path(config["data"]["ate_train"])
    model_save_path = Path(config["artifacts"]["ate_model"])
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ATE] Loading training data from: {data_path}")
    X_train, y_train = load_ate_dataset(data_path)
    print(f"[ATE] Loaded {len(X_train)} training sequences.")

    crf_cfg = config.get("crf_params", {})
    crf = sklearn_crfsuite.CRF(
        algorithm=crf_cfg.get("algorithm", "lbfgs"),
        c1=crf_cfg.get("c1", 0.1),
        c2=crf_cfg.get("c2", 0.1),
        max_iterations=crf_cfg.get("max_iterations", 100),
        all_possible_transitions=crf_cfg.get("all_possible_transitions", True)
    )

    print("[ATE] Fitting CRF sequence labeling model...")
    crf.fit(X_train, y_train)

    joblib.dump(crf, model_save_path)
    print(f"[ATE] Model successfully serialized to: {model_save_path}")
    return crf


if __name__ == "__main__":
    train_ate_model()