from pathlib import Path
from typing import Tuple, List, Dict, Any
import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.syntactic_extractor import extract_asc_features


def load_asc_dataset(csv_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Reads processed CSV dataset and extracts dependency-based feature dictionaries.
    """
    df = pd.read_csv(csv_path)
    X_features = []
    y_labels = []

    for _, row in df.iterrows():
        text = str(row["text"])
        aspect = str(row["aspect"])
        polarity = str(row["polarity"])

        feats = extract_asc_features(text, aspect)
        X_features.append(feats)
        y_labels.append(polarity)

    return X_features, y_labels


def train_asc_model(config_path: str = "configs/config.yaml") -> Pipeline:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = Path(config["data"]["asc_train"])
    model_save_path = Path(config["artifacts"]["asc_model"])
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ASC] Loading training instances from: {data_path}")
    X_train, y_train = load_asc_dataset(data_path)
    print(f"[ASC] Loaded {len(X_train)} labeled aspect instances.")

    asc_cfg = config.get("asc_params", {})
    pipeline = Pipeline([
        ("vectorizer", DictVectorizer(sparse=True)),
        ("classifier", LogisticRegression(
            penalty=asc_cfg.get("penalty", "l2"),
            C=asc_cfg.get("C", 1.0),
            max_iter=asc_cfg.get("max_iter", 500),
            class_weight=asc_cfg.get("class_weight", "balanced"),
            random_state=asc_cfg.get("random_state", 42)
        ))
    ])

    print("[ASC] Training LogisticRegression with syntactic features...")
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, model_save_path)
    print(f"[ASC] Pipeline successfully serialized to: {model_save_path}")
    return pipeline


if __name__ == "__main__":
    train_asc_model()