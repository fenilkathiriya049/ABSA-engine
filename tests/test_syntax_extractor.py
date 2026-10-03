import pytest
from src.features.sequence_features import extract_crf_features_from_text
from src.features.syntactic_extractor import extract_asc_features


def test_crf_feature_keys():
    text = "The pasta is good."
    features = extract_crf_features_from_text(text)
    assert len(features) > 0
    # First token must have BOS tag
    assert features[0]["BOS"] is True
    # Last token must have EOS tag
    assert features[-1]["EOS"] is True
    assert "pos" in features[0]
    assert "word.lower()" in features[0]


def test_syntactic_extractor_positive_modifier():
    text = "The crust was thin and crunchy."
    feats = extract_asc_features(text, "crust")
    assert feats["has_negation"] == 0
    assert "crunchy" in feats["modifiers"] or "thin" in feats["modifiers"]
    assert feats["head_lemma"] in ("be", "was")


def test_syntactic_extractor_negation_detection():
    text = "The food was not good."
    feats = extract_asc_features(text, "food")
    assert feats["has_negation"] == 1
    assert "good" in feats["modifiers"]


def test_contrastive_sentence_isolation():
    text = "The food was delicious, but the staff was rude."
    
    feats_food = extract_asc_features(text, "food")
    feats_staff = extract_asc_features(text, "staff")

    assert "delicious" in feats_food["modifiers"]
    assert "rude" not in feats_food["modifiers"]

    assert "rude" in feats_staff["modifiers"]
    assert "delicious" not in feats_staff["modifiers"]