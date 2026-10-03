import spacy
from typing import List, Dict, Any

nlp = spacy.load("en_core_web_sm")

def token_to_features(doc: spacy.tokens.Doc, i: int) -> Dict[str, Any]:
    token = doc[i]
    features = {
        "bias": 1.0,
        "word.lower()": token.text.lower(),
        "word.isupper()": token.is_upper,
        "word.istitle()": token.is_title,
        "word.isdigit()": token.is_digit,
        "postag": token.pos_,
        "dep": token.dep_,
        "shape": token.shape_,
    }
    # Left context (i-1)
    if i > 0:
        prev_token = doc[i - 1]
        features.update({
            "-1:word.lower()": prev_token.text.lower(),
            "-1:postag": prev_token.pos_,
            "-1:dep": prev_token.dep_,
        })
    else:
        features["BOS"] = True  # Beginning of sentence

    # Right context (i+1)
    if i < len(doc) - 1:
        next_token = doc[i + 1]
        features.update({
            "+1:word.lower()": next_token.text.lower(),
            "+1:postag": next_token.pos_,
            "+1:dep": next_token.dep_,
        })
    else:
        features["EOS"] = True  # End of sentence

    return features

def sent_to_features(text: str) -> List[Dict[str, Any]]:
    doc = nlp(text)
    return [token_to_features(doc, i) for i in range(len(doc))]