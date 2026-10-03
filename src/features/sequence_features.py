import spacy
from typing import List, Dict, Any, Union

nlp = spacy.load("en_core_web_sm")


def token_to_features(doc: spacy.tokens.Doc, i: int) -> Dict[str, Any]:
    """
    Constructs a rich feature dictionary for a single token at index `i`
    for sequence tagging with Conditional Random Fields.
    """
    token = doc[i]

    features: Dict[str, Any] = {
        "bias": 1.0,
        "word.lower()": token.text.lower(),
        "word.lemma": token.lemma_.lower(),
        "word[-3:]": token.text[-3:],
        "word[-2:]": token.text[-2:],
        "word[:3]": token.text[:3],
        "word[:2]": token.text[:2],
        "word.isupper": token.is_upper,
        "word.istitle": token.is_title,
        "word.isdigit": token.is_digit,
        "word.ispunct": token.is_punct,
        "pos": token.pos_,
        "tag": token.tag_,
        "dep": token.dep_,
        "shape": token.shape_,
    }

    # Left Context (Token i-1)
    if i > 0:
        prev_token = doc[i - 1]
        features.update({
            "-1:word.lower()": prev_token.text.lower(),
            "-1:word.istitle": prev_token.is_title,
            "-1:word.isupper": prev_token.is_upper,
            "-1:pos": prev_token.pos_,
            "-1:tag": prev_token.tag_,
            "-1:dep": prev_token.dep_,
        })
    else:
        features["BOS"] = True  # Beginning of sentence

    # Right Context (Token i+1)
    if i < len(doc) - 1:
        next_token = doc[i + 1]
        features.update({
            "+1:word.lower()": next_token.text.lower(),
            "+1:word.istitle": next_token.is_title,
            "+1:word.isupper": next_token.is_upper,
            "+1:pos": next_token.pos_,
            "+1:tag": next_token.tag_,
            "+1:dep": next_token.dep_,
        })
    else:
        features["EOS"] = True  # End of sentence

    return features


def extract_crf_features_from_tokens(tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Converts pre-tokenized strings into spaCy Doc to extract token feature dictionaries.
    """
    doc = spacy.tokens.Doc(nlp.vocab, words=tokens)
    # Run the tagger and parser components explicitly on pre-tokenized words
    for name, proc in nlp.pipeline:
        doc = proc(doc)
    return [token_to_features(doc, i) for i in range(len(doc))]


def extract_crf_features_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Processes raw text directly through spaCy's tokenizer and pipeline.
    """
    doc = nlp(text)
    return [token_to_features(doc, i) for i in range(len(doc))]