import spacy
from typing import Dict, Any, List

nlp = spacy.load("en_core_web_sm")

def extract_asc_features(text: str, aspect_term: str) -> Dict[str, Any]:
    doc = nlp(text)
    aspect_tokens = [t for t in doc if t.text.lower() in aspect_term.lower().split()]
    
    if not aspect_tokens:
        return {
            "aspect": aspect_term.lower(),
            "head": "NONE",
            "modifiers": "",
            "negated": "False",
            "dep_path": "NONE"
        }

    target = aspect_tokens[0]
    head = target.head
    modifiers: List[str] = []
    negated = False

    # Check direct children of the target aspect
    for child in target.children:
        if child.dep_ in ("amod", "advmod", "acomp"):
            modifiers.append(child.lemma_.lower())
        if child.dep_ == "neg" or child.lemma_.lower() in ("not", "never", "n't"):
            negated = True

    # Check siblings under the common head
    if head != target:
        for child in head.children:
            if child != target:
                if child.dep_ in ("acomp", "amod", "advmod", "dobj", "attr"):
                    modifiers.append(child.lemma_.lower())
                if child.dep_ == "neg" or child.lemma_.lower() in ("not", "never", "n't"):
                    negated = True

    return {
        "aspect": aspect_term.lower(),
        "head": head.lemma_.lower(),
        "modifiers": " ".join(modifiers),
        "negated": str(negated),
        "dep_path": f"{target.dep_}->{head.pos_}"
    }