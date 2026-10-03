import spacy
from typing import Dict, Any, List, Set, Optional

nlp = spacy.load("en_core_web_sm")

NEGATION_TOKENS: Set[str] = {"not", "never", "n't", "no", "hardly", "barely", "scarcely", "without"}
MODIFIER_DEPS: Set[str] = {"amod", "acomp", "advmod", "dobj", "attr", "oprd", "conj"}


def find_aspect_token_span(doc: spacy.tokens.Doc, aspect_term: str) -> List[spacy.tokens.Token]:
    """
    Finds the sub-sequence of tokens inside Doc that matches the aspect string.
    """
    aspect_words = aspect_term.strip().lower().split()
    if not aspect_words:
        return []

    doc_len = len(doc)
    span_len = len(aspect_words)

    for i in range(doc_len - span_len + 1):
        window_texts = [doc[i + j].text.lower() for j in range(span_len)]
        if window_texts == aspect_words:
            return [doc[i + j] for j in range(span_len)]

    # Fallback: substring matching on single tokens
    matched = [t for t in doc if t.text.lower() in aspect_words]
    return matched


def is_negated_node(token: spacy.tokens.Token) -> bool:
    """
    Checks if a token has a direct negation child or is itself a negation cue.
    """
    if token.text.lower() in NEGATION_TOKENS or token.lemma_.lower() in NEGATION_TOKENS:
        return True
    for child in token.children:
        if child.dep_ == "neg" or child.text.lower() in NEGATION_TOKENS:
            return True
    return False


def extract_asc_features(text: str, aspect_term: str) -> Dict[str, Any]:
    """
    Navigates the dependency tree from the aspect target outward to collect
    syntactic paths, modifiers, and negation context for sentiment polarity classification.
    """
    doc = nlp(text)
    aspect_tokens = find_aspect_token_span(doc, aspect_term)

    if not aspect_tokens:
        return {
            "aspect_lemma": aspect_term.lower(),
            "head_lemma": "NONE",
            "head_pos": "NONE",
            "modifiers": "",
            "has_negation": 0,
            "dep_signature": "NONE",
            "context_bow": text.lower()
        }

    # Use the root of the aspect span (last token in multi-word nominal compounds)
    target_node = aspect_tokens[-1]
    head_node = target_node.head

    modifiers: List[str] = []
    negated: bool = is_negated_node(target_node)

    # 1. Direct children of the aspect term (e.g., "crunchy [crust]")
    for child in target_node.children:
        if child not in aspect_tokens:
            if child.dep_ in MODIFIER_DEPS and child.pos_ in ("ADJ", "ADV", "VERB"):
                modifiers.append(child.lemma_.lower())
            if is_negated_node(child):
                negated = True

    # 2. Modifiers attached to the common governor (e.g., "[crust] is crunchy")
    if head_node != target_node:
        if is_negated_node(head_node):
            negated = True

        for sibling in head_node.children:
            if sibling not in aspect_tokens:
                if sibling.dep_ in MODIFIER_DEPS and sibling.pos_ in ("ADJ", "ADV", "NOUN"):
                    modifiers.append(sibling.lemma_.lower())
                    # Check conjunctions chained to the modifier (e.g., "fresh and delicious")
                    for conj in sibling.children:
                        if conj.dep_ == "conj" and conj.pos_ in ("ADJ", "ADV"):
                            modifiers.append(conj.lemma_.lower())
                if is_negated_node(sibling):
                    negated = True

    # 3. Construct a structural signature of the syntactic dependency
    dep_signature = f"{target_node.dep_}->{head_node.pos_}"

    return {
        "aspect_lemma": target_node.lemma_.lower(),
        "head_lemma": head_node.lemma_.lower(),
        "head_pos": head_node.pos_,
        "modifiers": " ".join(sorted(set(modifiers))),
        "has_negation": 1 if negated else 0,
        "dep_signature": dep_signature,
        "context_bow": " ".join([t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct])
    }