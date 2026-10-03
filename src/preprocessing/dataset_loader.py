import xml.etree.ElementTree as ET
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import spacy

from src.preprocessing.text_cleaner import clean_text

nlp = spacy.load("en_core_web_sm")


def parse_semeval_xml(xml_file_path: Path) -> List[Dict[str, Any]]:
    """
    Parses a SemEval 2014 Task 4 XML file into a list of structured dictionaries.
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    records = []

    for sentence in root.findall("sentence"):
        sent_id = sentence.attrib.get("id", "")
        text_elem = sentence.find("text")
        if text_elem is None or not text_elem.text:
            continue
            
        text = clean_text(text_elem.text)
        aspect_terms = []
        
        aspect_terms_elem = sentence.find("aspectTerms")
        if aspect_terms_elem is not None:
            for term in aspect_terms_elem.findall("aspectTerm"):
                aspect_terms.append({
                    "term": term.attrib.get("term", "").strip(),
                    "polarity": term.attrib.get("polarity", "").lower().strip(),
                    "from": int(term.attrib.get("from", 0)),
                    "to": int(term.attrib.get("to", 0))
                })

        records.append({
            "id": sent_id,
            "text": text,
            "aspect_terms": aspect_terms
        })

    return records


def generate_bio_tags(text: str, aspect_spans: List[Tuple[int, int]]) -> Tuple[List[str], List[str]]:
    """
    Tokenizes text using spaCy and aligns character offsets to produce BIO tags.
    """
    doc = nlp(text)
    tokens = [token.text for token in doc]
    tags = ["O"] * len(doc)

    for start_char, end_char in aspect_spans:
        # Match tokens covering the character span
        matched_indices = []
        for idx, token in enumerate(doc):
            tok_start = token.idx
            tok_end = tok_start + len(token.text)
            
            # Check overlap between token range and aspect span
            if max(tok_start, start_char) < min(tok_end, end_char):
                matched_indices.append(idx)

        if matched_indices:
            tags[matched_indices[0]] = "B-ASP"
            for idx in matched_indices[1:]:
                tags[idx] = "I-ASP"

    return tokens, tags


def process_and_save_data(raw_xml_path: Path, processed_dir: Path) -> None:
    """
    Parses raw XML and exports both ate_train.jsonl and asc_train.csv.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    parsed_sentences = parse_semeval_xml(raw_xml_path)

    ate_records = []
    asc_records = []

    for item in parsed_sentences:
        text = item["text"]
        terms = item["aspect_terms"]

        # 1. ATE preparation
        spans = [(t["from"], t["to"]) for t in terms if t["term"]]
        tokens, tags = generate_bio_tags(text, spans)
        ate_records.append({
            "id": item["id"],
            "tokens": tokens,
            "bio_tags": tags
        })

        # 2. ASC preparation
        for t in terms:
            # We filter out 'conflict' polarities as is standard in SemEval benchmarks
            if t["polarity"] in ["positive", "negative", "neutral"]:
                asc_records.append({
                    "id": item["id"],
                    "text": text,
                    "aspect": t["term"],
                    "polarity": t["polarity"]
                })

    # Save ATE as JSONL
    ate_output_path = processed_dir / "ate_train.jsonl"
    with open(ate_output_path, "w", encoding="utf-8") as f:
        for record in ate_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[ATE] Exported {len(ate_records)} sequences to {ate_output_path}")

    # Save ASC as CSV
    asc_df = pd.DataFrame(asc_records)
    asc_output_path = processed_dir / "asc_train.csv"
    asc_df.to_csv(asc_output_path, index=False, encoding="utf-8")
    print(f"[ASC] Exported {len(asc_df)} labeled aspect instances to {asc_output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process SemEval XML dataset.")
    parser.add_argument("--raw_path", type=str, default="data/raw/Restaurants_Train.xml", help="Path to raw XML")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save processed datasets")
    args = parser.parse_args()

    process_and_save_data(Path(args.raw_path), Path(args.output_dir))