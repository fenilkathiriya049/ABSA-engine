from pathlib import Path
from typing import List, Dict, Any, Union
import joblib
import yaml

from src.features.sequence_features import extract_crf_features_from_tokens, nlp
from src.features.syntactic_extractor import extract_asc_features
from src.preprocessing.text_cleaner import clean_text


class ABSAPipeline:
    """
    End-to-end inference engine for Aspect-Based Sentiment Analysis.
    Combines CRF token sequence labeling with Syntactic Dependency Classification.
    """
    def __init__(self, ate_model_path: Union[str, Path], asc_pipeline_path: Union[str, Path]):
        self.ate_model_path = Path(ate_model_path)
        self.asc_pipeline_path = Path(asc_pipeline_path)
        
        if not self.ate_model_path.exists():
            raise FileNotFoundError(f"ATE model artifact not found at {self.ate_model_path}")
        if not self.asc_pipeline_path.exists():
            raise FileNotFoundError(f"ASC pipeline artifact not found at {self.asc_pipeline_path}")

        self.ate_model = joblib.load(self.ate_model_path)
        self.asc_pipeline = joblib.load(self.asc_pipeline_path)

    @classmethod
    def from_config(cls, config_path: Union[str, Path] = "configs/config.yaml") -> "ABSAPipeline":
        """
        Initializes the pipeline directly using path definitions from config.yaml.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        ate_path = config["artifacts"]["ate_model"]
        asc_path = config["artifacts"]["asc_model"]
        return cls(ate_model_path=ate_path, asc_pipeline_path=asc_path)

    def extract_aspects(self, text: str) -> List[Dict[str, Any]]:
        """
        Runs spaCy tokenization and uses CRF to predict BIO sequence labels,
        reconstructing multi-word aspect spans with character offsets.
        """
        cleaned_text = clean_text(text)
        doc = nlp(cleaned_text)
        tokens = [token.text for token in doc]

        # Extract CRF token features and predict labels
        features = extract_crf_features_from_tokens(tokens)
        bio_tags = self.ate_model.predict([features])[0]

        aspects = []
        current_tokens = []
        start_char = None
        end_char = None

        for token, tag in zip(doc, bio_tags):
            if tag == "B-ASP":
                # Flush existing completed aspect
                if current_tokens:
                    aspects.append({
                        "term": " ".join(current_tokens),
                        "start": start_char,
                        "end": end_char
                    })
                    current_tokens = []
                current_tokens.append(token.text)
                start_char = token.idx
                end_char = token.idx + len(token.text)

            elif tag == "I-ASP" and current_tokens:
                current_tokens.append(token.text)
                end_char = token.idx + len(token.text)

            else:
                if current_tokens:
                    aspects.append({
                        "term": " ".join(current_tokens),
                        "start": start_char,
                        "end": end_char
                    })
                    current_tokens = []
                    start_char = None
                    end_char = None

        if current_tokens:
            aspects.append({
                "term": " ".join(current_tokens),
                "start": start_char,
                "end": end_char
            })

        return aspects

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Performs end-to-end ABSA on a given raw input string:
        1. Normalizes text
        2. Discovers aspect entities (ATE)
        3. Classifies sentiment per discovered entity (ASC)
        """
        cleaned_text = clean_text(text)
        extracted_aspects = self.extract_aspects(cleaned_text)
        aspect_results = []

        for item in extracted_aspects:
            aspect_term = item["term"]
            
            # Extract syntax & dependency graph features
            asc_feat = extract_asc_features(cleaned_text, aspect_term)
            
            # Predict polarity and confidence
            sentiment = self.asc_pipeline.predict([asc_feat])[0]
            probabilities = self.asc_pipeline.predict_proba([asc_feat])[0]
            confidence = float(max(probabilities))

            aspect_results.append({
                "term": aspect_term,
                "sentiment": sentiment,
                "confidence": round(confidence, 4),
                "span": [item["start"], item["end"]],
                "syntactic_modifiers": asc_feat.get("modifiers", ""),
                "negated": bool(asc_feat.get("has_negation", 0))
            })

        return {
            "text": cleaned_text,
            "total_aspects": len(aspect_results),
            "aspects": aspect_results
        }


if __name__ == "__main__":
    pipeline = ABSAPipeline.from_config()
    sample = "The crust is thin and crunchy, but the staff is aloof."
    output = pipeline.predict(sample)
    
    import json
    print(json.dumps(output, indent=2))