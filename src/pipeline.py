from pathlib import Path
import joblib
from typing import List, Dict, Any
from src.features.sequence_features import sent_to_features, nlp
from src.features.syntactic_extractor import extract_asc_features

class ABSAPipeline:
    def __init__(self, ate_model_path: Path, asc_pipeline_path: Path):
        self.ate_model = joblib.load(ate_model_path)
        self.asc_pipeline = joblib.load(asc_pipeline_path)

    def extract_aspect_terms(self, text: str) -> List[str]:
        doc = nlp(text)
        features = [sent_to_features(text)][0]
        bio_tags = self.ate_model.predict_single(features)
        
        aspects = []
        current_aspect = []
        
        for token, tag in zip(doc, bio_tags):
            if tag == "B-ASP":
                if current_aspect:
                    aspects.append(" ".join(current_aspect))
                    current_aspect = []
                current_aspect.append(token.text)
            elif tag == "I-ASP" and current_aspect:
                current_aspect.append(token.text)
            else:
                if current_aspect:
                    aspects.append(" ".join(current_aspect))
                    current_aspect = []
        if current_aspect:
            aspects.append(" ".join(current_aspect))
            
        return aspects

    def predict(self, text: str) -> List[Dict[str, Any]]:
        extracted_aspects = self.extract_aspect_terms(text)
        results = []

        for aspect in extracted_aspects:
            asc_feat = extract_asc_features(text, aspect)
            sentiment = self.asc_pipeline.predict([asc_feat])[0]
            confidence = max(self.asc_pipeline.predict_proba([asc_feat])[0])
            results.append({
                "aspect": aspect,
                "sentiment": sentiment,
                "confidence": round(float(confidence), 3)
            })

        return results