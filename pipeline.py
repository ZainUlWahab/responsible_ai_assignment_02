"""
Production Guardrail Pipeline for Toxicity Moderation

Three-layer system:
1. Input filter (regex pre-filter) - fast, high-precision blocks
2. Calibrated model layer - learned decision boundary
3. Human review queue - uncertainty handling
"""

import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os


# ==============================================================================
# LAYER 1: REGEX BLOCKLIST
# ==============================================================================

BLOCKLIST = {
    "direct_threat": [
        # Explicit threat to kill/murder/shoot/hurt/stab
        re.compile(r"\b(?:i|you|he|she|they)\s+(?:will|gonna|am going to|are going to)\s+(?:kill|murder|shoot|stab|hurt|beat|destroy)\s+(?:you|me|him|her|them)\b", re.IGNORECASE),
        # "you're going to die" / "you will die"
        re.compile(r"\byou\s+(?:will|are gonna|are going to)\s+(?:die|get killed|be killed)\b", re.IGNORECASE),
        # "I'll find where you live / your address"
        re.compile(r"\b(?:i|we)\s+(?:will|gonna|am going to)\s+find\s+(?:where\s+)?you\s+(?:live|are|stay)\b", re.IGNORECASE),
        # "I should/you should [threat verb]"
        re.compile(r"\b(?:someone|i|you)\s+should\s+(?:kill|murder|hurt|shoot|stab)\s+(?:you|me|yourself|themself|himself|herself)\b", re.IGNORECASE),
        # "I'll end you / finish you"
        re.compile(r"\b(?:i|we)\s+(?:will|gonna|am going to)\s+(?:end|finish|destroy|annihilate)\s+(?:you|me)\b", re.IGNORECASE),
        # "come at me / I'll beat you"
        re.compile(r"\b(?:i|we)\s+(?:will|am going to|gonna)\s+(?:beat|thrash|pummel|assault)\s+(?:you|your)\b", re.IGNORECASE),
    ],
    
    "self_harm_directed": [
        # "you should kill yourself / you should hurt yourself"
        re.compile(r"\byou\s+(?:should|must|need to)\s+(?:kill|hurt|harm|injure|cut)\s+yourself\b", re.IGNORECASE),
        # "go kill yourself / go hurt yourself"
        re.compile(r"\b(?:go|just)\s+(?:kill|hurt|harm|slit)\s+yourself\b", re.IGNORECASE),
        # "nobody would miss you / you don't deserve to live"
        re.compile(r"\b(?:nobody|no one|everyone)\s+(?:would|will)\s+(?:miss|care about|mourn)\s+you\b", re.IGNORECASE),
        # "do everyone a favour and disappear/die"
        re.compile(r"\b(?:do|make)\s+(?:everyone|us|the world)\s+a\s+(?:favour|favor|service)\s+and\s+(?:disappear|die|kill\s+yourself)\b", re.IGNORECASE),
        # "you should end it / finish it"
        re.compile(r"\byou\s+should\s+(?:end|finish)\s+(?:it|yourself)\b", re.IGNORECASE),
    ],
    
    "doxxing_stalking": [
        # "I know where you live / I found your address"
        re.compile(r"\b(?:i|we)\s+(?:know|found)\s+(?:where\s+)?you\s+(?:live|are|stay|home)\b", re.IGNORECASE),
        # "I'll post your address / I'll share your info"
        re.compile(r"\b(?:i|we)\s+(?:will|gonna|am going to)\s+(?:post|share|leak|release|expose)\s+your\s+(?:address|address|location|info|real\s+name|personal\s+info)\b", re.IGNORECASE),
        # "I found your real name / I know who you really are"
        re.compile(r"\b(?:i|we)\s+(?:found|know)\s+(?:your\s+)?(?:real\s+)?name\b", re.IGNORECASE),
        # "everyone will know who you really are"
        re.compile(r"\b(?:everyone|everyone\s+here)\s+(?:will|is gonna)\s+know\s+(?:who|your|your\s+real)\s+", re.IGNORECASE),
        # "I'm tracking you / I'm following you"
        re.compile(r"\b(?:i|we)\s+(?:am|are)\s+(?:tracking|following|stalking)\s+you\b", re.IGNORECASE),
    ],
    
    "dehumanization": [
        # "[group] are not human / [group] are animals"
        re.compile(r"\b(?:they|these|those|you|all)\s+(?:are|aren't|arent|is|isn't|isnt)\s+(?:human|people|person|real\s+humans)\b", re.IGNORECASE),
        # "[group] are animals / [group] are vermin"
        re.compile(r"\b(?:they|these|those|you|all)\s+(?:are|is)\s+(?:animals|pigs|rats|vermin|insects|cockroaches|dogs|beasts)\b", re.IGNORECASE),
        # "[group] should be exterminated / eliminated"
        re.compile(r"\b(?:they|these|those|you|all)\s+(?:should|must|need to)\s+be\s+(?:exterminated|eliminated|eradicated|destroyed)\b", re.IGNORECASE),
        # "[group] are a disease / plague / virus"
        re.compile(r"\b(?:they|these|those|you|all)\s+(?:are|is)\s+a\s+(?:disease|plague|virus|cancer|parasite|infection|blight)\b", re.IGNORECASE),
        # "[group] don't deserve to exist"
        re.compile(r"\b(?:they|these|those|you)\s+(?:don't|dont|don't|shouldnt|should not)\s+(?:deserve|have the right)\s+to\s+(?:live|exist)\b", re.IGNORECASE),
    ],
    
    "coordinated_harassment": [
        # "everyone report [target]" / "let's all report"
        re.compile(r"\b(?:let's|lets|everyone)\s+(?:all\s+)?(?:report|flag|block|ban)\s+(?:this|that|the|his|her|them)?\s*(?:account|user|guy|girl|person)?\b", re.IGNORECASE),
        # "let's go after / raid / attack"
        re.compile(r"\b(?:let's|lets|we|us|everyone)\s+(?:all\s+)?(?:go\s+after|raid|attack|gang\s+up\s+on|dogpile)\b", re.IGNORECASE),
        # "raid their profile / raid this account"
        re.compile(r"\b(?:raid|brigade|mass\s+report)\s+(?:this|that|his|her|their)\s+(?:profile|account|page|channel)\b", re.IGNORECASE),
        # "mass report this / everyone spam report"
        re.compile(r"\b(?:mass\s+report|spam\s+report|group\s+report|collective\s+report|brigade\s+report)\s+(?:this|that)?\b", re.IGNORECASE),
        # "someone dox this person / everyone find their info"
        re.compile(r"\b(?:someone|everyone|we|us)\s+(?:should|let's|lets)\s+(?:dox|find.*info|expose|leak)\b", re.IGNORECASE),
    ],
}

# Compile all patterns
for category in BLOCKLIST:
    BLOCKLIST[category] = [p for p in BLOCKLIST[category]]


def input_filter(text: str) -> dict | None:
    """
    Layer 1: Fast regex-based pre-filter.
    
    Args:
        text: Comment text to filter
        
    Returns:
        Block decision dict if matched, None otherwise
    """
    for category, patterns in BLOCKLIST.items():
        for pattern in patterns:
            if pattern.search(text):
                return {
                    "decision": "block",
                    "layer": "input_filter",
                    "category": category,
                    "confidence": 1.0
                }
    return None


# ==============================================================================
# LAYER 2: CALIBRATED MODEL
# ==============================================================================

class HFTextClassifier:
    """Wrapper for HuggingFace model predictions."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def predict_proba(self, texts):
        """Return probabilities for toxic class (class 1)."""
        if isinstance(texts, str):
            texts = [texts]
        
        probs = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs_text = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                probs.append(probs_text[1])  # probability of toxic (class 1)
        
        return np.array(probs)


# ==============================================================================
# LAYER 3: HUMAN REVIEW QUEUE
# ==============================================================================

class ModerationPipeline:
    """
    Three-layer moderation pipeline:
    1. Fast regex pre-filter
    2. Calibrated model with confidence-based decisions
    3. Human review queue for uncertain cases
    """
    
    def __init__(self, model_dir, device=None):
        """
        Initialize the pipeline.
        
        Args:
            model_dir: Path to best mitigated model checkpoint
            device: torch device (cuda or cpu)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=2
        ).to(device)
        self.model.eval()
        
        # Initialize classifier wrapper
        self.classifier = HFTextClassifier(self.model, self.tokenizer, device)
        
        # Try to load pre-fitted calibrator, otherwise we'll fit it during setup
        self.calibrator = None
        calib_path = os.path.join(model_dir, "calibrator.pkl")
        if os.path.exists(calib_path):
            import pickle
            with open(calib_path, "rb") as f:
                self.calibrator = pickle.load(f)
        
        # Uncertainty thresholds
        self.low_threshold = 0.4
        self.high_threshold = 0.6
    
    def set_uncertainty_thresholds(self, low, high):
        """Allow threshold adjustment for sensitivity analysis."""
        self.low_threshold = low
        self.high_threshold = high
    
    def fit_calibrator(self, texts, labels, method="isotonic"):
        """
        Fit isotonic calibration on a set of texts and labels.
        
        Args:
            texts: List of comment texts
            labels: Binary labels (0 or 1)
            method: Calibration method ('isotonic' or 'sigmoid')
        """
        # Get raw predictions (predict_proba returns 1D array of toxic probabilities)
        raw_probs = self.classifier.predict_proba(texts)
        
        # Fit calibrator directly on raw probabilities
        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(raw_probs, labels)
        else:
            # Sigmoid calibration using Platt scaling
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression()
            self.calibrator.fit(raw_probs.reshape(-1, 1), labels)
    
    def predict(self, text: str, return_raw=False) -> dict:
        """
        Three-layer moderation decision.
        
        Args:
            text: Comment to moderate
            return_raw: If True, include raw model probability
            
        Returns:
            Decision dict with layer, decision, and confidence
        """
        # Layer 1: Input filter
        filter_result = input_filter(text)
        if filter_result:
            return filter_result
        
        # Layer 2: Calibrated model
        raw_prob = self.classifier.predict_proba([text])[0]  # Probability of toxic class (predict_proba returns 1D array)
        
        if self.calibrator is not None:
            # Use calibrated probability
            from sklearn.isotonic import IsotonicRegression
            if isinstance(self.calibrator, IsotonicRegression):
                # IsotonicRegression returns a scalar
                calibrated_prob = float(self.calibrator.predict([raw_prob])[0])
            else:
                # LogisticRegression returns probability via predict_proba
                calibrated_prob = float(self.calibrator.predict_proba([[raw_prob]])[0, 1])
        else:
            calibrated_prob = raw_prob
        
        result = {
            "layer": "model",
            "confidence": float(calibrated_prob),
        }
        
        if return_raw:
            result["raw_probability"] = float(raw_prob)
        
        # Determine decision based on confidence thresholds
        if calibrated_prob >= self.high_threshold:
            result["decision"] = "block"
        elif calibrated_prob <= self.low_threshold:
            result["decision"] = "allow"
        else:
            result["decision"] = "review"
            result["layer"] = "human_review"
        
        return result
    
    def predict_batch(self, texts: list) -> list:
        """Predict on multiple texts."""
        return [self.predict(text) for text in texts]
