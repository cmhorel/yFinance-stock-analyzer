# news_analyzer/sentiment_analyzer.py
"""
Sentiment analysis module with multiple model support and fallback mechanisms.
"""
import logging
from typing import Optional, Dict, Any
from textblob import TextBlob
from transformers import pipeline
import torch

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Handles sentiment analysis with multiple model fallbacks."""
    
    def __init__(self):
        self.analyzer = None
        self.model_type = None
        self._initialize_analyzer()
    
    def _initialize_analyzer(self) -> None:
        """Initialize sentiment analyzer with fallback options."""
        # Try FinBERT first (financial sentiment)
        try:
            self.analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=0 if torch.cuda.is_available() else -1
            )
            self.model_type = "finbert"
            logger.info("Initialized FinBERT sentiment analyzer")
            return
        except Exception as e:
            logger.warning(f"Failed to load FinBERT: {e}")
        
        # Try RoBERTa fallback
        try:
            self.analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            self.model_type = "roberta"
            logger.info("Initialized RoBERTa sentiment analyzer")
            return
        except Exception as e:
            logger.warning(f"Failed to load RoBERTa: {e}")
        
        # Final fallback to TextBlob
        self.analyzer = None
        self.model_type = "textblob"
        logger.info("Using TextBlob sentiment analyzer as fallback")
    
    def analyze(self, text: str) -> float:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        if not text or not text.strip():
            return 0.0
        
        if self.analyzer is None:
            return self._analyze_with_textblob(text)
        
        try:
            # Truncate text to avoid token limits
            text = text[:2000]
            result = self.analyzer(text)[0]
            
            if self.model_type == "finbert":
                return self._process_finbert_result(result)
            elif self.model_type == "roberta":
                return self._process_roberta_result(result)
            
        except Exception as e:
            logger.error(f"Error in transformer sentiment analysis: {e}")
            return self._analyze_with_textblob(text)
        
        return 0.0
    
    def _analyze_with_textblob(self, text: str) -> float:
        """Fallback sentiment analysis using TextBlob."""
        try:
            analysis = TextBlob(text)
            return analysis.sentiment.polarity
        except Exception as e:
            logger.error(f"TextBlob analysis failed: {e}")
            return 0.0
    
    def _process_finbert_result(self, result: Dict[str, Any]) -> float:
        """Process FinBERT model results."""
        label = result['label'].lower()
        score = result['score']
        
        if label == 'positive':
            return score
        elif label == 'negative':
            return -score
        else:  # neutral
            return 0.0
    
    def _process_roberta_result(self, result: Dict[str, Any]) -> float:
        """Process RoBERTa model results."""
        label = result['label']
        score = result['score']
        
        if label == 'LABEL_2':  # positive
            return score
        elif label == 'LABEL_0':  # negative
            return -score
        else:  # neutral
            return 0.0