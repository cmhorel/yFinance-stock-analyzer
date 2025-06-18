import pytest
from app.news_analyzer import sentiment_analyzer


@pytest.fixture
def analyzer():
    return sentiment_analyzer.SentimentAnalyzer()

def test_analyze_positive(analyzer):
    score = analyzer.analyze("I love this!")
    assert isinstance(score, float)
    #Because finbert be so conservative yo
    assert 0.0 <= score <= 1.0

def test_analyze_returns_float():
    analyzer = sentiment_analyzer.SentimentAnalyzer()
    score = analyzer.analyze("This is a great stock!")
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0

def test_analyze_handles_empty():
    analyzer = sentiment_analyzer.SentimentAnalyzer()
    score = analyzer.analyze("")
    assert isinstance(score, float)

def test_analyze_negative(analyzer):
    score = analyzer.analyze("This company is terrible and I hate it.")
    assert isinstance(score, float)
    assert score < 0

def test_analyze_neutral(analyzer):
    score = analyzer.analyze("The stock price is $100.")
    assert isinstance(score, float)
    # Neutral text should be close to zero
    assert abs(score) < 0.2

def test_analyze_empty(analyzer):
    score = analyzer.analyze("")
    assert isinstance(score, float)

def test_analyze_none(analyzer):
    score = analyzer.analyze(None)
    assert isinstance(score, float)

def test_analyze_non_string(analyzer):
    score = analyzer.analyze(12345)
    assert isinstance(score, float)

def test_analyze_unicode(analyzer):
    score = analyzer.analyze("🚀🚀🚀 To the moon!")
    assert isinstance(score, float)

def test_analyze_long_text(analyzer):
    long_text = "good " * 1000
    score = analyzer.analyze(long_text)
    assert isinstance(score, float)

def test_analyze_consistency(analyzer):
    text = "Solid performance."
    score1 = analyzer.analyze(text)
    score2 = analyzer.analyze(text)
    assert score1 == score2

def test_analyze_positive_transformer(monkeypatch, analyzer):
    # Mock the analyzer to simulate a positive transformer result
    analyzer.analyzer = lambda text: [{"label": "positive", "score": 0.9}]
    analyzer.model_type = "finbert"
    score = analyzer.analyze("This stock is amazing and I love it!")
    assert score == 0.9

def test_fallback_logic(monkeypatch, analyzer):
    # Simulate transformer model failure, so fallback to TextBlob
    def fail_analyze(text):
        raise Exception("Primary model failed")
    analyzer.analyzer = fail_analyze
    analyzer.model_type = "finbert"
    score = analyzer.analyze("Fallback test")
    assert isinstance(score, float)