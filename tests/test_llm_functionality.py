import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from nlplot_llm import NLPlotLLM

@pytest.fixture
def npt_instance():
    df = pd.DataFrame({"text": ["this is a test."]})
    return NLPlotLLM(df, target_col="text")

def test_analyze_sentiment_llm(npt_instance):
    with patch("litellm.completion") as mock_completion:
        mock_response = MagicMock()
        mock_message = MagicMock(content="positive")
        mock_choice = MagicMock(message=mock_message)
        mock_response.choices = [mock_choice]
        mock_completion.return_value = mock_response

        text_series = pd.Series(["I love this product!"])
        result = npt_instance.analyze_sentiment_llm(text_series, model="test_model")
        assert not result.empty
        assert result.iloc[0]["sentiment"] == "positive"

def test_categorize_text_llm(npt_instance):
    with patch("litellm.completion") as mock_completion:
        mock_response = MagicMock()
        mock_message = MagicMock(content="tech")
        mock_choice = MagicMock(message=mock_message)
        mock_response.choices = [mock_choice]
        mock_completion.return_value = mock_response

        text_series = pd.Series(["This is a text about technology."])
        categories = ["tech", "sports", "finance"]
        result = npt_instance.categorize_text_llm(text_series, categories, model="test_model")
        assert not result.empty
        assert result.iloc[0]["category"] == "tech"

def test_summarize_text_llm(npt_instance):
    with patch("litellm.completion") as mock_completion:
        mock_response = MagicMock()
        mock_message = MagicMock(content="This is a summary.")
        mock_choice = MagicMock(message=mock_message)
        mock_response.choices = [mock_choice]
        mock_completion.return_value = mock_response

        text_series = pd.Series(["This is a long text to be summarized."])
        result = npt_instance.summarize_text_llm(text_series, model="test_model")
        assert not result.empty
        assert result.iloc[0]["summary"] == "This is a summary."
