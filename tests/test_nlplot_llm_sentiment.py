import pytest
import os
from unittest.mock import patch, MagicMock
import pandas as pd

# Attempt to import langchain classes for type hinting and patching targets.
try:
    from langchain_openai import ChatOpenAI
    from langchain_community.chat_models.ollama import OllamaChat
    from langchain_core.outputs import AIMessage
    from langchain_core.prompts import PromptTemplate
    LANGCHAIN_TEST_IMPORTS_AVAILABLE = True
except ImportError:
    LANGCHAIN_TEST_IMPORTS_AVAILABLE = False
    # Dummy classes for type hints and for @patch to have a target if not installed.
    class AIMessage:
        def __init__(self, content): self.content = content
    class ChatOpenAI: pass
    class OllamaChat: pass
    class PromptTemplate: pass

from nlplot import NLPlot
try:
    # Assuming LANGCHAIN_AVAILABLE is defined in nlplot.nlplot module
    from nlplot.nlplot import LANGCHAIN_AVAILABLE as MODULE_LANGCHAIN_AVAILABLE
except ImportError:
    MODULE_LANGCHAIN_AVAILABLE = False


@pytest.fixture
def npt_llm_instance(tmp_path):
    """Provides a basic NLPlot instance for LLM tests, ensuring output path exists."""
    # Using a minimal DataFrame for NLPlot initialization as it's not the focus of these tests.
    df = pd.DataFrame({'text': ["initial setup text to satisfy NLPlot constructor"]})
    output_dir = tmp_path / "llm_sentiment_test_outputs"
    os.makedirs(output_dir, exist_ok=True) # Ensure the output path exists
    return NLPlot(df, target_col='text', output_file_path=str(output_dir))

# --- TDD for LLM Sentiment Analysis (Cycle 2) ---

def test_analyze_sentiment_llm_initial_method_missing(npt_llm_instance):
    """(Red Phase) Ensure analyze_sentiment_llm method is initially missing."""
    with pytest.raises(AttributeError, match="'NLPlot' object has no attribute 'analyze_sentiment_llm'"):
        npt_llm_instance.analyze_sentiment_llm(
            text_series=pd.Series(["a test sentence"]),
            llm_provider="openai",
            model_name="gpt-3.5-turbo"
        )

# The tests below are for the Green/Refactor phase, once analyze_sentiment_llm is implemented.
# They assume _get_llm_client is functional and correctly mocked where necessary.

@patch('nlplot.nlplot.NLPlot._get_llm_client')
def test_analyze_sentiment_llm_openai_positive(mock_get_llm, npt_llm_instance):
    if not MODULE_LANGCHAIN_AVAILABLE or not LANGCHAIN_TEST_IMPORTS_AVAILABLE:
        pytest.skip("Langchain or its dependencies not available, skipping LLM sentiment test.")

    mock_llm_openai = MagicMock(spec=ChatOpenAI)
    mock_llm_openai.invoke.return_value = AIMessage(content="positive")
    mock_get_llm.return_value = mock_llm_openai

    test_series = pd.Series(["This is a wonderful and fantastic experience!"])
    result_df = npt_llm_instance.analyze_sentiment_llm(
        text_series=test_series,
        llm_provider="openai",
        model_name="gpt-3.5-turbo",
        openai_api_key="test_key" # Passed to _get_llm_client mock
    )

    mock_get_llm.assert_called_once_with(llm_provider="openai", model_name="gpt-3.5-turbo", openai_api_key="test_key")
    mock_llm_openai.invoke.assert_called_once()

    # Validate DataFrame structure and content
    assert isinstance(result_df, pd.DataFrame)
    assert list(result_df.columns) == ["text", "sentiment", "raw_llm_output"]
    assert len(result_df) == 1
    pd.testing.assert_series_equal(result_df["text"], test_series, check_names=False)
    assert result_df.iloc[0]["sentiment"] == "positive"
    assert result_df.iloc[0]["raw_llm_output"] == "positive"


@patch('nlplot.nlplot.NLPlot._get_llm_client')
def test_analyze_sentiment_llm_ollama_negative(mock_get_llm, npt_llm_instance):
    if not MODULE_LANGCHAIN_AVAILABLE or not LANGCHAIN_TEST_IMPORTS_AVAILABLE:
        pytest.skip("Langchain or its dependencies not available.")

    mock_llm_ollama = MagicMock(spec=OllamaChat)
    mock_llm_ollama.invoke.return_value = AIMessage(content=" negative ") # Test with extra spaces
    mock_get_llm.return_value = mock_llm_ollama

    test_series = pd.Series(["This is a terrible, awful thing."])
    result_df = npt_llm_instance.analyze_sentiment_llm(
        text_series=test_series,
        llm_provider="ollama",
        model_name="llama2"
    )
    mock_get_llm.assert_called_once_with(llm_provider="ollama", model_name="llama2")
    mock_llm_ollama.invoke.assert_called_once()
    assert result_df.iloc[0]["sentiment"] == "negative" # Expect normalization
    assert result_df.iloc[0]["raw_llm_output"] == " negative "


@patch('nlplot.nlplot.NLPlot._get_llm_client')
def test_analyze_sentiment_llm_neutral_and_unknown_output(mock_get_llm, npt_llm_instance):
    if not MODULE_LANGCHAIN_AVAILABLE or not LANGCHAIN_TEST_IMPORTS_AVAILABLE:
        pytest.skip("Langchain not available.")

    mock_llm = MagicMock() # Generic mock for an LLM client
    # Simulate a sequence of LLM outputs
    mock_llm.invoke.side_effect = [
        AIMessage(content="This sentence is neutral."), # First call
        AIMessage(content="UNSURE ABOUT THIS ONE"),     # Second call
        AIMessage(content=" positive")                  # Third call (for a three-item series)
    ]
    mock_get_llm.return_value = mock_llm

    test_series = pd.Series([
        "The sky is blue.",
        "This is an ambiguous statement.",
        "The product works as expected." # Could be neutral or slightly positive by default LLM
    ])
    result_df = npt_llm_instance.analyze_sentiment_llm(test_series, "openai", "any_model", openai_api_key="key")

    assert len(result_df) == 3
    assert result_df.iloc[0]["sentiment"] == "neutral" # Assuming "neutral." is parsed to "neutral"
    assert result_df.iloc[1]["sentiment"] == "unknown" # Fallback for unrecognized output
    assert result_df.iloc[1]["raw_llm_output"] == "UNSURE ABOUT THIS ONE"
    assert result_df.iloc[2]["sentiment"] == "positive"


def test_analyze_sentiment_llm_empty_series_input(npt_llm_instance):
    if not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("Langchain not available.")

    empty_series = pd.Series([], dtype=str)
    result_df = npt_llm_instance.analyze_sentiment_llm(empty_series, "openai", "any_model")
    assert result_df.empty
    assert list(result_df.columns) == ["text", "sentiment", "raw_llm_output"]


@patch('nlplot.nlplot.NLPlot._get_llm_client')
@patch('nlplot.nlplot.PromptTemplate') # To verify custom prompt usage
def test_analyze_sentiment_llm_custom_prompt(MockPromptTemplate, mock_get_llm, npt_llm_instance):
    if not MODULE_LANGCHAIN_AVAILABLE or not LANGCHAIN_TEST_IMPORTS_AVAILABLE:
        pytest.skip("Langchain not available.")

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="positive")
    mock_get_llm.return_value = mock_llm

    # Mock the PromptTemplate.from_template to check if it's called with the custom string
    mock_template_instance = MagicMock()
    MockPromptTemplate.from_template = MagicMock(return_value=mock_template_instance)

    custom_prompt_str = "Analyze this text: {text}. Is it good or bad?"
    test_series = pd.Series(["A custom prompt test."])

    npt_llm_instance.analyze_sentiment_llm(
        test_series, "openai", "any_model",
        prompt_template=custom_prompt_str, openai_api_key="key"
    )

    MockPromptTemplate.from_template.assert_called_once_with(custom_prompt_str)
    # Further assert that the chain or llm.invoke was called with output from this template if possible


@patch('nlplot.nlplot.NLPlot._get_llm_client')
def test_analyze_sentiment_llm_api_error(mock_get_llm, npt_llm_instance):
    if not MODULE_LANGCHAIN_AVAILABLE or not LANGCHAIN_TEST_IMPORTS_AVAILABLE:
        pytest.skip("Langchain not available.")

    mock_llm = MagicMock()
    # Simulate an API error during the invoke call
    mock_llm.invoke.side_effect = Exception("Simulated API Error")
    mock_get_llm.return_value = mock_llm

    test_series = pd.Series(["This should cause an API error."])
    # Expect the method to handle the error gracefully, e.g., by returning "error" or "unknown"
    # and logging the raw_llm_output as the error message or a specific marker.
    with patch('builtins.print') as mock_print: # Capture print output for warnings/errors
        result_df = npt_llm_instance.analyze_sentiment_llm(test_series, "openai", "any_model", openai_api_key="key")

    assert result_df.iloc[0]["sentiment"] == "error" # Or "unknown" depending on chosen strategy
    assert "Simulated API Error" in result_df.iloc[0]["raw_llm_output"]
    # Check if a warning/error was printed
    # mock_print.assert_any_call(containing="Error analyzing sentiment for text") # Example
    printed_output = "".join(call.args[0] for call in mock_print.call_args_list if call.args)
    assert "Error analyzing sentiment for text" in printed_output
    assert "Simulated API Error" in printed_output
```

この新しいテストファイル `tests/test_nlplot_llm_sentiment.py` を作成しました。
最初のテスト `test_analyze_sentiment_llm_initial_method_missing` が、`analyze_sentiment_llm` メソッドが存在しないことによる `AttributeError` を期待する Red フェーズのテストとなります。

他のテストケースは、メソッドのスタブが作られた後、具体的な機能を実装する際のRed/Green/Refactorサイクルで使用します。

次のステップは、`nlplot.py` に `analyze_sentiment_llm` メソッドの空のスタブを追加することです。
