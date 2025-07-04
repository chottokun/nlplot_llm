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
    LANGCHAIN_TEST_IMPORTS_AVAILABLE_CAT = True # Use a distinct name to avoid clashes if run in same session
except ImportError:
    LANGCHAIN_TEST_IMPORTS_AVAILABLE_CAT = False
    class AIMessage: # Dummy for type hint
        def __init__(self, content): self.content = content
    class ChatOpenAI: pass
    class OllamaChat: pass
    class PromptTemplate: pass

from nlplot import NLPlot # Assuming NLPlot class is in nlplot/__init__.py or nlplot/nlplot.py
try:
    # Assuming LANGCHAIN_AVAILABLE is defined in nlplot.nlplot module
    from nlplot.nlplot import LANGCHAIN_AVAILABLE as MODULE_LANGCHAIN_AVAILABLE_CAT
except ImportError: # Fallback if the import path or flag name is different
    MODULE_LANGCHAIN_AVAILABLE_CAT = False


@pytest.fixture
def npt_llm_instance_cat(tmp_path): # Renamed fixture for clarity
    """Provides a basic NLPlot instance for LLM categorization tests."""
    df = pd.DataFrame({'text': ["initial setup text for categorization"]})
    output_dir = tmp_path / "llm_categorize_test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    return NLPlot(df, target_col='text', output_file_path=str(output_dir))

# --- TDD for LLM Text Categorization (Cycle 3) ---

def test_categorize_text_llm_initial_method_missing(npt_llm_instance_cat):
    """(Red Phase) Ensure categorize_text_llm method is initially missing."""
    with pytest.raises(AttributeError, match="'NLPlot' object has no attribute 'categorize_text_llm'"):
        npt_llm_instance_cat.categorize_text_llm(
            text_series=pd.Series(["a test sentence for categorization"]),
            categories=["news", "sports", "weather"], # Example categories
            llm_provider="openai",
            model_name="gpt-3.5-turbo"
            # Add other required args if any, once signature is defined
        )

# The tests below are for the Green/Refactor phase, once categorize_text_llm is implemented.

@patch('nlplot.nlplot.NLPlot._get_llm_client')
def test_categorize_text_llm_single_label_openai(mock_get_llm, npt_llm_instance_cat):
    if not MODULE_LANGCHAIN_AVAILABLE_CAT or not LANGCHAIN_TEST_IMPORTS_AVAILABLE_CAT:
        pytest.skip("Langchain or its dependencies not available, skipping LLM categorization test.")

    mock_llm_openai = MagicMock(spec=ChatOpenAI if LANGCHAIN_TEST_IMPORTS_AVAILABLE_CAT else object)
    mock_llm_openai.invoke.return_value = AIMessage(content="sports")
    mock_get_llm.return_value = mock_llm_openai

    test_series = pd.Series(["The home team won the game with a last-minute goal."])
    categories = ["news", "sports", "weather", "finance"]

    try:
        result_df = npt_llm_instance_cat.categorize_text_llm(
            text_series=test_series,
            categories=categories,
            llm_provider="openai",
            model_name="gpt-3.5-turbo",
            openai_api_key="test_key_cat", # Passed to _get_llm_client mock
            multi_label=False # Explicitly single label
        )

        mock_get_llm.assert_called_once_with(llm_provider="openai", model_name="gpt-3.5-turbo", openai_api_key="test_key_cat")
        mock_llm_openai.invoke.assert_called_once()

        # Check prompt content to ensure categories were included
        prompt_passed_to_llm_input = mock_llm_openai.invoke.call_args[0][0]
        # The input to invoke could be a string or a list of messages.
        # For now, convert to string for checking.
        prompt_str = str(prompt_passed_to_llm_input)
        for cat in categories:
            assert cat in prompt_str, f"Category '{cat}' not found in prompt: {prompt_str}"
        assert "single category" in prompt_str.lower() or "one category" in prompt_str.lower()


        assert isinstance(result_df, pd.DataFrame)
        assert list(result_df.columns) == ["text", "category", "raw_llm_output"]
        assert len(result_df) == 1
        assert result_df.iloc[0]["category"] == "sports"
        assert result_df.iloc[0]["raw_llm_output"] == "sports"
    except AttributeError:
         pytest.fail("categorize_text_llm method not found. This test should run after method stub is added.")
    except ImportError:
        pytest.skip("A Langchain component import failed within nlplot.nlplot.")


@patch('nlplot.nlplot.NLPlot._get_llm_client')
def test_categorize_text_llm_multi_label_ollama(mock_get_llm, npt_llm_instance_cat):
    if not MODULE_LANGCHAIN_AVAILABLE_CAT or not LANGCHAIN_TEST_IMPORTS_AVAILABLE_CAT:
        pytest.skip("Langchain or its dependencies not available.")

    mock_llm_ollama = MagicMock(spec=OllamaChat if LANGCHAIN_TEST_IMPORTS_AVAILABLE_CAT else object)
    # Simulate LLM returning comma-separated list for multi-label
    mock_llm_ollama.invoke.return_value = AIMessage(content="news, finance")
    mock_get_llm.return_value = mock_llm_ollama

    test_series = pd.Series(["Company X announced record profits and new product line."])
    categories = ["news", "sports", "weather", "finance", "technology"]
    try:
        result_df = npt_llm_instance_cat.categorize_text_llm(
            text_series, categories, "ollama", "llama2", multi_label=True
        )
        mock_get_llm.assert_called_once_with(llm_provider="ollama", model_name="llama2")
        mock_llm_ollama.invoke.assert_called_once()
        prompt_passed_to_llm_input = mock_llm_ollama.invoke.call_args[0][0]
        prompt_str = str(prompt_passed_to_llm_input)
        assert "one or more categories" in prompt_str.lower() or "multiple categories" in prompt_str.lower()


        assert list(result_df.columns) == ["text", "categories", "raw_llm_output"]
        assert isinstance(result_df.iloc[0]["categories"], list)
        assert "news" in result_df.iloc[0]["categories"]
        assert "finance" in result_df.iloc[0]["categories"]
        assert len(result_df.iloc[0]["categories"]) == 2
        assert result_df.iloc[0]["raw_llm_output"] == "news, finance"
    except AttributeError:
        pytest.fail("categorize_text_llm method not found.")

@patch('nlplot.nlplot.NLPlot._get_llm_client')
def test_categorize_text_llm_no_matching_category(mock_get_llm, npt_llm_instance_cat):
    if not MODULE_LANGCHAIN_AVAILABLE_CAT or not LANGCHAIN_TEST_IMPORTS_AVAILABLE_CAT:
        pytest.skip("Langchain not available.")
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="This text is about general topics.")
    mock_get_llm.return_value = mock_llm
    test_series = pd.Series(["A very generic statement."])
    categories = ["politics", "art", "science"]
    try:
        result_df = npt_llm_instance_cat.categorize_text_llm(test_series, categories, "openai", "m", openai_api_key="k")
        # For single label, if no match, it might return "unknown" or the first category, or None
        # Depending on implementation. Let's assume "unknown" for now.
        assert result_df.iloc[0]["category"] == "unknown"
    except AttributeError:
        pytest.fail("categorize_text_llm method not found.")

def test_categorize_text_llm_empty_categories_list(npt_llm_instance_cat):
    if not MODULE_LANGCHAIN_AVAILABLE_CAT:
        pytest.skip("Langchain not available.")
    test_series = pd.Series(["Some text."])
    with pytest.raises(ValueError, match="Categories list cannot be empty."):
        npt_llm_instance_cat.categorize_text_llm(test_series, [], "openai", "m", openai_api_key="k")

# Additional tests to consider:
# - test_categorize_text_llm_api_error (similar to sentiment one)
# - test_categorize_text_llm_custom_prompt
# - test_categorize_text_llm_llm_returns_invalid_category (not in provided list)
# - test_categorize_text_llm_empty_text_series
# - test_categorize_text_llm_text_series_with_none_or_empty_strings
```

この新しいテストファイル `tests/test_nlplot_llm_categorize.py` を作成しました。
最初のテスト `test_categorize_text_llm_initial_method_missing` が、`categorize_text_llm` メソッドが存在しないことによる `AttributeError` を期待する Red フェーズのテストとなります。

次のステップは、`nlplot.py` に `categorize_text_llm` メソッドの空のスタブを追加することです。
