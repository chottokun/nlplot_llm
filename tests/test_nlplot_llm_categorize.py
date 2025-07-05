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

from nlplot_llm import NLPlotLLM # Updated import
try:
    # Assuming LANGCHAIN_AVAILABLE is defined in nlplot_llm.core module
    from nlplot_llm.core import LANGCHAIN_AVAILABLE as MODULE_LANGCHAIN_AVAILABLE_CAT
except ImportError: # Fallback if the import path or flag name is different
    MODULE_LANGCHAIN_AVAILABLE_CAT = False # This will be LITELLM_AVAILABLE from nlplot_llm.core

# pytest.skip("Skipping old Langchain categorize tests; will be reworked for LiteLLM.", allow_module_level=True) # Removing skip

# Attempt to import litellm for mocking its exceptions (similar to sentiment tests)
try:
    import litellm
    LITELLM_AVAILABLE_FOR_TEST_CAT = True # Use a distinct name
except ImportError:
    LITELLM_AVAILABLE_FOR_TEST_CAT = False
    class litellm_dummy_exc_cat: # type: ignore
        class exceptions: # type: ignore
            class APIConnectionError(Exception): pass
            class AuthenticationError(Exception): pass
            class RateLimitError(Exception): pass
    litellm = litellm_dummy_exc_cat() # type: ignore


@pytest.fixture
def npt_llm_instance_cat(tmp_path): # Renamed fixture for clarity
    """Provides a basic NLPlotLLM instance for LLM categorization tests."""
    df = pd.DataFrame({'text': ["initial setup text for categorization"]})
    output_dir = tmp_path / "llm_categorize_test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    return NLPlotLLM(df, target_col='text', output_file_path=str(output_dir)) # Updated class name

# --- TDD for LLM Text Categorization (Cycle 3) ---

def test_categorize_text_llm_initial_method_missing(npt_llm_instance_cat):
    """(Red Phase) Ensure categorize_text_llm method is initially missing."""
    # This test might be obsolete if the method already exists from previous work.
    # If it's already there, this will fail. We are now focusing on adapting to LiteLLM.
    # For now, keep it to ensure no regressions if method was accidentally removed.
    # Match should be NLPlotLLM now.
    with pytest.raises(AttributeError, match="'NLPlotLLM' object has no attribute 'categorize_text_llm'"):
        npt_llm_instance_cat.categorize_text_llm(
            text_series=pd.Series(["a test sentence for categorization"]),
            categories=["news", "sports", "weather"], # Example categories
            llm_provider="openai",
            model_name="gpt-3.5-turbo"
            # Add other required args if any, once signature is defined
        )

# The tests below are for the Green/Refactor phase, once categorize_text_llm is implemented.
# These will need to be adapted for LiteLLM.
# For now, we'll update the patch target for _get_llm_client if it were still used,
# but it will be removed. The new target will be 'litellm.completion'.

@patch('litellm.completion') # Changed patch target
def test_categorize_text_llm_single_label_openai(mock_litellm_completion, npt_llm_instance_cat):
    if not LITELLM_AVAILABLE_FOR_TEST_CAT or not MODULE_LANGCHAIN_AVAILABLE_CAT:
        pytest.skip("LiteLLM not available, skipping LLM categorization test.")

    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "sports" # Expected LLM output
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_litellm_completion.return_value = mock_response

    test_text = "The home team won the game with a last-minute goal."
    test_series = pd.Series([test_text])
    categories = ["news", "sports", "weather", "finance"]
    test_model_str = "openai/gpt-3.5-turbo"

    result_df = npt_llm_instance_cat.categorize_text_llm(
        text_series=test_series,
        categories=categories,
        model=test_model_str, # Updated argument
        api_key="test_key_cat", # Passed via **litellm_kwargs
        multi_label=False
    )

    category_list_str_for_prompt = ", ".join(f"'{c}'" for c in categories)
    expected_prompt_content = (
        f"Analyze the following text and classify it into exactly one of these categories: {category_list_str_for_prompt}. "
        f"Return only the single matching category name. If no categories match, return 'unknown'. Text: {test_text}"
    )
    expected_messages = [{"role": "user", "content": expected_prompt_content}]

    mock_litellm_completion.assert_called_once_with(
        model=test_model_str,
        messages=expected_messages,
        api_key="test_key_cat",
        temperature=0.0 # Default
    )

    assert isinstance(result_df, pd.DataFrame)
    assert list(result_df.columns) == ["text", "category", "raw_llm_output"]
    assert len(result_df) == 1
    assert result_df.iloc[0]["category"] == "sports"
    assert result_df.iloc[0]["raw_llm_output"] == "sports"


@patch('litellm.completion')
def test_categorize_text_llm_multi_label_ollama(mock_litellm_completion, npt_llm_instance_cat):
    if not LITELLM_AVAILABLE_FOR_TEST_CAT or not MODULE_LANGCHAIN_AVAILABLE_CAT:
        pytest.skip("LiteLLM not available.")

    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "news, finance" # Expected LLM output
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_litellm_completion.return_value = mock_response

    test_text = "Company X announced record profits and new product line."
    test_series = pd.Series([test_text])
    categories = ["news", "sports", "weather", "finance", "technology"]
    test_model_str = "ollama/llama2"
    test_api_base = "http://localhost:11434" # Example for Ollama

    result_df = npt_llm_instance_cat.categorize_text_llm(
        text_series=test_series,
        categories=categories,
        model=test_model_str,
        api_base=test_api_base,
        multi_label=True
    )

    category_list_str_for_prompt = ", ".join(f"'{c}'" for c in categories)
    expected_prompt_content = (
        f"Analyze the following text and classify it into one or more of these categories: {category_list_str_for_prompt}. "
        f"Return a comma-separated list of the matching category names. If no categories match, return 'none'. Text: {test_text}"
    )
    expected_messages = [{"role": "user", "content": expected_prompt_content}]

    mock_litellm_completion.assert_called_once_with(
        model=test_model_str,
        messages=expected_messages,
        api_base=test_api_base,
        temperature=0.0
    )

    assert list(result_df.columns) == ["text", "categories", "raw_llm_output"]
    assert isinstance(result_df.iloc[0]["categories"], list)
    assert "news" in result_df.iloc[0]["categories"]
    assert "finance" in result_df.iloc[0]["categories"]
    assert len(result_df.iloc[0]["categories"]) == 2
    assert result_df.iloc[0]["raw_llm_output"] == "news, finance"

@patch('litellm.completion')
def test_categorize_text_llm_no_matching_category(mock_litellm_completion, npt_llm_instance_cat):
    if not LITELLM_AVAILABLE_FOR_TEST_CAT or not MODULE_LANGCHAIN_AVAILABLE_CAT:
        pytest.skip("LiteLLM not available.")

    mock_response = MagicMock(choices=[MagicMock(message=MagicMock(content="This text is about general topics."))])
    mock_litellm_completion.return_value = mock_response

    test_series = pd.Series(["A very generic statement."])
    categories = ["politics", "art", "science"]
    test_model_str = "some_model/variant"

    result_df = npt_llm_instance_cat.categorize_text_llm(test_series, categories, model=test_model_str, api_key="k")
    assert result_df.iloc[0]["category"] == "unknown"


def test_categorize_text_llm_empty_categories_list(npt_llm_instance_cat):
    if not MODULE_LANGCHAIN_AVAILABLE_CAT:
        pytest.skip("Core LLM utilities not available.")
    test_series = pd.Series(["Some text."])
    with pytest.raises(ValueError, match="Categories list must be a non-empty list of non-empty strings."):
        npt_llm_instance_cat.categorize_text_llm(test_series, [], model="any/model", api_key="k")

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
