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

def test_categorize_text_llm_initial_method_exists(npt_llm_instance_cat):
    """Ensure categorize_text_llm method exists."""
    assert hasattr(npt_llm_instance_cat, 'categorize_text_llm')

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
    # Check that the raw output from LLM is preserved
    assert result_df.iloc[0]["raw_llm_output"] == "This text is about general topics."


def test_categorize_text_llm_empty_categories_list(npt_llm_instance_cat):
    if not MODULE_LANGCHAIN_AVAILABLE_CAT: # This is LITELLM_AVAILABLE in core
        pytest.skip("Core LLM utilities not available.")
    test_series = pd.Series(["Some text."])
    with pytest.raises(ValueError, match="Categories list must be a non-empty list of non-empty strings."):
        npt_llm_instance_cat.categorize_text_llm(test_series, [], model="any/model")


@pytest.mark.parametrize(
    "exception_type, error_message_detail",
    [
        (litellm.exceptions.AuthenticationError, "Simulated LiteLLM Auth Error for categorize"),
        (litellm.exceptions.RateLimitError, "Simulated LiteLLM Rate Limit Error for categorize"),
        (litellm.exceptions.APIConnectionError, "Simulated LiteLLM API Connection Error for categorize"),
        (Exception, "Generic Exception for categorize")
    ]
)
@patch('litellm.completion')
def test_categorize_text_llm_various_api_errors(mock_litellm_completion, exception_type, error_message_detail, npt_llm_instance_cat):
    if not LITELLM_AVAILABLE_FOR_TEST_CAT or not MODULE_LANGCHAIN_AVAILABLE_CAT:
        pytest.skip("LiteLLM not available for API error tests.")

    mock_litellm_completion.side_effect = exception_type(error_message_detail)
    test_series = pd.Series(["Text to cause error."])
    categories = ["cat1", "cat2"]
    category_col_name = "category" # for single_label

    with patch('builtins.print') as mock_print:
        result_df = npt_llm_instance_cat.categorize_text_llm(
            test_series, categories, model="error/model", multi_label=False
        )

    assert len(result_df) == 1
    assert result_df.iloc[0][category_col_name] == "error"
    assert error_message_detail in result_df.iloc[0]["raw_llm_output"]

    exception_name_in_output = isinstance(exception_type(error_message_detail), litellm.exceptions.APIConnectionError) or \
                               isinstance(exception_type(error_message_detail), litellm.exceptions.AuthenticationError) or \
                               isinstance(exception_type(error_message_detail), litellm.exceptions.RateLimitError)
    if exception_name_in_output:
        assert exception_type.__name__ in result_df.iloc[0]["raw_llm_output"]

    printed_output_str = "".join(call.args[0] for call in mock_print.call_args_list if call.args)
    assert "Error categorizing text" in printed_output_str
    assert error_message_detail in printed_output_str


def test_categorize_text_llm_invalid_prompt_template_no_text(npt_llm_instance_cat):
    if not MODULE_LANGCHAIN_AVAILABLE_CAT:
        pytest.skip("LiteLLM not available.")

    test_series = pd.Series(["Some text"])
    categories = ["cat1"]
    invalid_prompt = "This prompt is missing {text} placeholder, but has {categories}."
    with patch('builtins.print') as mock_print:
        result_df = npt_llm_instance_cat.categorize_text_llm(
            test_series, categories, model="any/model", prompt_template_str=invalid_prompt
        )

    assert len(result_df) == 1
    assert result_df.iloc[0]["category"] == "error" # Assuming single label default
    assert "Prompt template error: missing {text}" in result_df.iloc[0]["raw_llm_output"]
    mock_print.assert_any_call("Error: Prompt template must include '{text}' placeholder.")

# Note: The categorize_text_llm method has internal logic to create a default prompt if none is given.
# That default prompt construction depends on whether {categories} is in the user's prompt_template_str.
# A test for invalid prompt *with* {text} but missing {categories} when the method expects to inject it might be useful,
# but the current method logic for default prompts might make this complex to test without over-mocking.
# The core library should handle its default prompt generation correctly.
# The main check is if the user provides a bad prompt (e.g. missing {text}).

@patch('litellm.completion')
def test_categorize_text_llm_empty_and_none_strings_in_series(mock_litellm_completion, npt_llm_instance_cat):
    if not MODULE_LANGCHAIN_AVAILABLE_CAT:
        pytest.skip("LiteLLM not available.")

    test_series = pd.Series(["Categorize this", "", "   ", None, "And this too"])
    categories = ["A", "B"]

    mock_litellm_completion.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="A"))]), # For "Categorize this"
        MagicMock(choices=[MagicMock(message=MagicMock(content="B"))]), # For "And this too"
    ]

    result_df = npt_llm_instance_cat.categorize_text_llm(test_series, categories, model="test/model", multi_label=False)

    assert len(result_df) == 5
    assert result_df.iloc[0]["category"] == "A"
    assert result_df.iloc[1]["category"] == "unknown" # Empty string, should be 'unknown' or similar, not 'error'
    assert "Input text was empty or whitespace" in result_df.iloc[1]["raw_llm_output"]
    assert result_df.iloc[2]["category"] == "unknown" # Whitespace
    assert "Input text was empty or whitespace" in result_df.iloc[2]["raw_llm_output"]
    assert result_df.iloc[3]["category"] == "unknown" # None
    assert "Input text was empty or whitespace" in result_df.iloc[3]["raw_llm_output"]
    assert result_df.iloc[4]["category"] == "B"

    assert mock_litellm_completion.call_count == 2 # Only for non-empty texts


# Additional tests to consider:
# - test_categorize_text_llm_llm_returns_invalid_category (not in provided list)
# - test_categorize_text_llm_empty_text_series (already covered by sentiment tests structure)

