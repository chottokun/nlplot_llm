import pytest
import os
from unittest.mock import patch, MagicMock
import pandas as pd

try:
    from langchain_openai import ChatOpenAI
    from langchain_community.chat_models.ollama import OllamaChat
    from langchain_core.outputs import AIMessage
    from langchain_core.prompts import PromptTemplate

    LANGCHAIN_TEST_IMPORTS_AVAILABLE_CAT = True
except ImportError:
    LANGCHAIN_TEST_IMPORTS_AVAILABLE_CAT = False

    class AIMessage:
        def __init__(self, content):
            self.content = content

    class ChatOpenAI:
        pass

    class OllamaChat:
        pass

    class PromptTemplate:
        pass


from nlplot_llm import NLPlotLLM

try:
    from nlplot_llm.core import LANGCHAIN_AVAILABLE as MODULE_LANGCHAIN_AVAILABLE_CAT
except ImportError:
    MODULE_LANGCHAIN_AVAILABLE_CAT = False

try:
    import litellm

    LITELLM_AVAILABLE_FOR_TEST_CAT = True
except ImportError:
    LITELLM_AVAILABLE_FOR_TEST_CAT = False

    class litellm_dummy_exc_cat:
        class exceptions:
            class APIConnectionError(Exception):
                pass

            class AuthenticationError(Exception):
                pass

            class RateLimitError(Exception):
                pass

    litellm = litellm_dummy_exc_cat()


@pytest.fixture
def npt_llm_instance_cat(tmp_path):
    df = pd.DataFrame({"text": ["initial setup text for categorization"]})
    output_dir = tmp_path / "llm_categorize_test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    return NLPlotLLM(df, target_col="text", output_file_path=str(output_dir))


def test_categorize_text_llm_initial_method_exists(npt_llm_instance_cat):
    assert hasattr(npt_llm_instance_cat, "categorize_text_llm")


@patch("litellm.completion")
def test_categorize_text_llm_single_label_openai(
    mock_litellm_completion, npt_llm_instance_cat
):
    if not LITELLM_AVAILABLE_FOR_TEST_CAT or not MODULE_LANGCHAIN_AVAILABLE_CAT:
        pytest.skip("LiteLLM not available, skipping LLM categorization test.")

    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "sports"
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
        model=test_model_str,
        api_key="test_key_cat",
        multi_label=False,
    )

    category_list_str_for_prompt = ", ".join(f"'{c}'" for c in categories)
    expected_prompt_content = (
        "Analyze the following text and classify it into exactly one of "
        f"these categories: {category_list_str_for_prompt}. Return only the "
        "single matching category name. If no categories match, return "
        f"'unknown'. Text: {test_text}"
    )
    expected_messages = [{"role": "user", "content": expected_prompt_content}]

    mock_litellm_completion.assert_called_once_with(
        model=test_model_str,
        messages=expected_messages,
        api_key="test_key_cat",
        temperature=0.0,
    )

    assert isinstance(result_df, pd.DataFrame)
    assert list(result_df.columns) == ["text", "category", "raw_llm_output"]
    assert len(result_df) == 1
    assert result_df.iloc[0]["category"] == "sports"
    assert result_df.iloc[0]["raw_llm_output"] == "sports"


@patch("litellm.completion")
def test_categorize_text_llm_multi_label_ollama(
    mock_litellm_completion, npt_llm_instance_cat
):
    if not LITELLM_AVAILABLE_FOR_TEST_CAT or not MODULE_LANGCHAIN_AVAILABLE_CAT:
        pytest.skip("LiteLLM not available.")

    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "news, finance"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_litellm_completion.return_value = mock_response

    test_text = "Company X announced record profits and new product line."
    test_series = pd.Series([test_text])
    categories = ["news", "sports", "weather", "finance", "technology"]
    test_model_str = "ollama/llama2"
    test_api_base = "http://localhost:11434"

    result_df = npt_llm_instance_cat.categorize_text_llm(
        text_series=test_series,
        categories=categories,
        model=test_model_str,
        api_base=test_api_base,
        multi_label=True,
    )

    category_list_str_for_prompt = ", ".join(f"'{c}'" for c in categories)
    expected_prompt_content = (
        "Analyze the following text and classify it into one or more of "
        f"these categories: {category_list_str_for_prompt}. Return a "
        "comma-separated list of the matching category names. If no "
        f"categories match, return 'none'. Text: {test_text}"
    )
    expected_messages = [{"role": "user", "content": expected_prompt_content}]

    mock_litellm_completion.assert_called_once_with(
        model=test_model_str,
        messages=expected_messages,
        api_base=test_api_base,
        temperature=0.0,
    )

    assert list(result_df.columns) == ["text", "categories", "raw_llm_output"]
    assert isinstance(result_df.iloc[0]["categories"], list)
    assert "news" in result_df.iloc[0]["categories"]
    assert "finance" in result_df.iloc[0]["categories"]
    assert len(result_df.iloc[0]["categories"]) == 2
    assert result_df.iloc[0]["raw_llm_output"] == "news, finance"


@patch("litellm.completion")
def test_categorize_text_llm_no_matching_category(
    mock_litellm_completion, npt_llm_instance_cat
):
    if not LITELLM_AVAILABLE_FOR_TEST_CAT or not MODULE_LANGCHAIN_AVAILABLE_CAT:
        pytest.skip("LiteLLM not available.")

    mock_response = MagicMock(
        choices=[
            MagicMock(message=MagicMock(content="This text is about general topics."))
        ]
    )
    mock_litellm_completion.return_value = mock_response

    test_series = pd.Series(["A very generic statement."])
    categories = ["politics", "art", "science"]
    test_model_str = "some_model/variant"

    result_df = npt_llm_instance_cat.categorize_text_llm(
        test_series, categories, model=test_model_str, api_key="k"
    )
    assert result_df.iloc[0]["category"] == "unknown"
    assert result_df.iloc[0]["raw_llm_output"] == "This text is about general topics."


def test_categorize_text_llm_empty_categories_list(npt_llm_instance_cat):
    if not MODULE_LANGCHAIN_AVAILABLE_CAT:
        pytest.skip("Core LLM utilities not available.")
    test_series = pd.Series(["Some text."])
    with pytest.raises(
        ValueError,
        match="Categories list must be a non-empty list of non-empty strings.",
    ):
        npt_llm_instance_cat.categorize_text_llm(test_series, [], model="any/model")


@pytest.mark.parametrize(
    "exception_type, error_message_detail",
    [
        (
            litellm.exceptions.AuthenticationError,
            "Simulated LiteLLM Auth Error for categorize",
        ),
        (
            litellm.exceptions.RateLimitError,
            "Simulated LiteLLM Rate Limit Error for categorize",
        ),
        (
            litellm.exceptions.APIConnectionError,
            "Simulated LiteLLM API Connection Error for categorize",
        ),
        (Exception, "Generic Exception for categorize"),
    ],
)
@patch("litellm.completion")
def test_categorize_text_llm_various_api_errors(
    mock_litellm_completion, exception_type, error_message_detail, npt_llm_instance_cat
):
    if not LITELLM_AVAILABLE_FOR_TEST_CAT or not MODULE_LANGCHAIN_AVAILABLE_CAT:
        pytest.skip("LiteLLM not available for API error tests.")

    mock_litellm_completion.side_effect = exception_type(error_message_detail)
    test_series = pd.Series(["Text to cause error."])
    categories = ["cat1", "cat2"]
    category_col_name = "category"

    with patch("builtins.print") as mock_print:
        result_df = npt_llm_instance_cat.categorize_text_llm(
            test_series, categories, model="error/model", multi_label=False
        )

    assert len(result_df) == 1
    assert result_df.iloc[0][category_col_name] == "error"
    assert error_message_detail in result_df.iloc[0]["raw_llm_output"]

    exception_name_in_output = (
        isinstance(
            exception_type(error_message_detail), litellm.exceptions.APIConnectionError
        )
        or isinstance(
            exception_type(error_message_detail), litellm.exceptions.AuthenticationError
        )
        or isinstance(
            exception_type(error_message_detail), litellm.exceptions.RateLimitError
        )
    )
    if exception_name_in_output:
        assert exception_type.__name__ in result_df.iloc[0]["raw_llm_output"]

    printed_output_str = "".join(
        call.args[0] for call in mock_print.call_args_list if call.args
    )
    assert "Error categorizing text" in printed_output_str
    assert error_message_detail in printed_output_str


def test_categorize_text_llm_invalid_prompt_template_no_text(npt_llm_instance_cat):
    if not MODULE_LANGCHAIN_AVAILABLE_CAT:
        pytest.skip("LiteLLM not available.")

    test_series = pd.Series(["Some text"])
    categories = ["cat1"]
    invalid_prompt = "This prompt is missing {text} placeholder, but has {categories}."
    with patch("builtins.print") as mock_print:
        result_df = npt_llm_instance_cat.categorize_text_llm(
            test_series,
            categories,
            model="any/model",
            prompt_template_str=invalid_prompt,
        )

    assert len(result_df) == 1
    assert result_df.iloc[0]["category"] == "error"
    assert (
        "Prompt template error: missing {text}" in result_df.iloc[0]["raw_llm_output"]
    )
    mock_print.assert_any_call(
        "Error: Prompt template must include '{text}' placeholder."
    )


@patch("litellm.completion")
def test_categorize_text_llm_empty_and_none_strings_in_series(
    mock_litellm_completion, npt_llm_instance_cat
):
    if not MODULE_LANGCHAIN_AVAILABLE_CAT:
        pytest.skip("LiteLLM not available.")

    test_series = pd.Series(["Categorize this", "", "   ", None, "And this too"])
    categories = ["A", "B"]

    mock_litellm_completion.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="A"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="B"))]),
    ]

    result_df = npt_llm_instance_cat.categorize_text_llm(
        test_series, categories, model="test/model", multi_label=False
    )

    assert len(result_df) == 5
    assert result_df.iloc[0]["category"] == "A"
    assert result_df.iloc[1]["category"] == "unknown"
    assert "Input text was empty or whitespace" in result_df.iloc[1]["raw_llm_output"]
    assert result_df.iloc[2]["category"] == "unknown"
    assert "Input text was empty or whitespace" in result_df.iloc[2]["raw_llm_output"]
    assert result_df.iloc[3]["category"] == "unknown"
    assert "Input text was empty or whitespace" in result_df.iloc[3]["raw_llm_output"]
    assert result_df.iloc[4]["category"] == "B"

    assert mock_litellm_completion.call_count == 2
