import pytest
import os
from unittest.mock import patch, MagicMock
import pandas as pd

try:
    from langchain_openai import ChatOpenAI
    from langchain_community.chat_models.ollama import OllamaChat
    from langchain_core.outputs import AIMessage
    from langchain_core.prompts import PromptTemplate

    LANGCHAIN_TEST_IMPORTS_AVAILABLE = True
except ImportError:
    LANGCHAIN_TEST_IMPORTS_AVAILABLE = False

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
    from nlplot_llm.core import LANGCHAIN_AVAILABLE as MODULE_LANGCHAIN_AVAILABLE
except ImportError:
    MODULE_LANGCHAIN_AVAILABLE = False

try:
    import litellm

    LITELLM_AVAILABLE_FOR_TEST = True
except ImportError:
    LITELLM_AVAILABLE_FOR_TEST = False

    class litellm_dummy_exc:
        class exceptions:
            class APIConnectionError(Exception):
                pass

            class AuthenticationError(Exception):
                pass

            class RateLimitError(Exception):
                pass

            class BadRequestError(Exception):
                pass

    litellm = litellm_dummy_exc()


@pytest.fixture
def npt_llm_instance(tmp_path):
    df = pd.DataFrame({"text": ["initial setup text to satisfy NLPlotLLM constructor"]})
    output_dir = tmp_path / "llm_sentiment_test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    return NLPlotLLM(df, target_col="text", output_file_path=str(output_dir))


def test_analyze_sentiment_llm_initial_method_exists(npt_llm_instance):
    assert hasattr(npt_llm_instance, "analyze_sentiment_llm")


@patch("nlplot_llm.llm.sentiment.litellm.completion")
def test_analyze_sentiment_llm_openai_positive(
    mock_litellm_completion, npt_llm_instance
):
    if not LITELLM_AVAILABLE_FOR_TEST:
        pytest.skip("LiteLLM not available, skipping LLM sentiment test.")

    if not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("LiteLLM not available in nlplot_llm.core, skipping.")

    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "positive"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_litellm_completion.return_value = mock_response

    test_series = pd.Series(["This is a wonderful and fantastic experience!"])
    test_model_str = "openai/gpt-3.5-turbo"

    result_df = npt_llm_instance.analyze_sentiment_llm(
        text_series=test_series, model=test_model_str, api_key="test_key"
    )

    expected_messages = [
        {
            "role": "user",
            "content": (
                "Analyze the sentiment of the following text and classify it as "
                "'positive', 'negative', or 'neutral'. Return only the single "
                "word classification for the sentiment. Text: This is a "
                "wonderful and fantastic experience!"
            ),
        }
    ]
    mock_litellm_completion.assert_called_once_with(
        model=test_model_str,
        messages=expected_messages,
        api_key="test_key",
        temperature=0.0,
    )

    assert isinstance(result_df, pd.DataFrame)
    assert list(result_df.columns) == ["text", "sentiment", "raw_llm_output"]
    assert len(result_df) == 1
    pd.testing.assert_series_equal(result_df["text"], test_series, check_names=False)
    assert result_df.iloc[0]["sentiment"] == "positive"
    assert result_df.iloc[0]["raw_llm_output"] == "positive"


@patch("nlplot_llm.llm.sentiment.litellm.completion")
def test_analyze_sentiment_llm_ollama_negative(
    mock_litellm_completion, npt_llm_instance
):
    if not LITELLM_AVAILABLE_FOR_TEST or not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("LiteLLM not available, skipping.")

    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = " negative "
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_litellm_completion.return_value = mock_response

    test_text = "This is a terrible, awful thing."
    test_series = pd.Series([test_text])
    test_model_str = "ollama/llama2"
    test_api_base = "http://localhost:11434"

    result_df = npt_llm_instance.analyze_sentiment_llm(
        text_series=test_series, model=test_model_str, api_base=test_api_base
    )

    expected_messages = [
        {
            "role": "user",
            "content": (
                "Analyze the sentiment of the following text and classify it as "
                "'positive', 'negative', or 'neutral'. Return only the single "
                f"word classification for the sentiment. Text: {test_text}"
            ),
        }
    ]
    mock_litellm_completion.assert_called_once_with(
        model=test_model_str,
        messages=expected_messages,
        api_base=test_api_base,
        temperature=0.0,
    )
    assert result_df.iloc[0]["sentiment"] == "negative"
    assert result_df.iloc[0]["raw_llm_output"] == " negative "


@patch("nlplot_llm.llm.sentiment.litellm.completion")
def test_analyze_sentiment_llm_neutral_and_unknown_output(
    mock_litellm_completion, npt_llm_instance
):
    if not LITELLM_AVAILABLE_FOR_TEST or not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("LiteLLM not available.")

    mock_litellm_completion.side_effect = [
        MagicMock(
            choices=[MagicMock(message=MagicMock(content="This sentence is neutral."))]
        ),
        MagicMock(
            choices=[MagicMock(message=MagicMock(content="UNSURE ABOUT THIS ONE"))]
        ),
        MagicMock(choices=[MagicMock(message=MagicMock(content=" positive"))]),
    ]

    test_series_texts = [
        "The sky is blue.",
        "This is an ambiguous statement.",
        "The product works as expected.",
    ]
    test_series = pd.Series(test_series_texts)
    test_model_str = "some_model/some_variant"

    result_df = npt_llm_instance.analyze_sentiment_llm(
        test_series, model=test_model_str
    )

    assert mock_litellm_completion.call_count == len(test_series_texts)
    assert len(result_df) == 3
    assert result_df.iloc[0]["sentiment"] == "neutral"
    assert result_df.iloc[1]["sentiment"] == "unknown"
    assert result_df.iloc[1]["raw_llm_output"] == "UNSURE ABOUT THIS ONE"
    assert result_df.iloc[2]["sentiment"] == "positive"


def test_analyze_sentiment_llm_empty_series_input(npt_llm_instance):
    if not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("LiteLLM not available.")

    empty_series = pd.Series([], dtype=str)
    result_df = npt_llm_instance.analyze_sentiment_llm(empty_series, model="any/model")
    assert result_df.empty
    assert list(result_df.columns) == ["text", "sentiment", "raw_llm_output"]


@patch("nlplot_llm.llm.sentiment.litellm.completion")
def test_analyze_sentiment_llm_custom_prompt(mock_litellm_completion, npt_llm_instance):
    if not LITELLM_AVAILABLE_FOR_TEST or not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("LiteLLM not available.")

    mock_response = MagicMock(
        choices=[MagicMock(message=MagicMock(content="positive"))]
    )
    mock_litellm_completion.return_value = mock_response

    custom_prompt_str = "Analyze this text: {text}. Is it good or bad?"
    test_text = "A custom prompt test."
    test_series = pd.Series([test_text])
    test_model_str = "custom/model"

    npt_llm_instance.analyze_sentiment_llm(
        test_series,
        model=test_model_str,
        prompt_template_str=custom_prompt_str,
        api_key="key",
    )

    expected_messages = [
        {"role": "user", "content": custom_prompt_str.format(text=test_text)}
    ]
    mock_litellm_completion.assert_called_once_with(
        model=test_model_str, messages=expected_messages, api_key="key", temperature=0.0
    )


@patch("nlplot_llm.llm.sentiment.litellm.completion")
def test_analyze_sentiment_llm_api_error(mock_litellm_completion, npt_llm_instance):
    if not LITELLM_AVAILABLE_FOR_TEST or not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("LiteLLM not available.")

    mock_litellm_completion.side_effect = litellm.exceptions.APIConnectionError(
        "Simulated LiteLLM API Error"
    )

    test_series = pd.Series(["This should cause an API error."])
    with patch("builtins.print") as mock_print:
        result_df = npt_llm_instance.analyze_sentiment_llm(
            test_series, model="error/model", api_key="key"
        )

    assert result_df.iloc[0]["sentiment"] == "error"
    assert "Simulated LiteLLM API Error" in result_df.iloc[0]["raw_llm_output"]
    assert "APIConnectionError" in result_df.iloc[0]["raw_llm_output"]
    printed_output = "".join(
        call.args[0] for call in mock_print.call_args_list if call.args
    )
    assert "Error analyzing sentiment for text" in printed_output
    assert "Simulated API Error" in printed_output


@pytest.mark.parametrize(
    "exception_type, error_message_detail",
    [
        (litellm.exceptions.AuthenticationError, "Simulated LiteLLM Auth Error"),
        (litellm.exceptions.RateLimitError, "Simulated LiteLLM Rate Limit Error"),
        (litellm.exceptions.BadRequestError, "Simulated LiteLLM Bad Request Error"),
        (Exception, "Generic Exception"),
    ],
)
@patch("nlplot_llm.llm.sentiment.litellm.completion")
def test_analyze_sentiment_llm_various_api_errors(
    mock_litellm_completion, exception_type, error_message_detail, npt_llm_instance
):
    if not LITELLM_AVAILABLE_FOR_TEST or not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("LiteLLM not available for API error tests.")

    mock_litellm_completion.side_effect = exception_type(error_message_detail)

    test_series = pd.Series(["This should cause an API error."])
    with patch("builtins.print") as mock_print:
        result_df = npt_llm_instance.analyze_sentiment_llm(
            test_series, model="error/model", api_key="key"
        )

    assert len(result_df) == 1
    assert result_df.iloc[0]["sentiment"] == "error"
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
        or isinstance(
            exception_type(error_message_detail), litellm.exceptions.BadRequestError
        )
    )

    if exception_name_in_output:
        assert exception_type.__name__ in result_df.iloc[0]["raw_llm_output"]

    printed_output_str = "".join(
        call.args[0] for call in mock_print.call_args_list if call.args
    )
    assert "Error analyzing sentiment for text" in printed_output_str
    assert error_message_detail in printed_output_str


def test_analyze_sentiment_llm_invalid_prompt_template(npt_llm_instance):
    if not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("LiteLLM not available.")

    test_series = pd.Series(["Some text"])
    invalid_prompt = "This prompt is missing the placeholder."
    with patch("builtins.print") as mock_print:
        result_df = npt_llm_instance.analyze_sentiment_llm(
            test_series, model="any/model", prompt_template_str=invalid_prompt
        )

    assert len(result_df) == 1
    assert result_df.iloc[0]["sentiment"] == "error"
    assert (
        "Prompt template error: missing {text}" in result_df.iloc[0]["raw_llm_output"]
    )
    mock_print.assert_any_call(
        "Error: Prompt template must include '{text}' placeholder."
    )


@patch("nlplot_llm.llm.sentiment.litellm.completion")
def test_analyze_sentiment_llm_empty_text_in_series(
    mock_litellm_completion, npt_llm_instance
):
    if not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("LiteLLM not available.")

    test_series = pd.Series(["Good text", "", "   ", None, "Bad text"])

    mock_litellm_completion.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="positive"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="negative"))]),
    ]

    result_df = npt_llm_instance.analyze_sentiment_llm(test_series, model="test/model")

    assert len(result_df) == 5
    assert result_df.iloc[0]["sentiment"] == "positive"
    assert result_df.iloc[0]["raw_llm_output"] == "positive"

    assert result_df.iloc[1]["sentiment"] == "neutral"
    assert "Input text was empty or whitespace" in result_df.iloc[1]["raw_llm_output"]

    assert result_df.iloc[2]["sentiment"] == "neutral"
    assert "Input text was empty or whitespace" in result_df.iloc[2]["raw_llm_output"]

    assert result_df.iloc[3]["sentiment"] == "neutral"
    assert "Input text was empty or whitespace" in result_df.iloc[3]["raw_llm_output"]

    assert result_df.iloc[4]["sentiment"] == "negative"
    assert result_df.iloc[4]["raw_llm_output"] == "negative"

    assert mock_litellm_completion.call_count == 2
