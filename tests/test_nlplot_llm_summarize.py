import pytest
import os
from unittest.mock import patch, MagicMock
import pandas as pd

# Attempt to import langchain classes for type hinting and patching targets.
try:
    from langchain_openai import ChatOpenAI
    from langchain_community.chat_models.ollama import OllamaChat
    from langchain_core.outputs import AIMessage
    LANGCHAIN_TEST_IMPORTS_AVAILABLE_SUMMARIZE = True
except ImportError:
    LANGCHAIN_TEST_IMPORTS_AVAILABLE_SUMMARIZE = False
    class AIMessage:
        def __init__(self, content): self.content = content
    class ChatOpenAI: pass
    class OllamaChat: pass

from nlplot_llm import NLPlotLLM # Updated import
try:
    from nlplot_llm.core import LANGCHAIN_AVAILABLE as MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE # Updated path
except ImportError:
    MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE = False # This will be LITELLM_AVAILABLE from nlplot_llm.core

# pytest.skip("Skipping old Langchain summarization tests; will be reworked for LiteLLM.", allow_module_level=True) # Removing skip

# Attempt to import litellm for mocking its exceptions
try:
    import litellm
    LITELLM_AVAILABLE_FOR_TEST_SUMMARIZE = True # Use a distinct name
except ImportError:
    LITELLM_AVAILABLE_FOR_TEST_SUMMARIZE = False
    class litellm_dummy_exc_sum: # type: ignore
        class exceptions: # type: ignore
            class APIConnectionError(Exception): pass
            class AuthenticationError(Exception): pass
            class RateLimitError(Exception): pass
    litellm = litellm_dummy_exc_sum() # type: ignore


@pytest.fixture
def npt_llm_summarize_instance(tmp_path):
    """Provides a basic NLPlotLLM instance for LLM summarization tests.""" # Updated class name
    df = pd.DataFrame({'text': ["initial setup text for summarization"]})
    output_dir = tmp_path / "llm_summarize_test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    return NLPlotLLM(df, target_col='text', output_file_path=str(output_dir)) # Updated class name

# --- TDD for LLM Text Summarization ---

def test_summarize_text_llm_initial_method_exists(npt_llm_summarize_instance):
    """Ensure summarize_text_llm method exists."""
    assert hasattr(npt_llm_summarize_instance, 'summarize_text_llm')

# Further tests for Green/Refactor phase will be added below.

@patch('litellm.completion')
def test_summarize_text_llm_success_no_chunking(mock_litellm_completion, npt_llm_summarize_instance):
    if not LITELLM_AVAILABLE_FOR_TEST_SUMMARIZE or not MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE:
        pytest.skip("LiteLLM not available, skipping LLM summarization test.")

    mock_response = MagicMock()
    mock_message = MagicMock()
    expected_summary = "This is a concise summary."
    mock_message.content = expected_summary
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    mock_litellm_completion.return_value = mock_response

    test_text = "This is a relatively long piece of text that definitely needs to be summarized into a shorter version to save time for the reader."
    text_series = pd.Series([test_text])
    test_model_str = "openai/gpt-3.5-turbo"

    result_df = npt_llm_summarize_instance.summarize_text_llm(
        text_series=text_series,
        model=test_model_str,
        use_chunking=False,
        api_key="test_key_summarize" # Passed via **litellm_kwargs
    )

    expected_prompt_content = f"Please summarize the following text concisely: {test_text}" # Default prompt from method
    expected_messages = [{"role": "user", "content": expected_prompt_content}]

    mock_litellm_completion.assert_called_once_with(
        model=test_model_str,
        messages=expected_messages,
        api_key="test_key_summarize",
        temperature=0.0
    )

    assert isinstance(result_df, pd.DataFrame)
    assert list(result_df.columns) == ["original_text", "summary", "raw_llm_output"]
    assert len(result_df) == 1
    assert result_df.iloc[0]["original_text"] == test_text
    assert result_df.iloc[0]["summary"] == expected_summary
    # The raw_llm_output for LiteLLM will be the string representation of the response object
    assert isinstance(result_df.iloc[0]["raw_llm_output"], str)
    assert expected_summary in result_df.iloc[0]["raw_llm_output"]


def test_summarize_text_llm_empty_series(npt_llm_summarize_instance):
    if not MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE: # Represents LITELLM_AVAILABLE in core
        pytest.skip("LiteLLM not available.")

    empty_series = pd.Series([], dtype=str)
    result_df = npt_llm_summarize_instance.summarize_text_llm(
        empty_series, model="any/model"
    )
    assert result_df.empty
    assert list(result_df.columns) == ["original_text", "summary", "raw_llm_output"]


@patch('litellm.completion') # Mock litellm.completion
def test_summarize_text_llm_empty_string_in_series(mock_litellm_completion, npt_llm_summarize_instance):
    if not MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE:
        pytest.skip("LiteLLM not available.")

    test_series = pd.Series([""])
    result_df = npt_llm_summarize_instance.summarize_text_llm(
        test_series, model="any/model", use_chunking=False
    )
    assert len(result_df) == 1
    assert result_df.iloc[0]["original_text"] == ""
    assert result_df.iloc[0]["summary"] == ""
    assert "Input text was empty or whitespace." in result_df.iloc[0]["raw_llm_output"]
    mock_litellm_completion.assert_not_called() # LLM should not be called


@patch('litellm.completion') # Mock litellm.completion
def test_summarize_text_llm_api_error(mock_litellm_completion, npt_llm_summarize_instance):
    if not LITELLM_AVAILABLE_FOR_TEST_SUMMARIZE or not MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE:
        pytest.skip("LiteLLM not available for API error test.")

    mock_litellm_completion.side_effect = litellm.exceptions.APIConnectionError("Simulated LiteLLM API Error")

    test_series = pd.Series(["Some text that will cause an API error."])
    with patch('builtins.print') as mock_print:
        result_df = npt_llm_summarize_instance.summarize_text_llm(
            test_series, model="error/model", use_chunking=False, api_key="key"
        )

    assert len(result_df) == 1
    assert result_df.iloc[0]["summary"] == "error"
    assert "Simulated LiteLLM API Error" in result_df.iloc[0]["raw_llm_output"]
    assert "Direct summarization error" in result_df.iloc[0]["raw_llm_output"] # Check for our method's error context
    printed_output = "".join(call.args[0] for call in mock_print.call_args_list if call.args)
    assert "Error during direct summarization" in printed_output # Check for console print
    assert "Simulated LiteLLM API Error" in printed_output

# Tests for chunking logic with LiteLLM

@patch('nlplot_llm.llm.summarize.litellm.completion')
@patch('nlplot_llm.llm.summarize._chunk_text')
def test_summarize_text_llm_with_chunking_and_combine(mock_chunk_text, mock_litellm_completion, npt_llm_summarize_instance):
    if not LITELLM_AVAILABLE_FOR_TEST_SUMMARIZE or not MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE:
        pytest.skip("LiteLLM or its dependencies not available.")

    chunks = ["First part of the long text.", "Second part of the long text."]
    mock_chunk_text.return_value = chunks

    summary_chunk1 = "Summary of first part."
    summary_chunk2 = "Summary of second part."
    final_combined_summary = "Final combined summary of both parts."

    mock_litellm_completion.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content=summary_chunk1))]), # Chunk 1
        MagicMock(choices=[MagicMock(message=MagicMock(content=summary_chunk2))]), # Chunk 2
        MagicMock(choices=[MagicMock(message=MagicMock(content=final_combined_summary))]) # Combine
    ]

    long_text = "".join(chunks)
    text_series = pd.Series([long_text])
    test_model_str = "openai/gpt-3.5-turbo"

    chunk_prompt_str = "Summarize this piece: {text}"
    combine_prompt_str = "Combine these summaries: {text}"

    result_df = npt_llm_summarize_instance.summarize_text_llm(
        text_series=text_series,
        model=test_model_str,
        use_chunking=True,
        chunk_size=50,
        chunk_prompt_template_str=chunk_prompt_str,
        combine_prompt_template_str=combine_prompt_str,
        api_key="test_key_chunk_summarize"
    )

    assert mock_chunk_text.called_once_with(long_text, chunk_size=50, chunk_overlap=100)
    assert mock_litellm_completion.call_count == 3

    # Check messages for each call
    # Call 1 (Chunk 1)
    args_chunk1, kwargs_chunk1 = mock_litellm_completion.call_args_list[0]
    assert kwargs_chunk1['messages'][0]['content'] == chunk_prompt_str.format(text=chunks[0])
    # Call 2 (Chunk 2)
    args_chunk2, kwargs_chunk2 = mock_litellm_completion.call_args_list[1]
    assert kwargs_chunk2['messages'][0]['content'] == chunk_prompt_str.format(text=chunks[1])
    # Call 3 (Combine)
    args_combine, kwargs_combine = mock_litellm_completion.call_args_list[2]
    expected_combined_text_for_prompt = f"{summary_chunk1}\n\n{summary_chunk2}" # Note: implementation joins with \n\n
    assert kwargs_combine['messages'][0]['content'] == combine_prompt_str.format(text=expected_combined_text_for_prompt)


    assert len(result_df) == 1
    assert result_df.iloc[0]["original_text"] == long_text
    assert result_df.iloc[0]["summary"] == final_combined_summary

    raw_output = result_df.iloc[0]["raw_llm_output"]
    assert f"Chunk 1/{len(chunks)} Summary Raw:" in raw_output # Check new raw output format
    assert f"Chunk 2/{len(chunks)} Summary Raw:" in raw_output
    assert f"Final Combined Summary Raw:" in raw_output


@patch('nlplot_llm.llm.summarize.litellm.completion')
@patch('nlplot_llm.llm.summarize._chunk_text')
def test_summarize_text_llm_chunking_no_combine_prompt(mock_chunk_text, mock_litellm_completion, npt_llm_summarize_instance):
    if not LITELLM_AVAILABLE_FOR_TEST_SUMMARIZE or not MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE:
        pytest.skip("LiteLLM not available.")

    chunks = ["Chunk A.", "Chunk B."]
    mock_chunk_text.return_value = chunks

    summary_chunk_A = "Summary A."
    summary_chunk_B = "Summary B."
    mock_litellm_completion.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content=summary_chunk_A))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content=summary_chunk_B))])
    ]

    text_series = pd.Series(["Chunk A. Chunk B."])
    result_df = npt_llm_summarize_instance.summarize_text_llm(
        text_series, model="ollama/m", use_chunking=True, chunk_size=5,
    )

    assert mock_litellm_completion.call_count == 2
    assert result_df.iloc[0]["summary"] == f"{summary_chunk_A}\n\n{summary_chunk_B}" # Joined with \n\n now
    raw_output = result_df.iloc[0]["raw_llm_output"]
    assert f"Chunk 1/{len(chunks)} Summary Raw:" in raw_output
    assert f"Chunk 2/{len(chunks)} Summary Raw:" in raw_output
    assert "Final Combined Summary Raw:" not in raw_output

# TODO:
# - Test case where chunking results in only one chunk (combine step should be skipped).
# - Test case where chunk_prompt_template_str is invalid.
# - Test case where combine_prompt_template_str is invalid.
# - Test case where a chunk summarization fails but others succeed.

@pytest.mark.parametrize(
    "exception_type, error_message_detail",
    [
        (litellm.exceptions.AuthenticationError, "Simulated LiteLLM Auth Error for summarize"),
        (litellm.exceptions.RateLimitError, "Simulated LiteLLM Rate Limit Error for summarize"),
        (litellm.exceptions.APIConnectionError, "Simulated LiteLLM API Connection Error for summarize"),
        (Exception, "Generic Exception for summarize")
    ]
)
@patch('litellm.completion')
def test_summarize_text_llm_various_api_errors(mock_litellm_completion, exception_type, error_message_detail, npt_llm_summarize_instance):
    if not LITELLM_AVAILABLE_FOR_TEST_SUMMARIZE or not MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE:
        pytest.skip("LiteLLM not available for API error tests.")

    mock_litellm_completion.side_effect = exception_type(error_message_detail)
    test_series = pd.Series(["Text to cause error in summarization."])

    with patch('builtins.print') as mock_print:
        result_df = npt_llm_summarize_instance.summarize_text_llm(
            test_series, model="error/model", use_chunking=False # Test direct summarization error
        )

    assert len(result_df) == 1
    assert result_df.iloc[0]["summary"] == "error"
    assert error_message_detail in result_df.iloc[0]["raw_llm_output"]

    exception_name_in_output = isinstance(exception_type(error_message_detail), litellm.exceptions.APIConnectionError) or \
                               isinstance(exception_type(error_message_detail), litellm.exceptions.AuthenticationError) or \
                               isinstance(exception_type(error_message_detail), litellm.exceptions.RateLimitError)
    if exception_name_in_output:
        assert exception_type.__name__ in result_df.iloc[0]["raw_llm_output"]

    printed_output_str = "".join(call.args[0] for call in mock_print.call_args_list if call.args)
    assert "Error during direct summarization" in printed_output_str # Check specific error context
    assert error_message_detail in printed_output_str


def test_summarize_text_llm_invalid_direct_prompt(npt_llm_summarize_instance):
    if not MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE:
        pytest.skip("LiteLLM not available.")

    test_series = pd.Series(["Some text"])
    invalid_prompt = "This prompt is missing the required placeholder."
    with patch('builtins.print') as mock_print:
        result_df = npt_llm_summarize_instance.summarize_text_llm(
            test_series, model="any/model", use_chunking=False, prompt_template_str=invalid_prompt
        )

    assert len(result_df) == 1
    assert result_df.iloc[0]["summary"] == "error_prompt_direct"
    assert "Direct prompt error" in result_df.iloc[0]["raw_llm_output"]
    mock_print.assert_any_call("Error: Direct summarization prompt template must include '{text}' placeholder.")


@patch('nlplot_llm.llm.summarize._chunk_text') # Mock chunking to focus on prompt error
def test_summarize_text_llm_invalid_chunk_prompt(mock_chunk_text, npt_llm_summarize_instance):
    if not MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE:
        pytest.skip("LiteLLM not available.")

    mock_chunk_text.return_value = ["chunk1", "chunk2"] # Simulate some chunks
    test_series = pd.Series(["Some long text"])
    invalid_chunk_prompt = "This chunk prompt is invalid."

    with patch('litellm.completion'): # Mock litellm to prevent actual calls
        result_df = npt_llm_summarize_instance.summarize_text_llm(
            test_series, model="any/model", use_chunking=True, chunk_prompt_template_str=invalid_chunk_prompt
        )

    assert len(result_df) == 1
    assert result_df.iloc[0]["summary"] == "error_prompt_chunk"
    assert "Chunk prompt template error: missing {text}" in result_df.iloc[0]["raw_llm_output"]


@patch('nlplot_llm.llm.summarize._chunk_text')
@patch('nlplot_llm.llm.summarize.litellm.completion')
def test_summarize_text_llm_invalid_combine_prompt(mock_litellm_completion, mock_chunk_text, npt_llm_summarize_instance):
    if not MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE:
        pytest.skip("LiteLLM not available.")

    mock_chunk_text.return_value = ["chunk1 text", "chunk2 text"]
    # Simulate successful chunk summaries
    mock_litellm_completion.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="Summary of chunk1."))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="Summary of chunk2."))]),
        # The combine call will use the invalid prompt, but we don't need to mock its response
        # as the error should be caught before or during the call.
        # However, the current implementation prints a warning and uses intermediate if combine prompt is bad.
    ]

    test_series = pd.Series(["A very long text that needs chunking and combining."])
    invalid_combine_prompt = "This combine prompt is invalid (no {text})."

    with patch('builtins.print') as mock_print:
        result_df = npt_llm_summarize_instance.summarize_text_llm(
            test_series,
            model="any/model",
            use_chunking=True,
            chunk_prompt_template_str="Summarize: {text}", # Valid chunk prompt
            combine_prompt_template_str=invalid_combine_prompt
        )

    assert len(result_df) == 1
    # Current behavior: if combine_prompt is invalid, it uses the concatenated chunk summaries.
    expected_intermediate_summary = "Summary of chunk1.\n\nSummary of chunk2."
    assert result_df.iloc[0]["summary"] == expected_intermediate_summary

    printed_output = "".join(call.args[0] for call in mock_print.call_args_list if call.args)
    assert "Warning: combine_prompt_template_str does not contain '{text}'. Using combined chunk summaries directly." in printed_output
    assert "Final Combined Summary Raw:" not in result_df.iloc[0]["raw_llm_output"] # Combine call should not have happened effectively


@patch('litellm.completion')
@patch('nlplot_llm.core.LANGCHAIN_SPLITTERS_AVAILABLE', False) # Simulate splitters not available
def test_summarize_text_llm_chunking_true_but_splitters_unavailable(mock_litellm_completion, npt_llm_summarize_instance):
    if not MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE:
        pytest.skip("LiteLLM not available.")

    test_text = "This is some text that would normally be chunked."
    test_series = pd.Series([test_text])
    expected_direct_summary = "Direct summary because chunking failed."

    # Mock litellm.completion for the fallback direct summarization call
    mock_litellm_completion.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content=expected_direct_summary))])

    with patch('builtins.print') as mock_print:
        result_df = npt_llm_summarize_instance.summarize_text_llm(
            test_series, model="any/model", use_chunking=True # Attempt chunking
        )

    assert len(result_df) == 1
    assert result_df.iloc[0]["summary"] == expected_direct_summary
    assert "Chunking skipped (splitters unavailable). Direct summary raw:" in result_df.iloc[0]["raw_llm_output"]

    printed_output = "".join(call.args[0] for call in mock_print.call_args_list if call.args)
    assert "Warning: Langchain text splitters not available for chunking. Attempting direct summarization." in printed_output

    # Ensure litellm.completion was called once for the direct summarization
    mock_litellm_completion.assert_called_once()
    args, kwargs = mock_litellm_completion.call_args
    # Check if the prompt used was the default direct summarization prompt
    assert "Please summarize the following text concisely: {text}".format(text=test_text) in kwargs['messages'][0]['content']

