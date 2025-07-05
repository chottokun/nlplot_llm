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

from nlplot import NLPlot
try:
    from nlplot.nlplot import LANGCHAIN_AVAILABLE as MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE
except ImportError:
    MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE = False


@pytest.fixture
def npt_llm_summarize_instance(tmp_path):
    """Provides a basic NLPlot instance for LLM summarization tests."""
    df = pd.DataFrame({'text': ["initial setup text for summarization"]})
    output_dir = tmp_path / "llm_summarize_test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    return NLPlot(df, target_col='text', output_file_path=str(output_dir))

# --- TDD for LLM Text Summarization ---

def test_summarize_text_llm_initial_method_missing(npt_llm_summarize_instance):
    """(Red Phase) Ensure summarize_text_llm method is initially missing."""
    with pytest.raises(AttributeError, match="'NLPlot' object has no attribute 'summarize_text_llm'"):
        npt_llm_summarize_instance.summarize_text_llm(
            text_series=pd.Series(["A long piece of text to be summarized."]),
            llm_provider="openai",
            model_name="gpt-3.5-turbo"
        )

# Further tests for Green/Refactor phase will be added below.

@patch('nlplot.nlplot.NLPlot._get_llm_client')
def test_summarize_text_llm_success_no_chunking(mock_get_llm, npt_llm_summarize_instance):
    if not MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE or not LANGCHAIN_TEST_IMPORTS_AVAILABLE_SUMMARIZE:
        pytest.skip("Langchain or its dependencies not available, skipping LLM summarization test.")

    mock_llm_instance = MagicMock(spec=ChatOpenAI if LANGCHAIN_TEST_IMPORTS_AVAILABLE_SUMMARIZE else object)
    expected_summary = "This is a concise summary."
    mock_llm_instance.invoke.return_value = AIMessage(content=expected_summary)
    mock_get_llm.return_value = mock_llm_instance

    test_text = "This is a relatively long piece of text that definitely needs to be summarized into a shorter version to save time for the reader."
    text_series = pd.Series([test_text])

    # Call the method (assuming it exists and basic structure is in place)
    try:
        result_df = npt_llm_summarize_instance.summarize_text_llm(
            text_series=text_series,
            llm_provider="openai",
            model_name="gpt-3.5-turbo",
            use_chunking=False, # Explicitly disable chunking for this test
            openai_api_key="test_key_summarize"
        )

        mock_get_llm.assert_called_once_with(
            llm_provider="openai",
            model_name="gpt-3.5-turbo",
            openai_api_key="test_key_summarize",
            temperature=0.0 # Assuming default temperature if not specified
        )
        mock_llm_instance.invoke.assert_called_once()
        # We can also inspect the prompt passed to invoke if needed

        assert isinstance(result_df, pd.DataFrame)
        assert list(result_df.columns) == ["original_text", "summary", "raw_llm_output"]
        assert len(result_df) == 1
        assert result_df.iloc[0]["original_text"] == test_text
        assert result_df.iloc[0]["summary"] == expected_summary
        assert result_df.iloc[0]["raw_llm_output"] == expected_summary
    except AttributeError:
        pytest.fail("summarize_text_llm method not found. This test should run after the method stub is added.")
    except Exception as e:
        pytest.fail(f"Summarization test failed unexpectedly: {e}")


def test_summarize_text_llm_empty_series(npt_llm_summarize_instance):
    if not MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE:
        pytest.skip("Langchain not available.")

    empty_series = pd.Series([], dtype=str)
    result_df = npt_llm_summarize_instance.summarize_text_llm(
        empty_series, "openai", "any_model"
    )
    assert result_df.empty
    assert list(result_df.columns) == ["original_text", "summary", "raw_llm_output"]


@patch('nlplot.nlplot.NLPlot._get_llm_client')
def test_summarize_text_llm_empty_string_in_series(mock_get_llm, npt_llm_summarize_instance):
    if not MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE:
        pytest.skip("Langchain not available.")

    # No LLM client needed if text is empty, so mock_get_llm shouldn't be called.
    # However, the current stub might call it. Let's assume the implementation handles empty strings before LLM call.

    test_series = pd.Series([""]) # Contains one empty string
    try:
        result_df = npt_llm_summarize_instance.summarize_text_llm(
            text_series, "openai", "any_model", use_chunking=False
        )
        assert len(result_df) == 1
        assert result_df.iloc[0]["original_text"] == ""
        assert result_df.iloc[0]["summary"] == "" # Or some other indicator like "Input was empty"
        assert "empty" in result_df.iloc[0]["raw_llm_output"].lower() # Check for a note about empty input
        mock_get_llm.assert_not_called() # LLM should not be called for empty string
    except AttributeError:
        pytest.fail("summarize_text_llm method not found.")


@patch('nlplot.nlplot.NLPlot._get_llm_client')
def test_summarize_text_llm_api_error(mock_get_llm, npt_llm_summarize_instance):
    if not MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE or not LANGCHAIN_TEST_IMPORTS_AVAILABLE_SUMMARIZE:
        pytest.skip("Langchain not available for API error test.")

    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = Exception("Simulated API Error")
    mock_get_llm.return_value = mock_llm_instance

    test_series = pd.Series(["Some text that will cause an API error."])
    with patch('builtins.print') as mock_print:
        result_df = npt_llm_summarize_instance.summarize_text_llm(
            test_series, "openai", "any_model", use_chunking=False, openai_api_key="key"
        )

    assert len(result_df) == 1
    assert result_df.iloc[0]["summary"] == "error" # Or some error indicator
    assert "Simulated API Error" in result_df.iloc[0]["raw_llm_output"]
    printed_output = "".join(call.args[0] for call in mock_print.call_args_list if call.args)
    assert "Error summarizing text" in printed_output
    assert "Simulated API Error" in printed_output

# Tests for chunking logic will be added once the basic summarization is implemented.

@patch('nlplot.nlplot.NLPlot._get_llm_client')
@patch('nlplot.nlplot.NLPlot._chunk_text') # Mock _chunk_text as well
def test_summarize_text_llm_with_chunking_and_combine(mock_chunk_text, mock_get_llm, npt_llm_summarize_instance):
    if not MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE or not LANGCHAIN_TEST_IMPORTS_AVAILABLE_SUMMARIZE:
        pytest.skip("Langchain or its dependencies not available.")

    mock_llm = MagicMock(spec=ChatOpenAI if LANGCHAIN_TEST_IMPORTS_AVAILABLE_SUMMARIZE else object)
    mock_get_llm.return_value = mock_llm

    # Simulate _chunk_text returning two chunks
    chunks = ["First part of the long text.", "Second part of the long text."]
    mock_chunk_text.return_value = chunks

    # Simulate LLM responses: first for chunk 1, then for chunk 2, then for combining
    summary_chunk1 = "Summary of first part."
    summary_chunk2 = "Summary of second part."
    final_combined_summary = "Final combined summary of both parts."

    # LLM invoke should be called 3 times: chunk1, chunk2, combine
    mock_llm.invoke.side_effect = [
        AIMessage(content=summary_chunk1),
        AIMessage(content=summary_chunk2),
        AIMessage(content=final_combined_summary)
    ]

    long_text = "".join(chunks) # Create a long text for the original_text column
    text_series = pd.Series([long_text])

    chunk_prompt_str = "Summarize this piece: {text}"
    combine_prompt_str = "Combine these summaries: {text}"

    result_df = npt_llm_summarize_instance.summarize_text_llm(
        text_series=text_series,
        llm_provider="openai",
        model_name="gpt-3.5-turbo",
        use_chunking=True,
        chunk_size=50, # Small chunk size to ensure chunking is attempted
        chunk_prompt_template_str=chunk_prompt_str,
        combine_prompt_template_str=combine_prompt_str,
        openai_api_key="test_key_chunk_summarize"
    )

    assert mock_chunk_text.called_once_with(long_text, chunk_size=50, chunk_overlap=100) # Default overlap
    assert mock_llm.invoke.call_count == 3 # Called for each chunk and then for combine

    # Check prompts passed to LLM
    # First call (chunk 1)
    call_args_chunk1 = mock_llm.invoke.call_args_list[0][0][0]
    assert chunks[0] in call_args_chunk1
    assert "Summarize this piece:" in call_args_chunk1

    # Second call (chunk 2)
    call_args_chunk2 = mock_llm.invoke.call_args_list[1][0][0]
    assert chunks[1] in call_args_chunk2
    assert "Summarize this piece:" in call_args_chunk2

    # Third call (combine)
    call_args_combine = mock_llm.invoke.call_args_list[2][0][0]
    expected_combined_text_for_prompt = f"{summary_chunk1}\n{summary_chunk2}"
    assert expected_combined_text_for_prompt in call_args_combine
    assert "Combine these summaries:" in call_args_combine

    assert len(result_df) == 1
    assert result_df.iloc[0]["original_text"] == long_text
    assert result_df.iloc[0]["summary"] == final_combined_summary

    # Check raw_llm_output for structure
    raw_output = result_df.iloc[0]["raw_llm_output"]
    assert f"Chunk 1/{len(chunks)} Summary: {summary_chunk1}" in raw_output
    assert f"Chunk 2/{len(chunks)} Summary: {summary_chunk2}" in raw_output
    assert f"Final Combined Summary Raw Output: {final_combined_summary}" in raw_output


@patch('nlplot.nlplot.NLPlot._get_llm_client')
@patch('nlplot.nlplot.NLPlot._chunk_text')
def test_summarize_text_llm_chunking_no_combine_prompt(mock_chunk_text, mock_get_llm, npt_llm_summarize_instance):
    if not MODULE_LANGCHAIN_AVAILABLE_SUMMARIZE or not LANGCHAIN_TEST_IMPORTS_AVAILABLE_SUMMARIZE:
        pytest.skip("Langchain not available.")

    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm
    chunks = ["Chunk A.", "Chunk B."]
    mock_chunk_text.return_value = chunks

    summary_chunk_A = "Summary A."
    summary_chunk_B = "Summary B."
    mock_llm.invoke.side_effect = [
        AIMessage(content=summary_chunk_A),
        AIMessage(content=summary_chunk_B)
    ]

    text_series = pd.Series(["Chunk A. Chunk B."])
    result_df = npt_llm_summarize_instance.summarize_text_llm(
        text_series, "ollama", "m", use_chunking=True, chunk_size=5,
        # No combine_prompt_template_str provided
    )

    assert mock_llm.invoke.call_count == 2 # Only for chunks
    assert result_df.iloc[0]["summary"] == f"{summary_chunk_A}\n{summary_chunk_B}" # Simple join
    raw_output = result_df.iloc[0]["raw_llm_output"]
    assert f"Chunk 1/{len(chunks)} Summary: {summary_chunk_A}" in raw_output
    assert f"Chunk 2/{len(chunks)} Summary: {summary_chunk_B}" in raw_output
    assert "Final Combined Summary Raw Output:" not in raw_output # Ensure combine step was skipped from raw log

# TODO:
# - Test case where chunking results in only one chunk (combine step should be skipped).
# - Test case where chunk_prompt_template_str is invalid.
# - Test case where combine_prompt_template_str is invalid.
# - Test case where a chunk summarization fails but others succeed.
