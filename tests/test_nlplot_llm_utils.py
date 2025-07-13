import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock

from nlplot_llm import NLPlotLLM # Updated import

# Attempt to import Langchain components for type checking and direct use in tests if needed.
try:
    from nlplot_llm.core import LANGCHAIN_AVAILABLE as MODULE_LANGCHAIN_AVAILABLE # Updated path
    from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
    # Add other splitters if they will be tested directly or type-hinted.
    LANGCHAIN_SPLITTERS_AVAILABLE_FOR_TEST = True
except ImportError:
    MODULE_LANGCHAIN_AVAILABLE = False # Fallback if flag not found in nlplot_llm.core
    LANGCHAIN_SPLITTERS_AVAILABLE_FOR_TEST = False
    # Dummy classes for type hints if Langchain components are not installed in test environment
    class RecursiveCharacterTextSplitter: # type: ignore
        def __init__(self, chunk_size, chunk_overlap, length_function=None, **kwargs): pass
        def split_text(self, text): return [text] if text else []

    class CharacterTextSplitter: # type: ignore
        def __init__(self, separator, chunk_size, chunk_overlap, length_function=None, **kwargs): pass
        def split_text(self, text): return [text] if text else []

    class CharacterTextSplitter:
        def __init__(self, separator, chunk_size, chunk_overlap, length_function=None, **kwargs): pass
        def split_text(self, text): return text.split(separator) if text else []


@pytest.fixture
def npt_llm_utils_instance(tmp_path):
    """Provides a basic NLPlotLLM instance for LLM utility tests.""" # Updated class name
    df = pd.DataFrame({'text': ["initial setup text for utils"]})
    output_dir = tmp_path / "llm_utils_test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    return NLPlotLLM(df, target_col='text', output_file_path=str(output_dir)) # Updated class name

# --- TDD for Text Chunking (Cycle 4) ---
# This file's tests are currently valid as _chunk_text itself doesn't directly use LiteLLM,
# but relies on Langchain's text splitters. If those imports change, these might need updates.
# For now, just updating class names and patch targets.

def test_chunk_text_initial_method_exists():
    """Ensure _chunk_text method exists in the summarize module."""
    try:
        from nlplot_llm.llm.summarize import _chunk_text
    except ImportError:
        pytest.fail("_chunk_text not found in nlplot_llm.llm.summarize")

@pytest.mark.skipif(not MODULE_LANGCHAIN_AVAILABLE or not LANGCHAIN_SPLITTERS_AVAILABLE_FOR_TEST, reason="Langchain text splitters not available in nlplot_llm.core or test env.")
@patch('nlplot_llm.llm.summarize.RecursiveCharacterTextSplitter')
def test_chunk_text_recursive_char_mocked(MockRecursiveSplitter):
    from nlplot_llm.llm.summarize import _chunk_text
    mock_splitter_instance = MockRecursiveSplitter.return_value
    expected_chunks = ["chunk one", "chunk two", "chunk three"]
    mock_splitter_instance.split_text.return_value = expected_chunks

    long_text = "This is a very long text that needs to be split into several chunks for processing by an LLM." * 20
    chunk_size_param = 150
    chunk_overlap_param = 20

    chunks = _chunk_text(
        long_text,
        strategy="recursive_char",
        chunk_size=chunk_size_param,
        chunk_overlap=chunk_overlap_param
    )

    MockRecursiveSplitter.assert_called_once_with(
        chunk_size=chunk_size_param,
        chunk_overlap=chunk_overlap_param,
        length_function=len
    )
    mock_splitter_instance.split_text.assert_called_once_with(long_text)
    assert chunks == expected_chunks


@pytest.mark.skipif(not MODULE_LANGCHAIN_AVAILABLE or not LANGCHAIN_SPLITTERS_AVAILABLE_FOR_TEST, reason="Langchain text splitters not available.")
def test_chunk_text_short_text_no_chunking():
    from nlplot_llm.llm.summarize import _chunk_text
    short_text = "This is a short text, no chunking needed."
    chunks = _chunk_text(short_text, strategy="recursive_char", chunk_size=1000, chunk_overlap=100)
    assert chunks == [short_text], "Short text should be returned as a single chunk."


@pytest.mark.skipif(not MODULE_LANGCHAIN_AVAILABLE or not LANGCHAIN_SPLITTERS_AVAILABLE_FOR_TEST, reason="Langchain text splitters not available.")
def test_chunk_text_empty_input():
    from nlplot_llm.llm.summarize import _chunk_text
    chunks_empty_str = _chunk_text("", strategy="recursive_char")
    assert chunks_empty_str == [], "Empty string should result in an empty list of chunks."

    chunks_whitespace_str = _chunk_text("     ", strategy="recursive_char")
    assert chunks_whitespace_str == ["     "]

@pytest.mark.skipif(not MODULE_LANGCHAIN_AVAILABLE, reason="Langchain core not available in nlplot module for this test.")
def test_chunk_text_unsupported_strategy():
    from nlplot_llm.llm.summarize import _chunk_text
    with pytest.raises(ValueError, match="Unsupported chunking strategy: invalid_strategy_name"):
        _chunk_text("Some text to chunk.", strategy="invalid_strategy_name")

@pytest.mark.skipif(not MODULE_LANGCHAIN_AVAILABLE or not LANGCHAIN_SPLITTERS_AVAILABLE_FOR_TEST, reason="Langchain text splitters not available.")
@patch('nlplot_llm.llm.summarize.CharacterTextSplitter')
def test_chunk_text_character_strategy_mocked(MockCharacterSplitter):
    from nlplot_llm.llm.summarize import _chunk_text
    mock_splitter_instance = MockCharacterSplitter.return_value
    expected_chunks = ["paragraph one.", "paragraph two."]
    mock_splitter_instance.split_text.return_value = expected_chunks

    text_with_paragraphs = "paragraph one.\n\nparagraph two."
    separator = "\n\n"
    chunk_size_param = 200
    chunk_overlap_param = 0

    chunks = _chunk_text(
        text_with_paragraphs,
        strategy="character",
        chunk_size=chunk_size_param,
        chunk_overlap=chunk_overlap_param,
        separator=separator
    )
    MockCharacterSplitter.assert_called_once_with(
        separator=separator,
        chunk_size=chunk_size_param,
        chunk_overlap=chunk_overlap_param,
        length_function=len
    )
    mock_splitter_instance.split_text.assert_called_once_with(text_with_paragraphs)
    assert chunks == expected_chunks

# Further tests could include:
# - Different chunk_size and chunk_overlap values.
# - Texts that are exactly at the boundary of chunk_size.
# - Texts with various types of separators for CharacterTextSplitter.
# - If other strategies like "sentence_transformers_token" are implemented, tests for those.
# - Behavior when LANGCHAIN_SPLITTERS_AVAILABLE is False (should return original text with warning).
# - Non-string input to _chunk_text.

@patch("nlplot_llm.llm.summarize.LANGCHAIN_SPLITTERS_AVAILABLE", False)
@patch("builtins.print")
def test_chunk_text_langchain_splitters_not_available(mock_print):
    from nlplot_llm.llm.summarize import _chunk_text
    test_text = "This is a test text."
    chunks = _chunk_text(test_text, strategy="recursive_char", chunk_size=10, chunk_overlap=0)
    assert chunks == [test_text]
    mock_print.assert_any_call("Warning: Langchain text splitter components are not installed. Chunking will not be performed; returning original text as a single chunk.")
    chunks_empty = _chunk_text("", strategy="recursive_char")
    assert chunks_empty == []

@pytest.mark.skipif(not LANGCHAIN_SPLITTERS_AVAILABLE_FOR_TEST, reason="Langchain text splitters not available in test environment for detailed tests.")
def test_chunk_text_recursive_char_actual():
    from nlplot_llm.llm.summarize import _chunk_text
    long_text = "This is a long sentence that will be split. This is another long sentence that will also be split."
    chunks = _chunk_text(long_text, strategy="recursive_char", chunk_size=30, chunk_overlap=5)
    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert chunks[0] == "This is a long sentence that"
    assert "will be split." in chunks[1]
    assert sum(len(c) for c in chunks) >= len(long_text)

@pytest.mark.skipif(not LANGCHAIN_SPLITTERS_AVAILABLE_FOR_TEST, reason="Langchain text splitters not available for detailed tests.")
def test_chunk_text_character_actual():
    from nlplot_llm.llm.summarize import _chunk_text
    text_with_specific_sep = "Part1---Part2---Another Long Part3"
    chunks = _chunk_text(
        text_with_specific_sep,
        strategy="character",
        separator="---",
        chunk_size=100,
        chunk_overlap=0
    )
    assert chunks == ["Part1", "Part2", "Another Long Part3"]

