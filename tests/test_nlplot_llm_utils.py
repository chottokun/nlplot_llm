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

def test_chunk_text_initial_method_missing(npt_llm_utils_instance):
    """(Red Phase) Ensure _chunk_text method is initially missing."""
    with pytest.raises(AttributeError, match="'NLPlotLLM' object has no attribute '_chunk_text'"): # Updated class name
        npt_llm_utils_instance._chunk_text("some long text that needs chunking")

# The tests below are for the Green/Refactor phase, once _chunk_text is implemented.
# They assume 'nlplot_llm.core.RecursiveCharacterTextSplitter' etc. will be valid patch targets.

@pytest.mark.skipif(not MODULE_LANGCHAIN_AVAILABLE or not LANGCHAIN_SPLITTERS_AVAILABLE_FOR_TEST, reason="Langchain text splitters not available in nlplot_llm.core or test env.")
@patch('nlplot_llm.core.RecursiveCharacterTextSplitter') # Updated patch target
def test_chunk_text_recursive_char_mocked(MockRecursiveSplitter, npt_llm_utils_instance):
    npt = npt_llm_utils_instance
    mock_splitter_instance = MockRecursiveSplitter.return_value
    expected_chunks = ["chunk one", "chunk two", "chunk three"]
    mock_splitter_instance.split_text.return_value = expected_chunks

    long_text = "This is a very long text that needs to be split into several chunks for processing by an LLM." * 20
    chunk_size_param = 150 # Example parameter
    chunk_overlap_param = 20 # Example parameter

    try:
        chunks = npt._chunk_text(
            long_text,
            strategy="recursive_char",
            chunk_size=chunk_size_param,
            chunk_overlap=chunk_overlap_param
        )

        # Check that the splitter was initialized with correct parameters
        # Note: length_function=len is often a default in Langchain splitters.
        MockRecursiveSplitter.assert_called_once_with(
            chunk_size=chunk_size_param,
            chunk_overlap=chunk_overlap_param,
            length_function=len # Assuming this default is used or explicitly set
        )
        mock_splitter_instance.split_text.assert_called_once_with(long_text)
        assert chunks == expected_chunks
    except AttributeError:
        pytest.fail("_chunk_text method not found. This test should run after method stub is added.")
    except ImportError: # If RecursiveCharacterTextSplitter is not importable via nlplot_llm.core
        pytest.skip("RecursiveCharacterTextSplitter not found in nlplot_llm.core for patching.")


@pytest.mark.skipif(not MODULE_LANGCHAIN_AVAILABLE or not LANGCHAIN_SPLITTERS_AVAILABLE_FOR_TEST, reason="Langchain text splitters not available.")
def test_chunk_text_short_text_no_chunking(npt_llm_utils_instance):
    npt = npt_llm_utils_instance
    short_text = "This is a short text, no chunking needed."
    try:
        chunks = npt._chunk_text(short_text, strategy="recursive_char", chunk_size=1000, chunk_overlap=100)
        assert chunks == [short_text], "Short text should be returned as a single chunk."
    except AttributeError:
        pytest.fail("_chunk_text method not found.")


@pytest.mark.skipif(not MODULE_LANGCHAIN_AVAILABLE or not LANGCHAIN_SPLITTERS_AVAILABLE_FOR_TEST, reason="Langchain text splitters not available.")
def test_chunk_text_empty_input(npt_llm_utils_instance):
    npt = npt_llm_utils_instance
    try:
        chunks_empty_str = npt._chunk_text("", strategy="recursive_char")
        assert chunks_empty_str == [], "Empty string should result in an empty list of chunks."

        chunks_whitespace_str = npt._chunk_text("     ", strategy="recursive_char")
        assert chunks_whitespace_str == ["     "], "Whitespace-only string should be returned as a single chunk if not empty after strip by splitter."
        # Note: Behavior for whitespace-only might depend on splitter's strip behavior.
        # RecursiveCharacterTextSplitter might return it as is if it's below chunk_size.
    except AttributeError:
        pytest.fail("_chunk_text method not found.")


@pytest.mark.skipif(not MODULE_LANGCHAIN_AVAILABLE, reason="Langchain core not available in nlplot module for this test.")
def test_chunk_text_unsupported_strategy(npt_llm_utils_instance):
    npt = npt_llm_utils_instance
    try:
        with pytest.raises(ValueError, match="Unsupported chunking strategy: invalid_strategy_name"):
            npt._chunk_text("Some text to chunk.", strategy="invalid_strategy_name")
    except AttributeError:
        pytest.fail("_chunk_text method not found.")

@pytest.mark.skipif(not MODULE_LANGCHAIN_AVAILABLE or not LANGCHAIN_SPLITTERS_AVAILABLE_FOR_TEST, reason="Langchain text splitters not available.")
@patch('nlplot_llm.core.CharacterTextSplitter') # Updated patch target
def test_chunk_text_character_strategy_mocked(MockCharacterSplitter, npt_llm_utils_instance):
    npt = npt_llm_utils_instance
    mock_splitter_instance = MockCharacterSplitter.return_value
    expected_chunks = ["paragraph one.", "paragraph two."]
    mock_splitter_instance.split_text.return_value = expected_chunks

    text_with_paragraphs = "paragraph one.\n\nparagraph two."
    separator = "\n\n"
    chunk_size_param = 200
    chunk_overlap_param = 0

    try:
        chunks = npt._chunk_text(
            text_with_paragraphs,
            strategy="character",
            chunk_size=chunk_size_param,
            chunk_overlap=chunk_overlap_param,
            separator=separator # Kwarg for CharacterTextSplitter
        )
        MockCharacterSplitter.assert_called_once_with(
            separator=separator,
            chunk_size=chunk_size_param,
            chunk_overlap=chunk_overlap_param,
            length_function=len
        )
        mock_splitter_instance.split_text.assert_called_once_with(text_with_paragraphs)
        assert chunks == expected_chunks
    except AttributeError:
        pytest.fail("_chunk_text method not found.")
    except ImportError:
        pytest.skip("CharacterTextSplitter not found in nlplot_llm.core for patching.")

# Further tests could include:
# - Different chunk_size and chunk_overlap values.
# - Texts that are exactly at the boundary of chunk_size.
# - Texts with various types of separators for CharacterTextSplitter.
# - If other strategies like "sentence_transformers_token" are implemented, tests for those.
# - Behavior when LANGCHAIN_SPLITTERS_AVAILABLE is False (should return original text with warning).
# - Non-string input to _chunk_text.

@patch("nlplot_llm.core.LANGCHAIN_SPLITTERS_AVAILABLE", False)
@patch("builtins.print")
def test_chunk_text_langchain_splitters_not_available(mock_print, npt_llm_utils_instance):
    """Tests _chunk_text behavior when LANGCHAIN_SPLITTERS_AVAILABLE is False."""
    npt = npt_llm_utils_instance
    test_text = "This is a test text."
    try:
        chunks = npt._chunk_text(test_text, strategy="recursive_char", chunk_size=10, chunk_overlap=0)
        assert chunks == [test_text], "Should return the original text as a single chunk."
        mock_print.assert_any_call("Warning: Langchain text splitter components are not installed. Chunking will not be performed; returning original text as a single chunk.")

        # Test with empty text
        chunks_empty = npt._chunk_text("", strategy="recursive_char")
        assert chunks_empty == [], "Empty string should return empty list even if splitters unavailable."
    except AttributeError:
        pytest.fail("_chunk_text method not found. This test should run after the method stub is added.")

# Further tests for actual chunking logic (assuming LANGCHAIN_SPLITTERS_AVAILABLE is True)

@pytest.mark.skipif(not LANGCHAIN_SPLITTERS_AVAILABLE_FOR_TEST, reason="Langchain text splitters not available in test environment for detailed tests.")
def test_chunk_text_recursive_char_actual(npt_llm_utils_instance):
    """Tests RecursiveCharacterTextSplitter actual behavior via _chunk_text."""
    npt = npt_llm_utils_instance
    long_text = "This is a long sentence that will be split. This is another long sentence that will also be split."
    # Expected: With chunk_size=30, overlap=5 (using RecursiveCharacterTextSplitter defaults for separators)
    # "This is a long sentence that " (length 29)
    # "sentence that will be split. " (overlap "that ", "will be split. ")
    # "split. This is another long "
    # "another long sentence that will "
    # "that will also be split."

    try:
        chunks = npt._chunk_text(long_text, strategy="recursive_char", chunk_size=30, chunk_overlap=5)
        assert isinstance(chunks, list)
        assert len(chunks) > 1 # Expecting multiple chunks
        # More specific assertions can be added if exact output is predictable and stable
        # For example, check if total length of chunks (minus overlaps) is similar to original.
        # Or check if specific parts of the text are in expected chunks.
        # print(f"Recursive Chunks: {chunks}") # For debugging
        assert chunks[0] == "This is a long sentence that" # Adjust based on actual output
        assert "will be split." in chunks[1] # Example check
        assert sum(len(c) for c in chunks) >= len(long_text) # Due to overlap

    except AttributeError:
        pytest.fail("_chunk_text method not found.")
    except Exception as e:
        pytest.fail(f"_chunk_text with RecursiveCharacterTextSplitter failed: {e}")


@pytest.mark.skipif(not LANGCHAIN_SPLITTERS_AVAILABLE_FOR_TEST, reason="Langchain text splitters not available for detailed tests.")
def test_chunk_text_character_actual(npt_llm_utils_instance):
    """Tests CharacterTextSplitter actual behavior via _chunk_text."""
    npt = npt_llm_utils_instance
    text_with_specific_sep = "Part1---Part2---Another Long Part3"

    try:
        chunks = npt._chunk_text(
            text_with_specific_sep,
            strategy="character",
            separator="---",
            chunk_size=10, # chunk_size might be tricky with CharacterTextSplitter if separator makes parts too long
            chunk_overlap=0
        )
        assert isinstance(chunks, list)
        # Expected: ["Part1", "Part2", "Another Lo", "ng Part3"] if chunk_size forces splits within "Another Long Part3"
        # Or simply ["Part1", "Part2", "Another Long Part3"] if chunk_size is large enough for each part
        # Let's test with a chunk_size that doesn't force internal splits for simplicity of this example,
        # focusing on the separator.
        # print(f"Character Chunks (sep='---', size=10): {chunks}")
        # assert chunks == ["Part1", "Part2", "Another Lo", "ng Part3"] # This depends heavily on splitter's internal logic for size

        chunks_larger_size = npt._chunk_text(
            text_with_specific_sep,
            strategy="character",
            separator="---",
            chunk_size=100, # Large enough to not split "Another Long Part3"
            chunk_overlap=0
        )
        assert chunks_larger_size == ["Part1", "Part2", "Another Long Part3"]

    except AttributeError:
        pytest.fail("_chunk_text method not found.")
    except Exception as e:
        pytest.fail(f"_chunk_text with CharacterTextSplitter failed: {e}")

