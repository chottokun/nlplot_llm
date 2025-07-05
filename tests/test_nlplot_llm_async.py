import pytest
import asyncio
import pandas as pd
from unittest.mock import patch, MagicMock, AsyncMock

from nlplot_llm import NLPlotLLM
from nlplot_llm.core import LITELLM_AVAILABLE, DISKCACHE_AVAILABLE

# Use the same cache directory management as in cache tests for consistency if testing with cache
from .test_nlplot_llm_cache import TEST_CACHE_DIR, manage_cache_dir

@pytest.fixture
def npt_async_test_instance(tmp_path, manage_cache_dir): # manage_cache_dir ensures TEST_CACHE_DIR is clean
    df = pd.DataFrame({'text': ["initial setup text for async"]})
    # Enable cache for some tests, disable for others via method args
    return NLPlotLLM(df, target_col='text', output_file_path=str(tmp_path / "outputs_async"),
                     use_cache=True, cache_dir=TEST_CACHE_DIR)

@pytest.fixture
def sample_text_series_for_async():
    return pd.Series([
        "This is the first sentence for async processing.",
        "Here comes the second one, slightly longer.",
        "And a third, short and sweet."
    ])

# --- Red Phase Tests for Async Methods ---

@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available, skipping async tests.")
@pytest.mark.asyncio # Mark test as asyncio
async def test_analyze_sentiment_llm_async_basic(npt_async_test_instance, sample_text_series_for_async):
    """(Red) Test basic async sentiment analysis returns expected structure and calls acompletion."""
    npt = npt_async_test_instance

    # Mock litellm.acompletion
    mock_acompletion = AsyncMock()
    # Define side effects for multiple calls if texts are different, or same if processed identically
    mock_acompletion.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="positive"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="negative"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="neutral"))]),
    ]

    with patch('litellm.acompletion', mock_acompletion):
        result_df = await npt.analyze_sentiment_llm_async(
            sample_text_series_for_async,
            model="test/async_model_sentiment",
            use_cache=False # Disable cache for this basic call count test
        )

    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == len(sample_text_series_for_async)
    assert list(result_df.columns) == ["text", "sentiment", "raw_llm_output"]
    assert mock_acompletion.call_count == len(sample_text_series_for_async)

    # Check if results match mocked outputs (order should be preserved)
    assert result_df.iloc[0]["sentiment"] == "positive"
    assert result_df.iloc[1]["sentiment"] == "negative"
    assert result_df.iloc[2]["sentiment"] == "neutral"


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available.")
@pytest.mark.asyncio
async def test_categorize_text_llm_async_basic(npt_async_test_instance, sample_text_series_for_async):
    """(Red) Test basic async categorization."""
    npt = npt_async_test_instance
    categories = ["tech", "sports", "general"]

    mock_acompletion = AsyncMock(side_effect=[
        MagicMock(choices=[MagicMock(message=MagicMock(content="tech"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="sports"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="general"))]),
    ])

    with patch('litellm.acompletion', mock_acompletion):
        result_df = await npt.categorize_text_llm_async(
            sample_text_series_for_async,
            categories=categories,
            model="test/async_model_categorize",
            multi_label=False,
            use_cache=False
        )

    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == len(sample_text_series_for_async)
    assert "category" in result_df.columns
    assert mock_acompletion.call_count == len(sample_text_series_for_async)
    assert result_df.iloc[0]["category"] == "tech"


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available.")
@pytest.mark.asyncio
async def test_summarize_text_llm_async_basic(npt_async_test_instance, sample_text_series_for_async):
    """(Red) Test basic async summarization (non-chunked)."""
    npt = npt_async_test_instance

    mock_acompletion = AsyncMock(side_effect=[
        MagicMock(choices=[MagicMock(message=MagicMock(content="Summary 1"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="Summary 2"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="Summary 3"))]),
    ])

    with patch('litellm.acompletion', mock_acompletion):
        result_df = await npt.summarize_text_llm_async(
            sample_text_series_for_async,
            model="test/async_model_summarize",
            use_chunking=False, # Test direct summarization first
            use_cache=False
        )

    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == len(sample_text_series_for_async)
    assert "summary" in result_df.columns
    assert mock_acompletion.call_count == len(sample_text_series_for_async)
    assert result_df.iloc[0]["summary"] == "Summary 1"


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available.")
@pytest.mark.asyncio
async def test_async_methods_with_concurrency_limit(npt_async_test_instance, sample_text_series_for_async):
    """(Red) Test that concurrency_limit is respected (conceptual test via call count if semaphore mocked)."""
    npt = npt_async_test_instance
    limit = 1

    # This test is hard to verify precisely without deeper asyncio mocking or actual time delays.
    # We'll check that acompletion is called, and trust the semaphore logic once implemented.
    mock_acompletion = AsyncMock(side_effect=[
        MagicMock(choices=[MagicMock(message=MagicMock(content="s"))]) for _ in sample_text_series_for_async
    ])

    with patch('litellm.acompletion', mock_acompletion):
        # We'd need to inspect the usage of asyncio.Semaphore if we were to test it directly.
        # For now, just ensure the method runs and calls acompletion the correct number of times.
        await npt.analyze_sentiment_llm_async(
            sample_text_series_for_async,
            model="test/async_model_concurrency",
            concurrency_limit=limit,
            use_cache=False
        )
    assert mock_acompletion.call_count == len(sample_text_series_for_async)
    # A more robust test would involve a mock Semaphore and checking its acquire/release calls.


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available.")
@pytest.mark.asyncio
async def test_async_method_error_handling(npt_async_test_instance):
    """(Red) Test that errors in some async calls are handled and others succeed."""
    npt = npt_async_test_instance
    texts = pd.Series(["good text", "bad text causes error", "another good one"])

    mock_acompletion = AsyncMock()
    mock_acompletion.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content="positive"))]),
        litellm.exceptions.APIConnectionError("Simulated API error for async"), # Error for the second text
        MagicMock(choices=[MagicMock(message=MagicMock(content="neutral"))]),
    ]

    with patch('litellm.acompletion', mock_acompletion):
        result_df = await npt.analyze_sentiment_llm_async(
            texts, model="test/async_model_error", use_cache=False, return_exceptions=True # Assuming this kwarg for gather
        )

    assert len(result_df) == 3
    assert result_df.iloc[0]["sentiment"] == "positive"
    assert result_df.iloc[1]["sentiment"] == "error"
    assert "Simulated API error for async" in result_df.iloc[1]["raw_llm_output"]
    assert result_df.iloc[2]["sentiment"] == "neutral"
    assert mock_acompletion.call_count == 3


@pytest.mark.skipif(not (LITELLM_AVAILABLE and DISKCACHE_AVAILABLE), reason="LiteLLM or Diskcache not available.")
@pytest.mark.asyncio
async def test_async_method_with_caching(npt_async_test_instance, sample_text_series_for_async):
    """(Red) Test that caching works with async methods."""
    npt = npt_async_test_instance # Cache is ON by default for this instance

    mock_acompletion = AsyncMock(side_effect=[
        MagicMock(choices=[MagicMock(message=MagicMock(content="s1"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="s2"))]),
        MagicMock(choices=[MagicMock(message=MagicMock(content="s3"))]),
    ])

    with patch('litellm.acompletion', mock_acompletion):
        # First call - populates cache
        await npt.analyze_sentiment_llm_async(sample_text_series_for_async, model="test/async_cache")
        assert mock_acompletion.call_count == 3

        # Second call - should use cache
        await npt.analyze_sentiment_llm_async(sample_text_series_for_async, model="test/async_cache")
        assert mock_acompletion.call_count == 3 # Should NOT increment


@pytest.mark.asyncio
async def test_async_method_empty_series_input(npt_async_test_instance):
    """(Red) Test async method with empty input series."""
    npt = npt_async_test_instance
    empty_series = pd.Series([], dtype=str)

    result_df = await npt.analyze_sentiment_llm_async(empty_series, model="test/async_empty")
    assert result_df.empty
    assert list(result_df.columns) == ["text", "sentiment", "raw_llm_output"] # Check expected columns for empty

```
