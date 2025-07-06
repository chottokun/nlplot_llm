import pytest
import os
import time # For testing cache expiry
import shutil # For cleaning up cache directories
from unittest.mock import patch, MagicMock
import pandas as pd

from nlplot_llm import NLPlotLLM
from nlplot_llm.core import LITELLM_AVAILABLE # To conditionally skip tests

# Determine a temporary cache directory for tests
TEST_CACHE_DIR = "./temp_test_cache"

@pytest.fixture(scope="function", autouse=True)
def manage_cache_dir():
    """Ensure the test cache directory is clean before each test and removed after."""
    if os.path.exists(TEST_CACHE_DIR):
        shutil.rmtree(TEST_CACHE_DIR)
    os.makedirs(TEST_CACHE_DIR, exist_ok=True)
    yield
    if os.path.exists(TEST_CACHE_DIR):
        shutil.rmtree(TEST_CACHE_DIR)

@pytest.fixture
def npt_cache_test_instance_default_cache_on(tmp_path):
    df = pd.DataFrame({'text': ["initial setup text"]})
    # Use a unique output path for each instance if needed, but cache dir is more important here
    return NLPlotLLM(df, target_col='text', output_file_path=str(tmp_path / "outputs_default_cache"),
                     use_cache=True, cache_dir=TEST_CACHE_DIR)

@pytest.fixture
def npt_cache_test_instance_default_cache_off(tmp_path):
    df = pd.DataFrame({'text': ["initial setup text"]})
    return NLPlotLLM(df, target_col='text', output_file_path=str(tmp_path / "outputs_default_cache_off"),
                     use_cache=False, cache_dir=TEST_CACHE_DIR) # Cache dir specified but use_cache is False

@pytest.fixture
def sample_text_series():
    return pd.Series(["This is a test sentence for caching."])

@pytest.fixture
def sample_text_series_alt():
    return pd.Series(["This is an ALTERNATIVE test sentence for caching."])

# --- Red Phase Tests for Caching ---

@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available, skipping cache tests relying on LLM methods.")
@patch('litellm.completion')
def test_cache_works_on_second_call_with_same_input(mock_litellm_completion, npt_cache_test_instance_default_cache_on, sample_text_series):
    """(Red) Test that LLM is not called on the second identical request if cache is active."""
    npt = npt_cache_test_instance_default_cache_on
    mock_response = MagicMock()
    mock_message = MagicMock(content="positive")
    mock_choice = MagicMock(message=mock_message)
    mock_response.choices = [mock_choice]
    mock_litellm_completion.return_value = mock_response

    # First call - should call litellm.completion
    result1_df = npt.analyze_sentiment_llm(sample_text_series, model="test/model_cache_1")
    mock_litellm_completion.assert_called_once()
    assert not result1_df.empty
    assert result1_df.iloc[0]['sentiment'] == "positive"

    # Second call with identical parameters - should NOT call litellm.completion again
    result2_df = npt.analyze_sentiment_llm(sample_text_series, model="test/model_cache_1")
    mock_litellm_completion.assert_called_once() # Should still be 1, not 2
    assert not result2_df.empty
    pd.testing.assert_frame_equal(result1_df, result2_df)


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available.")
@patch('litellm.completion')
def test_cache_disabled_calls_llm_each_time(mock_litellm_completion, npt_cache_test_instance_default_cache_off, sample_text_series):
    """(Red) Test that LLM is called each time if cache is disabled via constructor."""
    npt = npt_cache_test_instance_default_cache_off
    mock_litellm_completion.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="neutral"))])

    npt.analyze_sentiment_llm(sample_text_series, model="test/model_cache_2")
    assert mock_litellm_completion.call_count == 1

    npt.analyze_sentiment_llm(sample_text_series, model="test/model_cache_2")
    assert mock_litellm_completion.call_count == 2


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available.")
@patch('litellm.completion')
def test_cache_disabled_via_method_override(mock_litellm_completion, npt_cache_test_instance_default_cache_on, sample_text_series):
    """(Red) Test that LLM is called each time if cache is disabled via method argument."""
    npt = npt_cache_test_instance_default_cache_on # Cache is ON by default for this instance
    mock_litellm_completion.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="neutral"))])

    # First call, cache disabled by method arg
    npt.analyze_sentiment_llm(sample_text_series, model="test/model_cache_3", use_cache=False)
    assert mock_litellm_completion.call_count == 1

    # Second call, cache still disabled by method arg
    npt.analyze_sentiment_llm(sample_text_series, model="test/model_cache_3", use_cache=False)
    assert mock_litellm_completion.call_count == 2

    # Third call, cache enabled by method arg (or default if None) - should use cache from a potential prior identical call or make new
    # To make this test clean, let's assume it's a new call that populates cache
    mock_litellm_completion.reset_mock()
    npt.analyze_sentiment_llm(sample_text_series, model="test/model_cache_3_variant", use_cache=True) # Populate cache
    assert mock_litellm_completion.call_count == 1
    npt.analyze_sentiment_llm(sample_text_series, model="test/model_cache_3_variant", use_cache=True) # Should use cache
    assert mock_litellm_completion.call_count == 1


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available.")
@patch('litellm.completion')
def test_cache_miss_on_different_text(mock_litellm_completion, npt_cache_test_instance_default_cache_on, sample_text_series, sample_text_series_alt):
    """(Red) Test that cache is missed if input text is different."""
    npt = npt_cache_test_instance_default_cache_on
    mock_litellm_completion.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="positive"))])

    npt.analyze_sentiment_llm(sample_text_series, model="test/model_cache_4")
    assert mock_litellm_completion.call_count == 1

    mock_litellm_completion.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="negative"))]) # Different response for different text
    npt.analyze_sentiment_llm(sample_text_series_alt, model="test/model_cache_4")
    assert mock_litellm_completion.call_count == 2


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available.")
@patch('litellm.completion')
def test_cache_miss_on_different_model(mock_litellm_completion, npt_cache_test_instance_default_cache_on, sample_text_series):
    """(Red) Test that cache is missed if model name is different."""
    npt = npt_cache_test_instance_default_cache_on
    mock_litellm_completion.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="positive"))])

    npt.analyze_sentiment_llm(sample_text_series, model="test/model_A_cache_5")
    assert mock_litellm_completion.call_count == 1

    npt.analyze_sentiment_llm(sample_text_series, model="test/model_B_cache_5")
    assert mock_litellm_completion.call_count == 2


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available.")
@patch('litellm.completion')
def test_cache_miss_on_different_prompt(mock_litellm_completion, npt_cache_test_instance_default_cache_on, sample_text_series):
    """(Red) Test that cache is missed if prompt_template_str is different."""
    npt = npt_cache_test_instance_default_cache_on
    mock_litellm_completion.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="positive"))])

    npt.analyze_sentiment_llm(sample_text_series, model="test/model_cache_6", prompt_template_str="Prompt v1: {text}")
    assert mock_litellm_completion.call_count == 1

    npt.analyze_sentiment_llm(sample_text_series, model="test/model_cache_6", prompt_template_str="Prompt v2: {text}")
    assert mock_litellm_completion.call_count == 2


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available.")
@patch('litellm.completion')
def test_cache_miss_on_different_kwargs(mock_litellm_completion, npt_cache_test_instance_default_cache_on, sample_text_series):
    """(Red) Test that cache is missed if litellm_kwargs are different."""
    npt = npt_cache_test_instance_default_cache_on
    mock_litellm_completion.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="positive"))])

    npt.analyze_sentiment_llm(sample_text_series, model="test/model_cache_7", temperature=0.1)
    assert mock_litellm_completion.call_count == 1

    npt.analyze_sentiment_llm(sample_text_series, model="test/model_cache_7", temperature=0.7)
    assert mock_litellm_completion.call_count == 2

    # Test with a different kwarg
    npt.analyze_sentiment_llm(sample_text_series, model="test/model_cache_7", temperature=0.7, max_tokens=100)
    assert mock_litellm_completion.call_count == 3


# More advanced tests for cache_dir and expiry would require inspecting the file system
# or more intricate mocking of `diskcache.Cache` if used.
# For now, these tests focus on the primary cache hit/miss logic.

# Example of a test that might be needed if we implement expiry:
@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available.")
@patch('litellm.completion')
def test_cache_expiry(mock_litellm_completion, tmp_path, sample_text_series):
    """(Red) Test that cache expires after the specified time."""
    # This test requires the actual cache implementation to respect expiry.
    # It also needs careful management of time.
    npt_expiring_cache = NLPlotLLM(
        pd.DataFrame({'text': ["text"]}), target_col='text',
        use_cache=True, cache_dir=os.path.join(TEST_CACHE_DIR, "expiry_test"),
        cache_expire=1 # 1 second expiry
    )
    mock_litellm_completion.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="cached_val"))])

    # First call, populates cache
    npt_expiring_cache.analyze_sentiment_llm(sample_text_series, model="test/model_expiry")
    assert mock_litellm_completion.call_count == 1

    # Second call, immediately, should use cache
    npt_expiring_cache.analyze_sentiment_llm(sample_text_series, model="test/model_expiry")
    assert mock_litellm_completion.call_count == 1

    # Wait for cache to expire
    time.sleep(1.5) # Wait for 1.5 seconds

    # Third call, after expiry, should call LLM again
    npt_expiring_cache.analyze_sentiment_llm(sample_text_series, model="test/model_expiry")
    assert mock_litellm_completion.call_count == 2


# Placeholder for a test for a different LLM method (e.g., categorize) to ensure cache works across methods
@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available.")
@patch('litellm.completion')
def test_cache_works_for_categorize_llm(mock_litellm_completion, npt_cache_test_instance_default_cache_on, sample_text_series):
    """(Red) Test cache functionality for categorize_text_llm."""
    npt = npt_cache_test_instance_default_cache_on
    mock_litellm_completion.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="category_A"))])
    categories = ["A", "B"]

    npt.categorize_text_llm(sample_text_series, categories=categories, model="test/model_cat_cache")
    assert mock_litellm_completion.call_count == 1
    npt.categorize_text_llm(sample_text_series, categories=categories, model="test/model_cat_cache")
    assert mock_litellm_completion.call_count == 1 # Should still be 1
```
