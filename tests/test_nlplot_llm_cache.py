import pytest
import os
import time
import shutil
from unittest.mock import patch, MagicMock
import pandas as pd

from nlplot_llm import NLPlotLLM
from nlplot_llm.core import LITELLM_AVAILABLE

TEST_CACHE_DIR = "./temp_test_cache"


@pytest.fixture(scope="function", autouse=True)
def manage_cache_dir():
    if os.path.exists(TEST_CACHE_DIR):
        shutil.rmtree(TEST_CACHE_DIR)
    os.makedirs(TEST_CACHE_DIR, exist_ok=True)
    yield
    if os.path.exists(TEST_CACHE_DIR):
        shutil.rmtree(TEST_CACHE_DIR)


@pytest.fixture
def npt_cache_test_instance_default_cache_on(tmp_path):
    df = pd.DataFrame({"text": ["initial setup text"]})
    return NLPlotLLM(
        df,
        target_col="text",
        output_file_path=str(tmp_path / "outputs_default_cache"),
        use_cache=True,
        cache_dir=TEST_CACHE_DIR,
    )


@pytest.fixture
def npt_cache_test_instance_default_cache_off(tmp_path):
    df = pd.DataFrame({"text": ["initial setup text"]})
    return NLPlotLLM(
        df,
        target_col="text",
        output_file_path=str(tmp_path / "outputs_default_cache_off"),
        use_cache=False,
        cache_dir=TEST_CACHE_DIR,
    )


@pytest.fixture
def sample_text_series():
    return pd.Series(["This is a test sentence for caching."])


@pytest.fixture
def sample_text_series_alt():
    return pd.Series(["This is an ALTERNATIVE test sentence for caching."])


@pytest.mark.skipif(
    not LITELLM_AVAILABLE,
    reason="LiteLLM not available, skipping cache tests relying on LLM methods.",
)
@patch("litellm.completion")
def test_cache_works_on_second_call_with_same_input(
    mock_litellm_completion,
    npt_cache_test_instance_default_cache_on,
    sample_text_series,
):
    npt = npt_cache_test_instance_default_cache_on
    mock_response = MagicMock()
    mock_message = MagicMock(content="positive")
    mock_choice = MagicMock(message=mock_message)
    mock_response.choices = [mock_choice]
    mock_litellm_completion.return_value = mock_response

    result1_df = npt.analyze_sentiment_llm(
        sample_text_series, model="test/model_cache_1"
    )
    mock_litellm_completion.assert_called_once()
    assert not result1_df.empty
    assert result1_df.iloc[0]["sentiment"] == "positive"

    result2_df = npt.analyze_sentiment_llm(
        sample_text_series, model="test/model_cache_1"
    )
    mock_litellm_completion.assert_called_once()
    assert not result2_df.empty
    pd.testing.assert_frame_equal(result1_df, result2_df)


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available.")
@patch("litellm.completion")
def test_cache_disabled_calls_llm_each_time(
    mock_litellm_completion,
    npt_cache_test_instance_default_cache_off,
    sample_text_series,
):
    npt = npt_cache_test_instance_default_cache_off
    mock_litellm_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="neutral"))]
    )

    npt.analyze_sentiment_llm(sample_text_series, model="test/model_cache_2")
    assert mock_litellm_completion.call_count == 1

    npt.analyze_sentiment_llm(sample_text_series, model="test/model_cache_2")
    assert mock_litellm_completion.call_count == 2


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available.")
@patch("litellm.completion")
def test_cache_disabled_via_method_override(
    mock_litellm_completion,
    npt_cache_test_instance_default_cache_on,
    sample_text_series,
):
    npt = npt_cache_test_instance_default_cache_on
    mock_litellm_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="neutral"))]
    )

    npt.analyze_sentiment_llm(
        sample_text_series, model="test/model_cache_3", use_cache=False
    )
    assert mock_litellm_completion.call_count == 1

    npt.analyze_sentiment_llm(
        sample_text_series, model="test/model_cache_3", use_cache=False
    )
    assert mock_litellm_completion.call_count == 2

    mock_litellm_completion.reset_mock()
    npt.analyze_sentiment_llm(
        sample_text_series, model="test/model_cache_3_variant", use_cache=True
    )
    assert mock_litellm_completion.call_count == 1
    npt.analyze_sentiment_llm(
        sample_text_series, model="test/model_cache_3_variant", use_cache=True
    )
    assert mock_litellm_completion.call_count == 1


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available.")
@patch("litellm.completion")
def test_cache_miss_on_different_text(
    mock_litellm_completion,
    npt_cache_test_instance_default_cache_on,
    sample_text_series,
    sample_text_series_alt,
):
    npt = npt_cache_test_instance_default_cache_on
    mock_litellm_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="positive"))]
    )

    npt.analyze_sentiment_llm(sample_text_series, model="test/model_cache_4")
    assert mock_litellm_completion.call_count == 1

    mock_litellm_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="negative"))]
    )
    npt.analyze_sentiment_llm(sample_text_series_alt, model="test/model_cache_4")
    assert mock_litellm_completion.call_count == 2


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available.")
@patch("litellm.completion")
def test_cache_miss_on_different_model(
    mock_litellm_completion,
    npt_cache_test_instance_default_cache_on,
    sample_text_series,
):
    npt = npt_cache_test_instance_default_cache_on
    mock_litellm_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="positive"))]
    )

    npt.analyze_sentiment_llm(sample_text_series, model="test/model_A_cache_5")
    assert mock_litellm_completion.call_count == 1

    npt.analyze_sentiment_llm(sample_text_series, model="test/model_B_cache_5")
    assert mock_litellm_completion.call_count == 2


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available.")
@patch("litellm.completion")
def test_cache_miss_on_different_prompt(
    mock_litellm_completion,
    npt_cache_test_instance_default_cache_on,
    sample_text_series,
):
    npt = npt_cache_test_instance_default_cache_on
    mock_litellm_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="positive"))]
    )

    npt.analyze_sentiment_llm(
        sample_text_series,
        model="test/model_cache_6",
        prompt_template_str="Prompt v1: {text}",
    )
    assert mock_litellm_completion.call_count == 1

    npt.analyze_sentiment_llm(
        sample_text_series,
        model="test/model_cache_6",
        prompt_template_str="Prompt v2: {text}",
    )
    assert mock_litellm_completion.call_count == 2


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available.")
@patch("litellm.completion")
def test_cache_miss_on_different_kwargs(
    mock_litellm_completion,
    npt_cache_test_instance_default_cache_on,
    sample_text_series,
):
    npt = npt_cache_test_instance_default_cache_on
    mock_litellm_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="positive"))]
    )

    npt.analyze_sentiment_llm(
        sample_text_series, model="test/model_cache_7", temperature=0.1
    )
    assert mock_litellm_completion.call_count == 1

    npt.analyze_sentiment_llm(
        sample_text_series, model="test/model_cache_7", temperature=0.7
    )
    assert mock_litellm_completion.call_count == 2

    npt.analyze_sentiment_llm(
        sample_text_series, model="test/model_cache_7", temperature=0.7, max_tokens=100
    )
    assert mock_litellm_completion.call_count == 3


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available.")
@patch("litellm.completion")
def test_cache_expiry(mock_litellm_completion, tmp_path, sample_text_series):
    npt_expiring_cache = NLPlotLLM(
        pd.DataFrame({"text": ["text"]}),
        target_col="text",
        use_cache=True,
        cache_dir=os.path.join(TEST_CACHE_DIR, "expiry_test"),
        cache_expire=1,
    )
    mock_litellm_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="cached_val"))]
    )

    npt_expiring_cache.analyze_sentiment_llm(
        sample_text_series, model="test/model_expiry"
    )
    assert mock_litellm_completion.call_count == 1

    npt_expiring_cache.analyze_sentiment_llm(
        sample_text_series, model="test/model_expiry"
    )
    assert mock_litellm_completion.call_count == 1

    time.sleep(1.5)

    npt_expiring_cache.analyze_sentiment_llm(
        sample_text_series, model="test/model_expiry"
    )
    assert mock_litellm_completion.call_count == 2


@pytest.mark.skipif(not LITELLM_AVAILABLE, reason="LiteLLM not available.")
@patch("litellm.completion")
def test_cache_works_for_categorize_llm(
    mock_litellm_completion,
    npt_cache_test_instance_default_cache_on,
    sample_text_series,
):
    npt = npt_cache_test_instance_default_cache_on
    mock_litellm_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="category_A"))]
    )
    categories = ["A", "B"]

    npt.categorize_text_llm(
        sample_text_series, categories=categories, model="test/model_cat_cache"
    )
    assert mock_litellm_completion.call_count == 1
    npt.categorize_text_llm(
        sample_text_series, categories=categories, model="test/model_cat_cache"
    )
    assert mock_litellm_completion.call_count == 1
