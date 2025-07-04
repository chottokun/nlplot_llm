import pytest
import os
from unittest.mock import patch, MagicMock
import pandas as pd

# Attempt to import langchain classes for type hinting and patching targets.
# These imports should mirror how they are (or will be) imported in nlplot.nlplot
try:
    # from langchain_openai import ChatOpenAI # Will be patched as 'nlplot.nlplot.ChatOpenAI'
    # from langchain_community.chat_models.ollama import OllamaChat # Will be patched as 'nlplot.nlplot.OllamaChat'
    LANGCHAIN_TEST_IMPORTS_AVAILABLE = True # Assume they *could* be imported for test setup
except ImportError:
    LANGCHAIN_TEST_IMPORTS_AVAILABLE = False


# Import NLPlot and its LANGCHAIN_AVAILABLE flag
from nlplot import NLPlot
try:
    from nlplot.nlplot import LANGCHAIN_AVAILABLE as MODULE_LANGCHAIN_AVAILABLE
except ImportError:
    MODULE_LANGCHAIN_AVAILABLE = False


@pytest.fixture
def npt_instance(tmp_path):
    """Provides a basic NLPlot instance for LLM tests."""
    df = pd.DataFrame({'text': ["some text"]})
    output_dir = tmp_path / "llm_test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    return NLPlot(df, target_col='text', output_file_path=str(output_dir))

# --- TDD for LLM Client Initialization (Cycle 1) ---

def test_get_llm_client_initial_method_missing(npt_instance):
    """(Red Phase) Ensure the _get_llm_client method itself is initially missing."""
    with pytest.raises(AttributeError, match="'NLPlot' object has no attribute '_get_llm_client'"):
        npt_instance._get_llm_client(llm_provider="openai", model_name="test")


# The following tests are for the Green/Refactor phase once _get_llm_client is implemented.
# They assume 'nlplot.nlplot.ChatOpenAI' and 'nlplot.nlplot.OllamaChat' will be valid patch targets.

@patch('nlplot.nlplot.ChatOpenAI')
def test_get_llm_client_openai_with_kwargs_key(MockChatOpenAIClass, npt_instance):
    if not MODULE_LANGCHAIN_AVAILABLE: # Check if nlplot itself thinks langchain is available
        pytest.skip("Langchain (OpenAI) not available in nlplot module, skipping relevant test.")

    # This test will initially fail if _get_llm_client is just a stub (doesn't call ChatOpenAI)
    # or if ChatOpenAI is not imported in nlplot.nlplot
    try:
        mock_openai_instance = MockChatOpenAIClass.return_value
        with patch.dict(os.environ, {}, clear=True):
            llm_client = npt_instance._get_llm_client(
                llm_provider="openai",
                model_name="gpt-3.5-turbo",
                openai_api_key="test_kwargs_key",
                temperature=0.5
            )
        MockChatOpenAIClass.assert_called_once_with(model="gpt-3.5-turbo", openai_api_key="test_kwargs_key", temperature=0.5)
        assert llm_client == mock_openai_instance
    except AttributeError: # Handles if _get_llm_client doesn't exist yet
        pytest.fail("_get_llm_client method not found. Should exist after initial Red phase for method presence.")
    except ImportError: # Handles if ChatOpenAI is not importable via nlplot.nlplot
        pytest.skip("ChatOpenAI not found in nlplot.nlplot for patching.")


@patch('nlplot.nlplot.ChatOpenAI')
@patch.dict(os.environ, {"OPENAI_API_KEY": "test_env_key"}, clear=True)
def test_get_llm_client_openai_with_env_key(MockChatOpenAIClass, npt_instance):
    if not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("Langchain (OpenAI) not available in nlplot module, skipping relevant test.")
    try:
        mock_openai_instance = MockChatOpenAIClass.return_value
        llm_client = npt_instance._get_llm_client(
            llm_provider="openai",
            model_name="gpt-4",
            temperature=0.7
        )
        MockChatOpenAIClass.assert_called_once_with(model="gpt-4", openai_api_key="test_env_key", temperature=0.7)
        assert llm_client == mock_openai_instance
    except AttributeError:
        pytest.fail("_get_llm_client method not found.")
    except ImportError:
        pytest.skip("ChatOpenAI not found in nlplot.nlplot for patching.")


@patch('nlplot.nlplot.ChatOpenAI')
@patch.dict(os.environ, {}, clear=True)
def test_get_llm_client_openai_missing_key(MockChatOpenAIClass, npt_instance):
    if not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("Langchain (OpenAI) not available in nlplot module, skipping relevant test.")
    try:
        # This assumes _get_llm_client itself raises ValueError before calling ChatOpenAI
        with pytest.raises(ValueError, match="OpenAI API key not found"):
            npt_instance._get_llm_client(llm_provider="openai", model_name="gpt-3.5-turbo")
    except AttributeError:
        pytest.fail("_get_llm_client method not found.")
    except ImportError:
        pytest.skip("ChatOpenAI not found in nlplot.nlplot for patching.")


@patch('nlplot.nlplot.OllamaChat')
def test_get_llm_client_ollama_default_url(MockOllamaChatClass, npt_instance):
    if not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("Langchain (Ollama) not available in nlplot module, skipping relevant test.")
    try:
        mock_ollama_instance = MockOllamaChatClass.return_value
        llm_client = npt_instance._get_llm_client(llm_provider="ollama", model_name="llama2", temperature=0.1)
        MockOllamaChatClass.assert_called_once_with(model="llama2", base_url="http://localhost:11434", temperature=0.1)
        assert llm_client == mock_ollama_instance
    except AttributeError:
        pytest.fail("_get_llm_client method not found.")
    except ImportError:
        pytest.skip("OllamaChat not found in nlplot.nlplot for patching.")


@patch('nlplot.nlplot.OllamaChat')
def test_get_llm_client_ollama_custom_url(MockOllamaChatClass, npt_instance):
    if not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("Langchain (Ollama) not available in nlplot module, skipping relevant test.")
    try:
        mock_ollama_instance = MockOllamaChatClass.return_value
        custom_url = "http://customhost:12345"
        llm_client = npt_instance._get_llm_client(
            llm_provider="ollama",
            model_name="mistral",
            base_url=custom_url,
            temperature=0.2
        )
        MockOllamaChatClass.assert_called_once_with(model="mistral", base_url=custom_url, temperature=0.2)
        assert llm_client == mock_ollama_instance
    except AttributeError:
        pytest.fail("_get_llm_client method not found.")
    except ImportError:
        pytest.skip("OllamaChat not found in nlplot.nlplot for patching.")


def test_get_llm_client_unsupported_provider(npt_instance):
    if not MODULE_LANGCHAIN_AVAILABLE:
        # If LANGCHAIN_AVAILABLE is False in nlplot.py, _get_llm_client should raise ImportError early.
        with pytest.raises(ImportError, match="Langchain or related packages are not installed"):
             npt_instance._get_llm_client(llm_provider="unknown_provider", model_name="test_model")
        return

    # If LANGCHAIN_AVAILABLE is True, but provider is unknown
    try:
        with pytest.raises(ValueError, match="Unsupported LLM provider: unknown_provider"):
            npt_instance._get_llm_client(llm_provider="unknown_provider", model_name="test_model")
    except AttributeError:
        pytest.fail("_get_llm_client method not found.")


@patch('nlplot.nlplot.LANGCHAIN_AVAILABLE', False)
def test_get_llm_client_langchain_module_not_available(npt_instance):
    with pytest.raises(ImportError, match="Langchain or related packages are not installed"):
        npt_instance._get_llm_client(llm_provider="openai", model_name="test")


@patch('nlplot.nlplot.ChatOpenAI', side_effect=Exception("OpenAI Init Failed"))
def test_get_llm_client_openai_init_fails(MockChatOpenAIClass, npt_instance):
    if not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("Langchain (OpenAI) not available, skipping.")
    try:
        with pytest.raises(ValueError, match="Failed to initialize OpenAI client"):
            npt_instance._get_llm_client(llm_provider="openai", model_name="gpt-3.5-turbo", openai_api_key="fakekey")
    except AttributeError:
        pytest.fail("_get_llm_client method not found.")
    except ImportError:
        pytest.skip("ChatOpenAI not found in nlplot.nlplot for patching.")


@patch('nlplot.nlplot.OllamaChat', side_effect=Exception("Ollama Init Failed"))
def test_get_llm_client_ollama_init_fails(MockOllamaChatClass, npt_instance):
    if not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("Langchain (Ollama) not available, skipping.")
    try:
        with pytest.raises(ValueError, match="Failed to initialize Ollama client"):
            npt_instance._get_llm_client(llm_provider="ollama", model_name="llama2")
    except AttributeError:
        pytest.fail("_get_llm_client method not found.")
    except ImportError:
        pytest.skip("OllamaChat not found in nlplot.nlplot for patching.")

# Note: For the @patch decorators to work correctly (e.g., @patch('nlplot.nlplot.ChatOpenAI')),
# the `nlplot.py` file must ensure that `ChatOpenAI` and `OllamaChat` are importable
# from its namespace, e.g., by having `from langchain_openai import ChatOpenAI` at its top level.
# If these are conditionally imported within `_get_llm_client` itself, the patching paths would
# need to be adjusted or a different mocking strategy used.
# The `MODULE_LANGCHAIN_AVAILABLE` flag from `nlplot.nlplot` is used to determine if nlplot
# itself believes Langchain is usable, which helps in skipping tests appropriately.
# `LANGCHAIN_TEST_IMPORTS_AVAILABLE` is a flag for the test environment itself.
# Ideally, `MODULE_LANGCHAIN_AVAILABLE` should be the primary guard in tests for nlplot's
# Langchain-dependent features.
# The try-except AttributeError/ImportError blocks in the tests are further guards for when
# the TDD cycle is in its very early stages (method/imports not yet in nlplot.py).
# Once the method stub and imports are in nlplot.py, these try-excepts are less critical
# and the main assertions or pytest.raises will define the test outcome.
