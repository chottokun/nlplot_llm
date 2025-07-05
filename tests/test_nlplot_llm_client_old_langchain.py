import pytest
import os
from unittest.mock import patch, MagicMock
import pandas as pd

# This entire file is skipped because _get_llm_client is being removed
# in favor of direct LiteLLM integration.
pytest.skip("Skipping old Langchain client tests; _get_llm_client is deprecated.", allow_module_level=True)

# Attempt to import langchain classes for type hinting and patching targets.
# These imports should mirror how they are (or will be) imported in nlplot.nlplot
try:
    # from langchain_openai import ChatOpenAI # Will be patched as 'nlplot.nlplot.ChatOpenAI'
    # from langchain_community.chat_models.ollama import OllamaChat # Will be patched as 'nlplot.nlplot.OllamaChat'
    LANGCHAIN_TEST_IMPORTS_AVAILABLE = True # Assume they *could* be imported for test setup
except ImportError:
    LANGCHAIN_TEST_IMPORTS_AVAILABLE = False


# Import NLPlot and its LANGCHAIN_AVAILABLE flag
from nlplot_llm import NLPlotLLM # Updated import
try:
    from nlplot_llm.core import LANGCHAIN_AVAILABLE as MODULE_LANGCHAIN_AVAILABLE # Updated path
except ImportError:
    MODULE_LANGCHAIN_AVAILABLE = False


@pytest.fixture
def npt_instance(tmp_path):
    """Provides a basic NLPlotLLM instance for LLM tests.""" # Updated class name in docstring
    df = pd.DataFrame({'text': ["some text"]})
    output_dir = tmp_path / "llm_test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    return NLPlotLLM(df, target_col='text', output_file_path=str(output_dir)) # Updated class name

# --- TDD for LLM Client Initialization (Cycle 1) ---

def test_get_llm_client_initial_method_missing(npt_instance):
    """(Red Phase) Ensure the _get_llm_client method itself is initially missing."""
    with pytest.raises(AttributeError, match="'NLPlotLLM' object has no attribute '_get_llm_client'"): # Updated class name
        npt_instance._get_llm_client(llm_provider="openai", model_name="test")


# The following tests are for the Green/Refactor phase once _get_llm_client is implemented.
# They assume 'nlplot_llm.core.ChatOpenAI' and 'nlplot_llm.core.OllamaChat' will be valid patch targets.

@patch('nlplot_llm.core.ChatOpenAI') # Updated patch target
def test_get_llm_client_openai_with_kwargs_key(MockChatOpenAIClass, npt_instance):
    if not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("Langchain (OpenAI) not available in nlplot_llm module, skipping relevant test.")

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
    except AttributeError:
        pytest.fail("_get_llm_client method not found. Should exist after initial Red phase for method presence.")
    except ImportError:
        pytest.skip("ChatOpenAI not found in nlplot_llm.core for patching.")


@patch('nlplot_llm.core.ChatOpenAI') # Updated patch target
@patch.dict(os.environ, {"OPENAI_API_KEY": "test_env_key"}, clear=True)
def test_get_llm_client_openai_with_env_key(MockChatOpenAIClass, npt_instance):
    if not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("Langchain (OpenAI) not available in nlplot_llm module, skipping relevant test.")
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
        pytest.skip("ChatOpenAI not found in nlplot_llm.core for patching.")


@patch('nlplot_llm.core.ChatOpenAI') # Updated patch target
@patch.dict(os.environ, {}, clear=True)
def test_get_llm_client_openai_missing_key(MockChatOpenAIClass, npt_instance):
    if not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("Langchain (OpenAI) not available in nlplot_llm module, skipping relevant test.")
    try:
        with pytest.raises(ValueError, match="OpenAI API key not found"):
            npt_instance._get_llm_client(llm_provider="openai", model_name="gpt-3.5-turbo")
    except AttributeError:
        pytest.fail("_get_llm_client method not found.")
    except ImportError:
        pytest.skip("ChatOpenAI not found in nlplot_llm.core for patching.")


@patch('nlplot_llm.core.OllamaChat') # Updated patch target
def test_get_llm_client_ollama_default_url(MockOllamaChatClass, npt_instance):
    if not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("Langchain (Ollama) not available in nlplot_llm module, skipping relevant test.")
    try:
        mock_ollama_instance = MockOllamaChatClass.return_value
        llm_client = npt_instance._get_llm_client(llm_provider="ollama", model_name="llama2", temperature=0.1)
        MockOllamaChatClass.assert_called_once_with(model="llama2", base_url="http://localhost:11434", temperature=0.1)
        assert llm_client == mock_ollama_instance
    except AttributeError:
        pytest.fail("_get_llm_client method not found.")
    except ImportError:
        pytest.skip("OllamaChat not found in nlplot_llm.core for patching.")


@patch('nlplot_llm.core.OllamaChat') # Updated patch target
def test_get_llm_client_ollama_custom_url(MockOllamaChatClass, npt_instance):
    if not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("Langchain (Ollama) not available in nlplot_llm module, skipping relevant test.")
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
        pytest.skip("OllamaChat not found in nlplot_llm.core for patching.")


def test_get_llm_client_unsupported_provider(npt_instance):
    if not MODULE_LANGCHAIN_AVAILABLE:
        with pytest.raises(ImportError, match="Langchain or related packages are not installed"):
             npt_instance._get_llm_client(llm_provider="unknown_provider", model_name="test_model")
        return

    try:
        with pytest.raises(ValueError, match="Unsupported LLM provider: 'unknown_provider'"): # Updated match string
            npt_instance._get_llm_client(llm_provider="unknown_provider", model_name="test_model")
    except AttributeError:
        pytest.fail("_get_llm_client method not found.")


@patch('nlplot_llm.core.LANGCHAIN_AVAILABLE', False) # Updated patch target
def test_get_llm_client_langchain_module_not_available(npt_instance):
    with pytest.raises(ImportError, match="Langchain or related packages are not installed"):
        npt_instance._get_llm_client(llm_provider="openai", model_name="test")


@patch('nlplot_llm.core.ChatOpenAI', side_effect=Exception("OpenAI Init Failed")) # Updated patch target
def test_get_llm_client_openai_init_fails(MockChatOpenAIClass, npt_instance):
    if not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("Langchain (OpenAI) not available, skipping.")
    try:
        with pytest.raises(ValueError, match="Failed to initialize OpenAI client"):
            npt_instance._get_llm_client(llm_provider="openai", model_name="gpt-3.5-turbo", openai_api_key="fakekey")
    except AttributeError:
        pytest.fail("_get_llm_client method not found.")
    except ImportError:
        pytest.skip("ChatOpenAI not found in nlplot_llm.core for patching.")


@patch('nlplot_llm.core.OllamaChat', side_effect=Exception("Ollama Init Failed")) # Updated patch target
def test_get_llm_client_ollama_init_fails(MockOllamaChatClass, npt_instance):
    if not MODULE_LANGCHAIN_AVAILABLE:
        pytest.skip("Langchain (Ollama) not available, skipping.")
    try:
        with pytest.raises(ValueError, match="Failed to initialize Ollama client"):
            npt_instance._get_llm_client(llm_provider="ollama", model_name="llama2")
    except AttributeError:
        pytest.fail("_get_llm_client method not found.")
    except ImportError:
        pytest.skip("OllamaChat not found in nlplot_llm.core for patching.")

# Note: For the @patch decorators to work correctly (e.g., @patch('nlplot_llm.core.ChatOpenAI')),
# the `core.py` file must ensure that `ChatOpenAI` and `OllamaChat` are importable
# from its namespace if _get_llm_client was to remain. Since it's being removed, these specifics are moot
# for this file but relevant for other tests that might mock Langchain components before full LiteLLM switch.
# ... (rest of the comments, if any, can be kept or removed)
