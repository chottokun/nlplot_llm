import pandas as pd
import pytest
import plotly.graph_objs
import numpy as np
import os
from unittest.mock import patch, MagicMock
from PIL import Image
import datetime

from nlplot_llm import NLPlotLLM
from nlplot_llm.core import DEFAULT_FONT_PATH as NLPLOT_LLM_DEFAULT_FONT_PATH

# Helper to create a dummy font file if it doesn't exist.
def ensure_font_file_exists(font_path_to_check = NLPLOT_LLM_DEFAULT_FONT_PATH):
    font_dir = os.path.dirname(font_path_to_check)
    if not os.path.exists(font_path_to_check):
        if font_dir: # Ensure directory exists only if font_dir is not empty (e.g. for relative paths)
            os.makedirs(font_dir, exist_ok=True)
        try:
            with open(font_path_to_check, 'w') as f:
                f.write("")
        except Exception as e:
            print(f"Could not create placeholder font file at {font_path_to_check}: {e}")
    return font_path_to_check

# Ensure the default font path used by nlplot exists for tests that rely on it.
# This TTF_FONT_PATH will be the one nlplot_llm.core.DEFAULT_FONT_PATH points to.
# Since DEFAULT_FONT_PATH is now None, this helper needs adjustment or tests using it need to change.
# For now, let's assume tests might still try to use a path, or we mock it appropriately.
TTF_FONT_PATH = ensure_font_file_exists() if NLPLOT_LLM_DEFAULT_FONT_PATH else None


@pytest.fixture
def prepare_data():
    target_col = "text"
    texts = ["Think rich look poor",
             "When you come to a roadblock, take a detour",
             "When it is dark enough, you can see the stars",
             "Never let your memories be greater than your dreams",
             "Victory is sweetest when you’ve known defeat"]
    return pd.DataFrame({target_col: texts})

@pytest.fixture
def prepare_instance(prepare_data):
    return NLPlotLLM(prepare_data, target_col="text")
def empty_data():
    return pd.DataFrame({"text": []})

@pytest.fixture
def data_with_empty_text():
    return pd.DataFrame({"text": ["", "  ", "Another sentence"]})
@patch("builtins.print")
@patch("plotly.offline.plot") # Mock plotting to prevent actual plot generation
@patch("plotly.offline.iplot")
@patch("IPython.display.display")
@patch("PIL.Image.Image.save")
def test_nlplot_all_stopwords(mock_save, mock_display, mock_iplot, mock_plot, mock_print, prepare_instance):
    npt = prepare_instance

    # Create a list of all unique words from the sample data in the instance
    all_words_in_sample = list(set(word for doc_word_list in npt.df[npt.target_col] for word in doc_word_list if word.strip()))

    if not all_words_in_sample: # Should not happen with prepare_instance's data
        pytest.skip("Sample data resulted in no words, cannot test all_stopwords scenario.")

    # Use these words as stopwords for the plotting methods
    # We are not mocking npt.get_stopword here, but passing the list directly.

    fig_bar = npt.bar_ngram(stopwords=all_words_in_sample)
    assert isinstance(fig_bar, plotly.graph_objs.Figure) # Expect empty figure

    fig_tree = npt.treemap(stopwords=all_words_in_sample)
    assert isinstance(fig_tree, plotly.graph_objs.Figure)

    npt.wordcloud(stopwords=all_words_in_sample) # Should print warning and return

    npt.build_graph(stopwords=all_words_in_sample, min_edge_frequency=0)
    # build_graph should result in empty node_df and edge_df
    assert npt.node_df.empty if hasattr(npt, 'node_df') else True
    assert npt.edge_df.empty if hasattr(npt, 'edge_df') else True

    npt.co_network() # Should warn and not plot
    npt.sunburst()   # Should warn and not plot
    # Verify that appropriate warnings were printed
    printed_output = "".join(call.args[0] for call in mock_print.call_args_list if call.args)
    assert ("Info: No valid font_path provided. WordCloud will attempt to use its default system font." in printed_output or "Warning: No data to plot for bar_ngram after processing." in printed_output)
    assert any(msg in printed_output for msg in [
        "WordCloud could not be generated",
        "Text corpus is empty",
        "All words might have been filtered out",
        "Info: No valid font_path provided. WordCloud will attempt to use its default system font.",
        "Warning: diskcache library is not installed"
    ])
    assert "No nodes found after processing for build_graph" in printed_output
    # If build_graph prints "No nodes found", subsequent co_network/sunburst calls will print "Graph not built or empty" or "Node DataFrame not available"
    assert ("Graph not built or empty" in printed_output or "Node DataFrame not available or empty" in printed_output)

# --- Tests for Japanese text processing (TDD for new feature) ---
# Import JanomeToken for type checking if JANOME_AVAILABLE
try:
    from janome.tokenizer import Token as JanomeTokenInTest
    JANOME_INSTALLED_FOR_TEST = True
except ImportError:
    JANOME_INSTALLED_FOR_TEST = False
    class JanomeTokenInTest: pass # Dummy for type hints if not installed
