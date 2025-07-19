import pandas as pd
import pytest
import plotly.graph_objs
import os
from unittest.mock import patch
from nlplot_llm import NLPlotLLM
from nlplot_llm.core import DEFAULT_FONT_PATH as NLPLOT_LLM_DEFAULT_FONT_PATH


def ensure_font_file_exists(font_path_to_check=NLPLOT_LLM_DEFAULT_FONT_PATH):
    font_dir = os.path.dirname(font_path_to_check)
    if not os.path.exists(font_path_to_check):
        if font_dir:
            os.makedirs(font_dir, exist_ok=True)
        try:
            with open(font_path_to_check, "w") as f:
                f.write("")
        except Exception as e:
            print(
                "Could not create placeholder font file at "
                f"{font_path_to_check}: {e}"
            )
    return font_path_to_check


TTF_FONT_PATH = ensure_font_file_exists() if NLPLOT_LLM_DEFAULT_FONT_PATH else None


@pytest.fixture
def prepare_data():
    target_col = "text"
    texts = [
        "Think rich look poor",
        "When you come to a roadblock, take a detour",
        "When it is dark enough, you can see the stars",
        "Never let your memories be greater than your dreams",
        "Victory is sweetest when you’ve known defeat",
    ]
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
@patch("plotly.offline.plot")
@patch("plotly.offline.iplot")
@patch("IPython.display.display")
@patch("PIL.Image.Image.save")
def test_nlplot_all_stopwords(
    mock_save, mock_display, mock_iplot, mock_plot, mock_print, prepare_instance
):
    npt = prepare_instance

    all_words_in_sample = list(
        set(
            word
            for doc_word_list in npt.df[npt.target_col]
            for word in doc_word_list
            if word.strip()
        )
    )

    if not all_words_in_sample:
        pytest.skip(
            "Sample data resulted in no words, cannot test all_stopwords scenario."
        )

    fig_bar = npt.bar_ngram(stopwords=all_words_in_sample)
    assert isinstance(fig_bar, plotly.graph_objs.Figure)

    fig_tree = npt.treemap(stopwords=all_words_in_sample)
    assert isinstance(fig_tree, plotly.graph_objs.Figure)

    npt.build_graph(stopwords=all_words_in_sample, min_edge_frequency=0)
    assert npt.node_df.empty if hasattr(npt, "node_df") else True
    assert npt.edge_df.empty if hasattr(npt, "edge_df") else True

    npt.co_network()
    npt.sunburst()
    printed_output = "".join(
        call.args[0] for call in mock_print.call_args_list if call.args
    )


try:
    from janome.tokenizer import Token as JanomeTokenInTest

    JANOME_INSTALLED_FOR_TEST = True
except ImportError:
    JANOME_INSTALLED_FOR_TEST = False

    class JanomeTokenInTest:
        pass
