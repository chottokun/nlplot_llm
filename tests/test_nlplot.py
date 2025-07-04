import pandas as pd
import pytest
import plotly.graph_objs
import numpy as np
import os
from unittest.mock import patch, mock_open

from nlplot import NLPlot, get_colorpalette, generate_freq_df

# Helper to create a dummy font file if it doesn't exist, as WordCloud needs it.
# This should ideally be part of a test setup/teardown fixture if used across multiple tests.
def ensure_font_file_exists():
    font_dir = "nlplot/data"
    font_path = os.path.join(font_dir, "mplus-1c-regular.ttf")
    if not os.path.exists(font_path):
        os.makedirs(font_dir, exist_ok=True)
        try:
            # Create a minimal, valid TTF if possible, or just an empty file as placeholder
            # For real testing, a proper (small) TTF font would be needed.
            # Here, just creating an empty file to satisfy WordCloud's font_path check.
            # This might still cause issues if WordCloud tries to actually parse the font.
            with open(font_path, 'w') as f:
                f.write("") # Placeholder, WordCloud might complain if it's not a real font
            print(f"Created placeholder font file at {font_path} for testing.")
        except Exception as e:
            print(f"Could not create placeholder font file: {e}")
    return font_path

# Call it once at the start of the test module
TTF_FONT_PATH = ensure_font_file_exists()


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
def empty_data():
    return pd.DataFrame({"text": []})

@pytest.fixture
def data_with_empty_text():
    return pd.DataFrame({"text": ["", "  ", "Another sentence"]})


@pytest.fixture
def prepare_instance(prepare_data):
    # Using a temporary directory for output files during tests might be better
    return NLPlot(prepare_data.copy(), target_col="text", output_file_path="./test_outputs/")

@pytest.fixture
def prepare_instance_empty_df(empty_data):
    return NLPlot(empty_data.copy(), target_col="text", output_file_path="./test_outputs/")


@pytest.fixture
def series_data_for_freq_df(prepare_data):
    # This fixture provides data in the format generate_freq_df expects (Series of space-separated strings)
    return prepare_data["text"].copy()


# --- Test get_colorpalette ---
@pytest.mark.parametrize(
    "color_palette, n_legends, expected_len_check",
    [
        ("hls", 1, lambda l: l == 1),
        ("hls", 10, lambda l: l == 10),
        # ("hls", 100, lambda l: l == 100), # This can be slow with seaborn
    ]
)
def test_get_colorpalette_valid(color_palette, n_legends, expected_len_check):
    rgbs = get_colorpalette(color_palette, n_legends)
    assert isinstance(rgbs, list)
    assert expected_len_check(len(rgbs))
    if n_legends > 0:
        assert all(isinstance(rgb_str, str) and rgb_str.startswith("rgb(") for rgb_str in rgbs)

@pytest.mark.parametrize("invalid_n_colors", [0, -1, 0.5, "abc"])
def test_get_colorpalette_invalid_n_colors(invalid_n_colors):
    with pytest.raises(ValueError, match="n_colors must be a positive integer"):
        get_colorpalette("hls", invalid_n_colors)


# --- Test generate_freq_df ---
@pytest.mark.parametrize(
    "n_gram, top_n, expected_rows_check, check_content_type",
    [
        (1, 1, lambda r: r <= 1, True),
        (1, 50, lambda r: r <= 50, True), # Max possible ngrams from sample
        (2, 3, lambda r: r <= 3, True),
        (1, 0, lambda r: r == 0, False),
    ]
)
def test_generate_freq_df_valid(series_data_for_freq_df, n_gram, top_n, expected_rows_check, check_content_type):
    word_frequency = generate_freq_df(series_data_for_freq_df, n_gram=n_gram, top_n=top_n,
                                      stopwords=[], verbose=False)
    expect_columns = ["word", "word_count"]
    assert isinstance(word_frequency, pd.DataFrame)
    assert word_frequency.ndim == 2
    assert expected_rows_check(len(word_frequency))
    assert list(word_frequency.columns) == expect_columns
    if check_content_type and top_n > 0 and not word_frequency.empty:
        assert isinstance(word_frequency["word"].iloc[0], str)
        assert isinstance(word_frequency["word_count"].iloc[0], (int, np.integer))


@pytest.mark.parametrize("invalid_n_gram", [0, -1, 0.5, "abc"])
def test_generate_freq_df_invalid_n_gram(series_data_for_freq_df, invalid_n_gram):
    with pytest.raises(ValueError, match="n_gram must be a positive integer"):
        generate_freq_df(series_data_for_freq_df, n_gram=invalid_n_gram, top_n=10)

@pytest.mark.parametrize("invalid_top_n", [-1, 0.5, "abc"])
def test_generate_freq_df_invalid_top_n(series_data_for_freq_df, invalid_top_n):
    with pytest.raises(ValueError, match="top_n must be a non-negative integer"):
        generate_freq_df(series_data_for_freq_df, n_gram=1, top_n=invalid_top_n)

def test_generate_freq_df_empty_input_series():
    empty_series = pd.Series([], dtype=str)
    df_empty = generate_freq_df(empty_series, n_gram=1, top_n=10)
    assert df_empty.empty
    assert list(df_empty.columns) == ["word", "word_count"]

def test_generate_freq_df_with_stopwords(series_data_for_freq_df):
    # "you" is a common word in the sample
    stopwords = ["you", "to"]
    df_with_stopwords = generate_freq_df(series_data_for_freq_df, n_gram=1, top_n=10, stopwords=stopwords)
    if not df_with_stopwords.empty:
        assert "you" not in df_with_stopwords["word"].tolist()
        assert "to" not in df_with_stopwords["word"].tolist()


# --- Test NLPlot class ---
def test_nlplot_init(prepare_data):
    npt = NLPlot(prepare_data.copy(), target_col="text")
    assert isinstance(npt.df, pd.DataFrame)
    if not npt.df.empty:
        assert isinstance(npt.df["text"].iloc[0], list)

def test_nlplot_init_with_default_stopwords_file(tmp_path):
    stopwords_content = "stopword1\nstopword2\n"
    # Create a temporary stopwords file
    p = tmp_path / "stopwords.txt"
    p.write_text(stopwords_content)

    df = pd.DataFrame({"text": ["stopword1 and testword1"]})
    npt = NLPlot(df, target_col="text", default_stopwords_file_path=str(p))
    assert "stopword1" in npt.default_stopwords
    assert "stopword2" in npt.default_stopwords
    # Test that these stopwords are used
    stopwords_calc = npt.get_stopword(top_n=0, min_freq=0) # only default should be there
    assert "stopword1" in stopwords_calc

def test_nlplot_init_with_malformed_text_column(prepare_data):
    # Test if non-string data in target column is handled (astype(str) should manage this)
    df_malformed = prepare_data.copy()
    df_malformed.loc[0, "text"] = 12345 # number instead of string
    npt = NLPlot(df_malformed, target_col="text")
    assert isinstance(npt.df["text"].iloc[0], list)
    assert npt.df["text"].iloc[0] == ["12345"]


@pytest.mark.parametrize("top_n, min_freq", [(1,0), (0,1), (0,0)])
def test_nlplot_get_stopword(prepare_instance, top_n, min_freq):
    npt = prepare_instance
    stopwords = npt.get_stopword(top_n=top_n, min_freq=min_freq)
    assert isinstance(stopwords, list)
    # Specific checks depend on the data and params
    if top_n == 1 and not npt.df.empty:
         # 'you' is the most frequent word (2 times) in the sample data
         # 'when' is also 2 times, 'the' is also 2 times. Order might vary.
        most_commons = [item[0] for item in Counter(word for doc in npt.df[npt.target_col] for word in doc).most_common(top_n)]
        for common_word in most_commons:
            assert common_word in stopwords

@pytest.mark.parametrize("invalid_input", [-1, "a"])
def test_nlplot_get_stopword_invalid_params(prepare_instance, invalid_input):
    npt = prepare_instance
    with pytest.raises(ValueError):
        npt.get_stopword(top_n=invalid_input, min_freq=0)
    with pytest.raises(ValueError):
        npt.get_stopword(top_n=0, min_freq=invalid_input)


# --- Test Plotting Methods (basic checks for figure type and no exceptions) ---
@patch("plotly.offline.plot") # Mock to prevent actual file saving/opening
def test_nlplot_bar_ngram(mock_plotly_plot, prepare_instance):
    npt = prepare_instance
    fig = npt.bar_ngram(title='uni-gram', ngram=1, top_n=5, save=True)
    assert isinstance(fig, plotly.graph_objs.Figure)
    if not npt.ngram_df.empty : # Check if df is populated
         assert "word" in npt.ngram_df.columns
         assert "word_count" in npt.ngram_df.columns
    mock_plotly_plot.assert_called_once() # Check if save was attempted

@patch("plotly.offline.plot")
def test_nlplot_treemap(mock_plotly_plot, prepare_instance):
    npt = prepare_instance
    fig = npt.treemap(title='Tree Map', ngram=1, top_n=5, save=True)
    assert isinstance(fig, plotly.graph_objs.Figure)
    if not npt.treemap_df.empty:
        assert "word" in npt.treemap_df.columns
    mock_plotly_plot.assert_called_once()

@patch("plotly.offline.plot")
def test_nlplot_word_distribution(mock_plotly_plot, prepare_instance):
    npt = prepare_instance
    fig = npt.word_distribution(title='Word Dist', save=True)
    assert isinstance(fig, plotly.graph_objs.Figure)
    assert (npt.target_col + '_length') in npt.df.columns
    mock_plotly_plot.assert_called_once()

# For WordCloud, we need to mock IPython.display and PIL Image saving
@patch("IPython.display.display")
@patch("PIL.Image.Image.save") # Mock PIL save to prevent actual file writing
@patch("nlplot.nlplot.TTF_FILE_NAME", TTF_FONT_PATH) # Ensure it uses our test font path
def test_nlplot_wordcloud(mock_pil_save, mock_ipython_display, prepare_instance):
    npt = prepare_instance
    # Ensure the output directory exists if save=True
    os.makedirs(npt.output_file_path, exist_ok=True)
    try:
        npt.wordcloud(save=True) # Test with save=True
    except Exception as e:
        # WordCloud can be tricky with fonts, especially placeholder ones.
        # If it's a font issue, this test might need a real (small) TTF.
        if "cannot open resource" in str(e) or "not a TTF file" in str(e):
            pytest.skip(f"Skipping wordcloud test due to font issue: {e}")
        else:
            pytest.fail(f"wordcloud raised an exception: {e}")

    mock_ipython_display.assert_called_once()
    if TTF_FONT_PATH and os.path.exists(TTF_FONT_PATH): # Only assert save if font likely worked
        mock_pil_save.assert_called() # Check if Image.save was called (once for stream, once for file if save=True)
                                      # The mock above is for PIL.Image.Image.save, not wordcloud.to_file
                                      # The internal show_array calls img.save(stream) and img.save(filepath)

@patch("plotly.offline.iplot") # Mock iplot for co_network
@patch("plotly.offline.plot")  # Mock plot for save=True in co_network
def test_nlplot_co_network(mock_plotly_plot, mock_iplot, prepare_instance):
    npt = prepare_instance
    npt.build_graph(min_edge_frequency=0) # Build graph first
    assert hasattr(npt, 'G')
    if not npt.node_df.empty: # Only plot if graph is meaningful
        npt.co_network(title='Co-occurrence', save=True)
        mock_iplot.assert_called_once()
        mock_plotly_plot.assert_called_once() # For save=True
    else:
        # If node_df is empty, co_network should warn and not plot
        with patch('builtins.print') as mock_print:
            npt.co_network(title='Co-occurrence')
            assert any("Graph not built or empty" in call.args[0] for call in mock_print.call_args_list)


@patch("plotly.offline.plot")
def test_nlplot_sunburst(mock_plotly_plot, prepare_instance):
    npt = prepare_instance
    npt.build_graph(min_edge_frequency=0)
    if not npt.node_df.empty:
        fig = npt.sunburst(title='Sunburst', save=True, colorscale=True)
        assert isinstance(fig, plotly.graph_objs.Figure)
        mock_plotly_plot.assert_called_once()
    else:
        with patch('builtins.print') as mock_print:
            fig = npt.sunburst(title='Sunburst')
            assert any("Node DataFrame not available or empty" in call.args[0] for call in mock_print.call_args_list)
            assert isinstance(fig, plotly.graph_objs.Figure) # Should return empty fig

# Test saving tables
@patch("pandas.DataFrame.to_csv")
def test_save_tables(mock_to_csv, prepare_instance):
    npt = prepare_instance
    # Ensure output directory exists
    os.makedirs(npt.output_file_path, exist_ok=True)

    # Populate node_df and edge_df for testing save
    npt.node_df = pd.DataFrame({'id': ['node1'], 'id_code': [0]})
    npt.edge_df = pd.DataFrame({'source': ['node1'], 'target': ['node1'], 'edge_frequency': [1]})

    npt.save_tables(prefix="test_save")

    # Expected calls: one for node_df, one for edge_df.
    # Original df saving is commented out in nlplot.py, so 2 calls.
    assert mock_to_csv.call_count == 2

    # Check if filenames contain the prefix and date (date is harder to mock precisely here)
    # Example check for one of the calls:
    args_list = mock_to_csv.call_args_list
    assert any("test_save_node_df.csv" in call[0][0] for call in args_list)
    assert any("test_save_edge_df.csv" in call[0][0] for call in args_list)

# --- Tests for empty or problematic data ---
@patch("plotly.offline.plot")
def test_nlplot_bar_ngram_empty_df(mock_plotly_plot, prepare_instance_empty_df):
    npt = prepare_instance_empty_df
    with patch('builtins.print') as mock_print:
        fig = npt.bar_ngram(title='uni-gram', ngram=1, top_n=5)
        assert isinstance(fig, plotly.graph_objs.Figure) # Should return empty figure
        # generate_freq_df returns empty df, so ngram_df will be empty.
        # The plotting function should handle this by printing a warning.
        # Check for the specific warning if generate_freq_df itself doesn't print for empty input.
        # nlplot.py's bar_ngram has a check: if self.ngram_df.empty: print("Warning: No data to plot...")
        assert any("No data to plot for bar_ngram" in call.args[0] for call in mock_print.call_args_list)


def test_nlplot_build_graph_empty_df(prepare_instance_empty_df):
    npt = prepare_instance_empty_df
    with patch('builtins.print') as mock_print:
        npt.build_graph()
        # Expect warnings about no nodes found
        assert any("No nodes found" in call.args[0] for call in mock_print.call_args_list)
        assert npt.node_df.empty
        assert not hasattr(npt, 'edge_df') or npt.edge_df.empty # edge_df might not be created if node_df is empty


def test_nlplot_init_data_with_empty_text(data_with_empty_text):
    # Test that NLPlot can be initialized with texts that are empty or only spaces
    # and that these are handled correctly (e.g., result in empty lists of words)
    npt = NLPlot(data_with_empty_text, target_col="text")
    assert isinstance(npt.df["text"].iloc[0], list)
    assert npt.df["text"].iloc[0] == [] # "" splits to empty
    assert isinstance(npt.df["text"].iloc[1], list)
    assert npt.df["text"].iloc[1] == [] # "  " splits to empty (after strip perhaps, or filter in generate_ngrams)
    assert npt.df["text"].iloc[2] == ["Another", "sentence"]


# TODO:
# - Test verbose=False for generate_freq_df (check tqdm is not used)
# - Test more variations of stopwords in NLPlot methods
# - Test mask_file functionality in wordcloud more thoroughly (e.g. with a tiny actual mask image)
# - Test different layout functions for co_network
# - Test color_palette and colorscale options in co_network and sunburst
# - Test error handling for file operations (e.g. read-only directory for save)
# - Test TTF_FILE_NAME not found scenario (if WordCloud handles it gracefully or we should)
# - Test various data types within the list of words in target_col (e.g. numbers)
# - Test edge cases for min_edge_frequency in build_graph
# - Test that `stopwords` argument in plotting methods correctly combines with `self.default_stopwords`
# - Test `save_plot` sanitization of title_prefix.
# - Test `NLPlot` methods when `default_stopwords_file_path` is invalid or unreadable.

# Cleanup test output directory if created
def teardown_module(module):
    output_dir = "./test_outputs/"
    if os.path.exists(output_dir):
        import shutil
        # shutil.rmtree(output_dir) # Be careful with rmtree
        print(f"Test output directory {output_dir} can be manually removed.")

    # Clean up dummy font file
    if os.path.exists(TTF_FONT_PATH) and "placeholder" in TTF_FONT_PATH: # Basic check
        try:
            os.remove(TTF_FONT_PATH)
            print(f"Removed placeholder font file at {TTF_FONT_PATH}.")
            font_data_dir = os.path.dirname(TTF_FONT_PATH)
            if not os.listdir(font_data_dir): # If directory is empty
                 os.rmdir(font_data_dir)
                 print(f"Removed empty font data directory {font_data_dir}.")

        except OSError as e:
            print(f"Error removing placeholder font or directory: {e}")
