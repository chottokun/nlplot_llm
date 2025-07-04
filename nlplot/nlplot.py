"""Visualization Module for Natural Language Processing"""

import os
import gc
import itertools
import IPython.display
from io import BytesIO
from PIL import Image
from collections import defaultdict, Counter
import datetime as datetime

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
import seaborn as sns
import plotly
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import iplot
from wordcloud import WordCloud
import networkx as nx
from networkx.algorithms import community

# Default path for Japanese Font, can be overridden
DEFAULT_FONT_PATH = str(os.path.dirname(__file__)) + '/data/mplus-1c-regular.ttf'


def _ranked_topics_for_edges(batch_list: list) -> list:
    """Sorts a list of topics (words). Helper for edge generation."""
    return sorted(list(map(str, batch_list)))

def _unique_combinations_for_edges(batch_list: list) -> list:
    """Creates unique combinations of topics from a sorted list. Helper for edge generation."""
    return list(itertools.combinations(_ranked_topics_for_edges(batch_list), 2))

def _add_unique_combinations_to_dict(unique_combs: list, combo_dict: dict) -> dict:
    """Counts occurrences of combinations. Helper for edge generation."""
    for combination in unique_combs:
        combo_dict[combination] = combo_dict.get(combination, 0) + 1
    return combo_dict


def get_colorpalette(colorpalette: str, n_colors: int) -> list:
    """Get a color palette

    Args:
        colorpalette (str): cf.https://qiita.com/SaitoTsutomu/items/c79c9973a92e1e2c77a7 .
        n_colors (int): Number of colors to be displayed.

    Returns:
        list of str: e.g. ['rgb(220.16,95.0272,87.03)', 'rgb(220.16,209.13005714285714,87.03)', ...]
    """
    if not isinstance(n_colors, int) or n_colors <= 0:
        raise ValueError("n_colors must be a positive integer")
    palette = sns.color_palette(colorpalette, n_colors)
    rgb = ['rgb({},{},{})'.format(*[x*256 for x in rgb_val]) for rgb_val in palette]
    return rgb


def generate_freq_df(value: pd.Series, n_gram: int = 1, top_n: int = 50, stopwords: list = [],
                     verbose: bool = True) -> pd.DataFrame:
    """Generate a data frame of frequent word

    Args:
        value (pd.Series): Separated by space values.
        n_gram (int, optional): N number of N grams. Dafaults to 1.
        top_n (int, optional): N to get TOP N. Dafaults to 50.
        stopwords (list of str, optional): A list of words to specify for the stopword.
        verbose (bool, optional): Whether or not to output the log by tqdm.

    Returns:
        pd.DataFrame: frequent word DataFrame
    """
    if not isinstance(n_gram, int) or n_gram <= 0:
        raise ValueError("n_gram must be a positive integer")
    if not isinstance(top_n, int) or top_n < 0:
        raise ValueError("top_n must be a non-negative integer")

    def generate_ngrams(text: str, n_gram: int = 1) -> list:
        """Generate ngram

        Args:
            text (str): Target text
            n_gram (int, optional): N number of N grams. Defaults to 1.

        Returns:
            list of str: ngram list
        """
        token = [token for token in str(text).lower().split(" ")  # Ensure text is string
                 if token != "" if token not in stopwords]
        ngrams = zip(*[token[i:] for i in range(n_gram)])
        return [" ".join(ngram) for ngram in ngrams]

    freq_dict = defaultdict(int)
    iterable_value = tqdm(value) if verbose else value
    for sent in iterable_value:
        for word in generate_ngrams(sent, n_gram): # Pass sent directly
            freq_dict[word] += 1

    # frequent word DataFrame
    output_df = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    if not output_df.empty:
        output_df.columns = ['word', 'word_count']
    else:
        output_df = pd.DataFrame(columns=['word', 'word_count'])
    return output_df.head(top_n)


class NLPlot():
    """Visualization Module for Natural Language Processing

    Attributes:
        df (pd.DataFrame): Original dataframe to be graphed.
        target_col (str): Name of the column in `df` to be analyzed.
                          Assumes this column contains text data that will be tokenized (if not already a list of strings).
        output_file_path (str): Default directory path to save generated plots and tables.
        default_stopwords_file_path (str): Path to a file containing default stopwords, one per line.
        font_path (str): Default path to the font file (e.g., TTF) to be used for plots like word clouds.
                         If None, uses a bundled default font.

    """
    def __init__(
        self, df: pd.DataFrame,
        target_col: str,
        output_file_path: str = './',
        default_stopwords_file_path: str = '',
        font_path: str = None
    ):
        """
        Initializes the NLPlot object.

        Args:
            df (pd.DataFrame): The input DataFrame.
            target_col (str): The name of the column containing text to analyze.
            output_file_path (str, optional): Directory to save outputs. Defaults to './'.
            default_stopwords_file_path (str, optional): Path to a custom stopwords file. Defaults to ''.
            font_path (str, optional): Path to a .ttf font file for word clouds.
                                       If None, a default font is used. Defaults to None.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.df.dropna(subset=[self.target_col], inplace=True)
        if not self.df.empty and not pd.isna(self.df[self.target_col].iloc[0]) and \
           not isinstance(self.df[self.target_col].iloc[0], list):
            self.df.loc[:, self.target_col] = self.df[self.target_col].astype(str).map(lambda x: x.split())
        self.output_file_path = output_file_path
        self.font_path = font_path if font_path and os.path.exists(font_path) else DEFAULT_FONT_PATH
        if font_path and not os.path.exists(font_path): # User specified a font, but it wasn't found
            print(f"Warning: Specified font_path '{font_path}' not found. Falling back to default: {self.font_path}")

        if not os.path.exists(self.font_path): # Check if the determined font_path (either custom or default) exists
            print(f"Warning: The determined font path '{self.font_path}' does not exist. WordCloud may fail if a valid font is not provided at runtime or if the default font is missing.")

        self.default_stopwords = []
        if default_stopwords_file_path and os.path.exists(default_stopwords_file_path):
            try:
                with open(default_stopwords_file_path, 'r', encoding='utf-8') as f:
                    self.default_stopwords = [line.strip() for line in f if line.strip()]
            except PermissionError:
                print(f"Warning: Permission denied to read stopwords file '{default_stopwords_file_path}'. Continuing without these default stopwords.")
                self.default_stopwords = []
            except IOError as e:
                print(f"Warning: Could not read stopwords file '{default_stopwords_file_path}' due to an IO error: {e}. Continuing without these default stopwords.")
                self.default_stopwords = []
            except Exception as e:
                print(f"Warning: An unexpected error occurred while reading stopwords file '{default_stopwords_file_path}': {e}. Continuing without these default stopwords.")
                self.default_stopwords = []

    def get_stopword(self, top_n: int = 10, min_freq: int = 5) -> list:
        """Calculate the stop word.

        Calculate the top_n words with the highest number of occurrences
        and the words that occur only below the min_freq as stopwords.

        Args:
            top_n (int, optional): Top N of the number of occurrences of words to exclude. Defaults to 10.
            min_freq (int, optional): Bottom of the number of occurrences of words to exclude. Defaults to 5.

        Returns:
            list: list of stop words
        """
        if not isinstance(top_n, int) or top_n < 0:
            raise ValueError("top_n must be a non-negative integer.")
        if not isinstance(min_freq, int) or min_freq < 0:
            raise ValueError("min_freq must be a non-negative integer.")

        fdist = Counter()

        # Count the number of occurrences per word.
        for doc in self.df[self.target_col]:
            if isinstance(doc, list): #Ensure doc is a list of words
                for word in doc:
                    fdist[word] += 1
            # else: handle cases where doc might not be a list, or raise error
        # word with a high frequency
        common_words = {word for word, freq in fdist.most_common(top_n)}
        # word with a low frequency
        rare_words = {word for word, freq in fdist.items() if freq <= min_freq}
        stopwords = list(common_words.union(rare_words))
        # Add default stopwords, ensuring no duplicates
        stopwords.extend([sw for sw in self.default_stopwords if sw not in stopwords])
        return stopwords

    def bar_ngram(
        self,
        title: str = None,
        xaxis_label: str = '',
        yaxis_label: str = '',
        ngram: int = 1,
        top_n: int = 50,
        width: int = 800,
        height: int = 1100,
        color: str = None,
        horizon: bool = True,
        stopwords: list = [],
        verbose: bool = True,
        save: bool = False
    ) -> plotly.graph_objs.Figure:
        """Plots of n-gram bar chart
        # ... (docstringはそのまま)
        """
        # Combine provided stopwords with default stopwords, avoid duplicates
        current_stopwords = list(set(stopwords + self.default_stopwords))

        # Prepare data for n-gram generation (ensure it's a series of space-separated strings)
        # NLPlot constructor already converts target_col to list of words.
        # generate_freq_df expects a Series of space-separated strings.
        temp_series = self.df[self.target_col].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

        self.ngram_df = generate_freq_df(
            temp_series,
            n_gram=ngram,
            top_n=top_n,
            stopwords=current_stopwords,
            verbose=verbose
        )

        if self.ngram_df.empty:
             print("Warning: No data to plot for bar_ngram after processing. Empty DataFrame.")
             # Return an empty figure or handle as appropriate
             return go.Figure()


        if horizon:
            fig = px.bar(
                self.ngram_df.sort_values('word_count'),
                y='word',
                x='word_count',
                text='word_count',
                orientation='h',)
        else:
            fig = px.bar(
                self.ngram_df,
                y='word_count',
                x='word',
                text='word_count',)

        fig.update_traces(
            texttemplate='%{text:.2s}',
            textposition='auto',
            marker_color=color,)
        fig.update_layout(
            title=str(title) if title else 'N-gram Bar Chart',
            xaxis_title=str(xaxis_label),
            yaxis_title=str(yaxis_label),
            width=width,
            height=height,)

        if save:
            self.save_plot(fig, title if title else "bar_ngram")

        return fig

    def treemap(
        self,
        title: str = None,
        ngram: int = 1,
        top_n: int = 50,
        width: int = 1300,
        height: int = 600,
        stopwords: list = [],
        verbose: bool = True,
        save: bool = False
    ) -> plotly.graph_objs.Figure:
        """Plots of Tree Map
        # ... (docstringはそのまま)
        """
        current_stopwords = list(set(stopwords + self.default_stopwords))
        temp_series = self.df[self.target_col].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

        self.treemap_df = generate_freq_df(
            temp_series,
            n_gram=ngram,
            top_n=top_n,
            stopwords=current_stopwords,
            verbose=verbose
        )

        if self.treemap_df.empty or 'word' not in self.treemap_df.columns or 'word_count' not in self.treemap_df.columns:
            print("Warning: No data to plot for treemap after processing. Empty or malformed DataFrame.")
            return go.Figure()

        fig = px.treemap(
            self.treemap_df,
            path=[px.Constant("all"), 'word'], # Add a root node for better structure if needed
            values='word_count',
        )
        fig.update_layout(
            title=str(title) if title else 'Treemap',
            width=width,
            height=height,
        )

        if save:
            self.save_plot(fig, title if title else "treemap")

        return fig

    def word_distribution(
        self,
        title: str = None,
        xaxis_label: str = '',
        yaxis_label: str = '',
        width: int = 1000,
        height: int = 600,
        color: str = None,
        template: str = 'plotly',
        bins: int = None,
        save: bool = False
    ) -> plotly.graph_objs.Figure:
        """Plots of word count histogram
        # ... (docstringはそのまま)
        """
        col_name = self.target_col + '_length'
        # Ensure elements in target_col are lists to correctly calculate length
        self.df.loc[:, col_name] = self.df[self.target_col].apply(lambda x: len(x) if isinstance(x, list) else 0)

        if self.df[col_name].empty:
            print("Warning: No data to plot for word_distribution.")
            return go.Figure()

        fig = px.histogram(self.df, x=col_name, color=color, template=template, nbins=bins)
        fig.update_layout(
            title=str(title) if title else 'Word Distribution',
            xaxis_title=str(xaxis_label) if xaxis_label else 'Number of Words',
            yaxis_title=str(yaxis_label) if yaxis_label else 'Frequency',
            width=width,
            height=height,)

        if save:
            self.save_plot(fig, title if title else "word_distribution")

        return fig

    def wordcloud(
        self,
        width: int = 800,
        height: int = 500,
        max_words: int = 100,
        max_font_size: int = 80,
        stopwords: list = [],
        colormap: str = None,
        mask_file: str = None,
        font_path: str = None,
        save: bool = False
    ) -> None:
        """Plots of WordCloud

        Args:
            width (int, optional): Width of the graph. Defaults to 800.
            height (int, optional): Height of the graph. Defaults to 500.
            max_words (int, optional): Number of words to display. Defaults to 100.
            max_font_size (int, optional): Maximum font size for displayed words. Defaults to 80.
            stopwords (list, optional): A list of stopwords to exclude. Defaults to [].
            colormap (str, optional): Colormap for the word cloud.
                                      e.g., 'viridis', 'plasma', cf. matplotlib colormaps. Defaults to None.
            mask_file (str, optional): Path to an image file to use as a mask for the word cloud. Defaults to None.
            font_path (str, optional): Path to the font file (.ttf). Overrides the instance's default font_path.
                                       Defaults to None (uses instance default).
            save (bool, optional): Whether to save the generated image to a file. Defaults to False.

        Returns:
            None: Displays the word cloud in IPython and optionally saves it.

        Notes:
            - Font Handling:
                The method prioritizes fonts in this order:
                1. `font_path` argument of this method (if provided and valid).
                2. `self.font_path` (set during `NLPlot` initialization, can be custom or default).
                3. `DEFAULT_FONT_PATH` (the library's bundled default font).
                If a specified font file is not found, a warning is printed, and the next font in priority is attempted.
                If a font file is found but is invalid (e.g., corrupted, not a TTF), a warning is printed. If this was a custom
                font, a fallback to the default font (`DEFAULT_FONT_PATH`) is attempted. If the default font itself is invalid
                or if all fallbacks fail, an error message is printed, and the word cloud will not be generated.
            - Mask File:
                If `mask_file` is specified but not found, cannot be read (e.g., due to permissions or file corruption),
                or is not a valid image format, a warning is printed, and the word cloud is generated without a mask.
            - Text Processing:
                Input texts are converted to lowercase. Words are tokenized by spaces.
                If after applying stopwords, the resulting text corpus is empty, a warning is printed, and no
                word cloud is generated.
        """
        current_font_path = font_path if font_path and os.path.exists(font_path) else self.font_path
        if font_path and not os.path.exists(font_path): # User specified a font for this call, but it wasn't found
             print(f"Warning: Specified font_path '{font_path}' for wordcloud not found. Falling back to instance/default: {current_font_path}")

        # current_font_path is now the best candidate (method arg > instance default > library default constant path)
        # The try-except block below will handle if it's missing or invalid.

        mask = None
        if mask_file and os.path.exists(mask_file):
            try:
                mask = np.array(Image.open(mask_file))
            except PermissionError:
                print(f"Warning: Permission denied to read mask file {mask_file}. Proceeding without mask.")
                mask = None
            except IOError as e:
                print(f"Warning: Could not load mask file {mask_file} due to an IO error: {e}. Proceeding without mask.")
                mask = None
            except Exception as e: # Other PIL errors
                print(f"Warning: Could not load mask file {mask_file}: {e}. Proceeding without mask.")
                mask = None
        elif mask_file: # Path provided but os.path.exists was false
            print(f"Warning: Mask file {mask_file} not found. Proceeding without mask.")


        # Ensure elements in target_col are lists of strings, then join them
        # Filter out non-list elements or convert them appropriately
        processed_texts = []
        for item in self.df[self.target_col]:
            if isinstance(item, list):
                processed_texts.append(' '.join(map(str, item))) # Ensure all elements in list are strings
            elif isinstance(item, str):
                processed_texts.append(item)
            # else: skip or log warning for unexpected types

        if not processed_texts:
            print("Warning: No text data available for wordcloud after processing.")
            return

        text_corpus = ' '.join(processed_texts)
        current_stopwords = set(stopwords + self.default_stopwords) # Use set for efficient lookup

        if not text_corpus.strip():
             print("Warning: Text corpus is empty after processing stopwords for wordcloud.")
             return

        wordcloud_instance = WordCloud(
                        background_color='white',
                        font_step=1,
                        contour_width=0,
                        contour_color='steelblue',
                        font_path=current_font_path,
                        stopwords=current_stopwords,
                        max_words=max_words,
                        max_font_size=max_font_size,
                        random_state=42,
                        width=width,
                        height=height,
                        mask=mask,
                        collocations=False, # Default is True, but often set to False
                        prefer_horizontal=1,
                        colormap=colormap)
        try:
            # First attempt with current_font_path
            if not os.path.exists(current_font_path): # Check before attempting to use
                 raise OSError(f"Font file not found at {current_font_path}")

            wordcloud_instance = WordCloud(
                            background_color='white', font_step=1, contour_width=0, contour_color='steelblue',
                            font_path=current_font_path, stopwords=current_stopwords, max_words=max_words,
                            max_font_size=max_font_size, random_state=42, width=width, height=height,
                            mask=mask, collocations=False, prefer_horizontal=1, colormap=colormap)
            wordcloud_instance.generate(text_corpus)

        except (OSError, TypeError) as e: # Errors related to font file issues (not found, wrong type, corrupted)
            print(f"Warning: Error processing font at '{current_font_path}': {e}.")
            # Attempt to fallback to DEFAULT_FONT_PATH if current_font_path wasn't already it AND DEFAULT_FONT_PATH exists
            if current_font_path != DEFAULT_FONT_PATH and os.path.exists(DEFAULT_FONT_PATH):
                print(f"Attempting to fallback to default font: {DEFAULT_FONT_PATH}")
                try:
                    current_font_path = DEFAULT_FONT_PATH # Switch to default
                    wordcloud_instance = WordCloud(
                                    background_color='white', font_step=1, contour_width=0, contour_color='steelblue',
                                    font_path=current_font_path, stopwords=current_stopwords, max_words=max_words,
                                    max_font_size=max_font_size, random_state=42, width=width, height=height,
                                    mask=mask, collocations=False, prefer_horizontal=1, colormap=colormap)
                    wordcloud_instance.generate(text_corpus)
                except Exception as fallback_e:
                    print(f"Error: Fallback to default font ('{DEFAULT_FONT_PATH}') also failed: {fallback_e}. WordCloud cannot be generated.")
                    return
            elif current_font_path == DEFAULT_FONT_PATH: # The default font itself caused the error
                 print(f"Error: Default font at '{DEFAULT_FONT_PATH}' seems to be an issue. WordCloud cannot be generated. Details: {e}")
                 return
            else: # Default font path does not exist, and the custom one failed
                 print(f"Error: Default font not found at '{DEFAULT_FONT_PATH}' and custom font failed. WordCloud cannot be generated.")
                 return
        except ValueError as e: # Specifically for generate() if text corpus is empty after stopwords
            if "empty" in str(e).lower() or "zero" in str(e).lower():
                 print(f"Warning: WordCloud could not be generated. All words might have been filtered out or corpus is empty. Details: {e}")
                 return
            else:
                print(f"An unexpected ValueError occurred during WordCloud generation: {e}") # Other ValueErrors
                return # Or raise e if it should be fatal
        except Exception as e: # Catch-all for other unexpected errors
            print(f"An unexpected error occurred during WordCloud generation: {e}")
            return

        # If we reach here, wordcloud_instance should be valid and generated.
        img_array = wordcloud_instance.to_array()

        # Nested function for display and save
        def show_array(img_array_to_show, save_flag, output_path, filename_prefix_wc):
            stream = BytesIO()
            pil_img = Image.fromarray(img_array_to_show)
            if save_flag:
                date_str = pd.to_datetime(datetime.datetime.now()).strftime('%Y-%m-%d') # More standard date format
                filename = f"{date_str}_{filename_prefix_wc}_wordcloud.png"
                full_save_path = os.path.join(output_path, filename)
                try:
                    os.makedirs(output_path, exist_ok=True)
                    pil_img.save(full_save_path)
                    print(f"Wordcloud image saved to {full_save_path}")
                except PermissionError:
                    print(f"Error: Permission denied to save wordcloud image to '{full_save_path}'. Please check directory permissions.")
                except Exception as e_save:
                    print(f"Error saving wordcloud image to '{full_save_path}': {e_save}")

            pil_img.save(stream, 'png') # Save to stream for display
            IPython.display.display(IPython.display.Image(data=stream.getvalue()))

        show_array(img_array, save, self.output_file_path, "wordcloud_plot")
        return None


    def get_edges_nodes(self, batches: list, min_edge_frequency: int) -> None:
        """Generating the Edge and Node data frames for a graph
        # ... (docstringはそのまま)
        """
        if not isinstance(min_edge_frequency, int) or min_edge_frequency < 0:
             raise ValueError("min_edge_frequency must be a non-negative integer.")

        edge_dict = {}
        for batch in batches:
            if isinstance(batch, list) and batch: # Ensure batch is a non-empty list
                 unique_elements_in_batch = list(set(batch))
                 if len(unique_elements_in_batch) >= 2: # Combinations only if at least 2 unique elements
                    edge_dict = _add_unique_combinations_to_dict(
                        _unique_combinations_for_edges(unique_elements_in_batch),
                        edge_dict
                    )

        source, target, edge_frequency_list = [], [], []
        for key, value in edge_dict.items():
            source.append(key[0])
            target.append(key[1])
            edge_frequency_list.append(value)

        edge_df = pd.DataFrame({'source': source, 'target': target, 'edge_frequency': edge_frequency_list})
        edge_df = edge_df[edge_df['edge_frequency'] > min_edge_frequency].sort_values(
            by='edge_frequency', ascending=False
        ).reset_index(drop=True)

        if edge_df.empty:
            self.edge_df = pd.DataFrame(columns=['source', 'target', 'edge_frequency', 'source_code', 'target_code'])
            self.node_df = pd.DataFrame(columns=['id', 'id_code']) # Ensure columns exist even if empty
            self.node_dict = {}
            self.edge_dict = edge_dict
            return

        unique_nodes = list(set(edge_df['source']).union(set(edge_df['target'])))
        node_df = pd.DataFrame({'id': unique_nodes})
        if not node_df.empty:
            node_df['id_code'] = node_df.index
            node_dict = dict(zip(node_df['id'], node_df['id_code']))
            edge_df['source_code'] = edge_df['source'].map(node_dict)
            edge_df['target_code'] = edge_df['target'].map(node_dict)
            edge_df.dropna(subset=['source_code', 'target_code'], inplace=True)
        else: # Should not happen if edge_df is not empty, but as a safeguard
            node_dict = {}
            # edge_df would be effectively empty of valid codes, or this path means unique_nodes was empty
            # which contradicts edge_df not being empty. This state indicates an issue.
            # For safety, clear edge_df if node mapping fails completely.
            edge_df = pd.DataFrame(columns=['source', 'target', 'edge_frequency', 'source_code', 'target_code'])


        self.edge_df = edge_df
        self.node_df = node_df
        self.node_dict = node_dict
        self.edge_dict = edge_dict

        return None

    def get_graph(self) -> nx.Graph:
        """create Networkx
        # ... (docstringはそのまま)
        """
        G = nx.Graph()
        if not hasattr(self, 'node_df') or self.node_df.empty:
            print("Warning: Node DataFrame is not initialized or empty. Cannot build graph.")
            return G # Return empty graph

        G.add_nodes_from(self.node_df.id_code)

        if not hasattr(self, 'edge_df') or self.edge_df.empty:
            # print("Warning: Edge DataFrame is empty. Graph will have nodes but no edges.")
            return G # Nodes added, but no edges if edge_df is empty

        edge_tuples = []
        for i in range(len(self.edge_df)):
            # Ensure source_code and target_code are integers if they are node IDs
            # and exist in G.nodes. G.add_edge will handle this.
            source_node = self.edge_df['source_code'].iloc[i]
            target_node = self.edge_df['target_code'].iloc[i]
            edge_tuples.append((source_node, target_node))

        G.add_edges_from(edge_tuples)
        return G

    def build_graph(self, stopwords: list = [], min_edge_frequency: int = 10) -> None:
        """Preprocessing to output a co-occurrence network."""
        self._prepare_data_for_graph(stopwords)
        self.get_edges_nodes(self._batches, min_edge_frequency) # Use self._batches

        if self.node_df.empty:
            self._initialize_empty_graph_attributes()
            print('Warning: No nodes found after processing for build_graph. Co-occurrence network cannot be built.')
            print('node_size:0, edge_size:0')
            return

        self.G = self.get_graph()
        if not self.G.nodes():
            self._initialize_empty_graph_attributes(graph_exists_but_no_nodes=True)
            print('Warning: Graph has no nodes. Further calculations for co-occurrence network will be skipped.')
            print(f'node_size:{len(self.node_df)}, edge_size:{len(self.edge_df if hasattr(self, "edge_df") else [])}')
            return

        self._calculate_graph_metrics()
        self._detect_communities()

        print(f'node_size:{len(self.node_df)}, edge_size:{len(self.edge_df if hasattr(self, "edge_df") else [])}')
        return None

    def _prepare_data_for_graph(self, stopwords_param: list):
        """Helper to prepare data (self.df_edit, self._batches) for graph building."""
        current_stopwords = list(set(stopwords_param + self.default_stopwords))
        self.df_edit = self.df.copy()
        self.df_edit.loc[:, self.target_col] = self.df_edit[self.target_col].apply(
            lambda doc: list(set(w for w in doc if w not in current_stopwords)) if isinstance(doc, list) else []
        )
        self._batches = self.df_edit[self.target_col].tolist()

    def _initialize_empty_graph_attributes(self, graph_exists_but_no_nodes=False):
        """Helper to initialize graph attributes when the graph is empty or cannot be built."""
        self.G = nx.Graph()
        self.adjacencies = {}
        self.betweeness = {}
        self.clustering_coeff = {}
        self.communities = []
        self.communities_dict = {}
        if not graph_exists_but_no_nodes and hasattr(self, 'node_df') and not self.node_df.empty :
             # If node_df was supposed to be empty but isn't, this is an inconsistent state.
             # However, if called because graph has no nodes but node_df might exist (e.g. all isolated),
             # we might still want to assign empty community info.
             # For now, if node_df exists, ensure 'community' column is present.
             self.node_df['community'] = -1
        # If node_df is truly empty, it's handled by the caller (build_graph)

    def _calculate_graph_metrics(self):
        """Calculates and assigns graph metrics (adjacency, betweenness, clustering) to node_df."""
        if not hasattr(self, 'G') or not self.G.nodes():
            print("Warning: Graph not available for metric calculation.")
            return

        self.adjacencies = dict(self.G.adjacency())
        self.betweeness = nx.betweenness_centrality(self.G)
        self.clustering_coeff = nx.clustering(self.G)

        self.node_df['adjacency_frequency'] = self.node_df['id_code'].map(lambda x: len(self.adjacencies.get(x, {})))
        self.node_df['betweeness_centrality'] = self.node_df['id_code'].map(lambda x: self.betweeness.get(x, 0.0))
        self.node_df['clustering_coefficient'] = self.node_df['id_code'].map(lambda x: self.clustering_coeff.get(x, 0.0))

    def _detect_communities(self):
        """Detects communities and assigns them to node_df."""
        if not hasattr(self, 'G') or not self.G.nodes() or self.node_df.empty:
            print("Warning: Graph or node_df not available for community detection.")
            self.communities = []
            self.communities_dict = {}
            if hasattr(self, 'node_df') and not self.node_df.empty:
                 self.node_df['community'] = -1
            return

        # greedy_modularity_communities can return an empty list or list of frozensets
        raw_communities = community.greedy_modularity_communities(self.G)
        self.communities = [list(comm) for comm in raw_communities if comm] # Ensure list of lists, remove empty sets

        self.communities_dict = {i: comm_nodes for i, comm_nodes in enumerate(self.communities)}

        def community_allocation(id_code):
            for k, v_list in self.communities_dict.items():
                if id_code in v_list:
                    return k
            return -1
        self.node_df['community'] = self.node_df['id_code'].map(community_allocation)


    def _create_network_trace(self, trace_type: str, **kwargs) -> go.Scatter:
        """Helper to create a single trace for the network graph."""
        if trace_type == "edge":
            return go.Scatter(
                x=kwargs['x'], y=kwargs['y'], mode='lines',
                line={'width': kwargs['width'], 'color': kwargs['color']},
                line_shape='spline', opacity=kwargs['opacity']
            )
        elif trace_type == "node":
            return go.Scatter(
                x=kwargs['x'], y=kwargs['y'], text=kwargs['text'],
                mode='markers+text', textposition='bottom center',
                hoverinfo="text", marker=kwargs['marker']
            )
        raise ValueError(f"Unknown trace_type: {trace_type}")

    def co_network(self, title:str = None, sizing:int=100, node_size_col:str='adjacency_frequency',
                   color_palette:str='hls', layout_func=nx.kamada_kawai_layout, #renamed layout to layout_func
                   light_theme:bool=True, width:int=1700, height:int=1200, save:bool=False) -> None:
        """Plots of co-occurrence networks
        # ... (docstringはそのまま)
        """
        if not hasattr(self, 'G') or not self.G.nodes():
            print("Warning: Graph not built or empty. Cannot plot co-occurrence network.")
            return
        if not hasattr(self, 'node_df') or self.node_df.empty:
            print("Warning: Node DataFrame not available or empty. Cannot plot co-occurrence network.")
            return
        if node_size_col not in self.node_df.columns:
            print(f"Warning: node_size column '{node_size_col}' not found in node_df. Using 'adjacency_frequency'.")
            node_size_col = 'adjacency_frequency'
            if node_size_col not in self.node_df.columns: # Still not there
                 print(f"Warning: Default node_size column 'adjacency_frequency' also not found. Node sizes will be uniform.")
                 # Create a dummy column for uniform size if necessary, or handle in marker size calculation
                 self.node_df['uniform_size'] = 10 # Example uniform size value
                 node_size_col = 'uniform_size'


        back_col, edge_col = ('#ffffff', '#ece8e8') if light_theme else ('#000000', '#2d2b2b')

        final_node_sizes = self._calculate_node_sizes(node_size_col, sizing)

        pos = layout_func(self.G)
        for node_id_code in self.G.nodes(): # G.nodes() are id_codes
            self.G.nodes[node_id_code]['pos'] = list(pos[node_id_code])

        edge_traces = []
        for edge_nodes in self.G.edges(): # edge_nodes are (id_code_1, id_code_2)
            x0, y0 = self.G.nodes[edge_nodes[0]]['pos']
            x1, y1 = self.G.nodes[edge_nodes[1]]['pos']
            edge_traces.append(self._create_network_trace(
                trace_type="edge", x=[x0, x1, None], y=[y0, y1, None],
                width=1.2, color=edge_col, opacity=1
            ))

        node_x, node_y, node_hover_text, node_marker_colors, node_marker_sizes = [], [], [], [], []

        # Ensure 'community' column exists and is numeric for coloring
        if 'community' not in self.node_df.columns or not pd.api.types.is_numeric_dtype(self.node_df['community']):
            self.node_df['community_display'] = 0 # Default community for coloring if missing/invalid
        else:
            self.node_df['community_display'] = self.node_df['community']

        num_communities = self.node_df['community_display'].nunique()
        palette_colors = get_colorpalette(color_palette, num_communities if num_communities > 0 else 1)


        # node_df is indexed by default pd index, 'id_code' is a column.
        # We need to map G.nodes (which are id_codes) to rows in node_df.
        # A common way is to set 'id_code' as index for quick lookup if node_df is large,
        # or iterate and filter if it's small.
        # For simplicity, assuming node_df['id_code'] contains unique values corresponding to G.nodes()

        # Pre-calculate mapping from id_code to its actual display name ('id') and community
        id_code_to_info = self.node_df.set_index('id_code')

        for id_code_node in self.G.nodes(): # id_code_node is an id_code from G
            x, y = self.G.nodes[id_code_node]['pos']
            node_x.append(x)
            node_y.append(y)

            node_specific_info = id_code_to_info.loc[id_code_node]
            node_hover_text.append(node_specific_info['id']) # Original node name for hover/text

            community_val = int(node_specific_info['community_display'])
            node_marker_colors.append(palette_colors[community_val % len(palette_colors)])

            # final_node_sizes should align with node_df's original index if it was used for scaling
            # If final_node_sizes was created from node_df[node_size_col].values, it aligns with default 0..N-1 index
            # We need the size for *this* specific id_code_node.
            # Assuming final_node_sizes is a pd.Series with the same index as self.node_df before set_index:
            node_marker_sizes.append(final_node_sizes.loc[node_specific_info.name]) # .name is the original index (if id_code was not index)
                                                                                  # If id_code was already index, then final_node_sizes should also be indexed by id_code.
                                                                                  # Let's assume final_node_sizes is a Series aligned with self.node_df's original index.
                                                                                  # The node_specific_info.name will give that original index.

        node_trace = self._create_network_trace(
            trace_type="node", x=node_x, y=node_y, text=node_hover_text,
            marker={'size': node_marker_sizes, 'line': dict(width=0.5, color=edge_col), 'color': node_marker_colors}
        )
        fig_data = edge_traces + [node_trace]
        fig_layout = go.Layout(
            title=str(title) if title else "Co-occurrence Network",
            font=dict(family='Arial', size=12), width=width, height=height, autosize=True,
            showlegend=False, xaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
            margin=dict(l=40, r=40, b=85, t=100, pad=0), hovermode='closest', plot_bgcolor=back_col
        )
        fig = go.Figure(data=fig_data, layout=fig_layout)
        iplot(fig)

        if save:
            self.save_plot(fig, title if title else "co_network")

        gc.collect()
        return None

    def _calculate_node_sizes(self, node_size_col: str, sizing_factor: int) -> pd.Series:
        """Helper to calculate scaled node sizes for plotting."""
        if node_size_col not in self.node_df.columns or self.node_df[node_size_col].isnull().all():
            print(f"Warning: Node size column '{node_size_col}' not found or all nulls. Using uniform small size.")
            return pd.Series([sizing_factor * 0.1] * len(self.node_df), index=self.node_df.index)

        node_sizes_numeric = pd.to_numeric(self.node_df[node_size_col], errors='coerce').fillna(0)

        if len(node_sizes_numeric) == 0: # Should not happen if node_df is not empty
            return pd.Series(index=self.node_df.index, dtype=float)

        # Handle cases where all values are the same or all are zero to avoid scaler issues
        if node_sizes_numeric.nunique() <= 1: # Includes all same, or all zero
            if node_sizes_numeric.iloc[0] == 0 : # All zeros
                 return pd.Series([sizing_factor * 0.1] * len(self.node_df), index=self.node_df.index)
            else: # All same non-zero value
                 return pd.Series([sizing_factor * 0.5] * len(self.node_df), index=self.node_df.index)

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 1.0)) # Avoid zero size
        scaled_values = min_max_scaler.fit_transform(node_sizes_numeric.values.reshape(-1, 1)).flatten()
        final_sizes = pd.Series(scaled_values, index=self.node_df.index) * sizing_factor
        return final_sizes

    def sunburst(self, title:str=None, colorscale:bool=False, color_col:str='betweeness_centrality',
                 color_continuous_scale:str='Oryel', width:int=1100, height:int=1100, save:bool=False) -> plotly.graph_objs.Figure:
        """Plots of sunburst chart
        # ... (docstringはそのまま)
        """
        if not hasattr(self, 'node_df') or self.node_df.empty:
            print("Warning: Node DataFrame not available or empty. Cannot plot sunburst chart.")
            return go.Figure() # Return empty figure

        _df = self.node_df.copy()

        if 'community' not in _df.columns:
            _df['community'] = '0' # Default community if missing
        else:
            _df['community'] = _df['community'].astype(str)

        if 'id' not in _df.columns: # Should always exist
             _df['id'] = "Unknown"

        # Ensure values for path and values are not empty and valid
        if 'adjacency_frequency' not in _df.columns or _df['adjacency_frequency'].isnull().all():
            print("Warning: 'adjacency_frequency' column is missing or all nulls. Sunburst may be empty or error.")
            _df['adjacency_frequency'] = 1 # Dummy value to prevent error if missing/all null

        path_cols = ['community', 'id']

        try:
            if colorscale:
                if color_col not in _df.columns or _df[color_col].isnull().all():
                    print(f"Warning: color_col '{color_col}' for sunburst is missing or all nulls. Using default coloring.")
                    fig = px.sunburst(_df, path=path_cols, values='adjacency_frequency', color='community')
                else:
                    # Ensure color_col is numeric for continuous scale
                    _df[color_col] = pd.to_numeric(_df[color_col], errors='coerce').fillna(0)
                    fig = px.sunburst(_df, path=path_cols, values='adjacency_frequency',
                                      color=color_col, hover_data=None, # Consider adding some hover_data
                                      color_continuous_scale=color_continuous_scale,
                                      color_continuous_midpoint=np.average(
                                          _df[color_col].fillna(0), weights=_df['adjacency_frequency'].fillna(1) # Handle potential NaNs
                                      ))
            else:
                fig = px.sunburst(_df, path=path_cols, values='adjacency_frequency', color='community')
        except Exception as e:
            print(f"Error creating sunburst chart: {e}. Returning empty figure.")
            return go.Figure()


        fig.update_layout(
            title=str(title) if title else 'Sunburst Chart',
            width=width,
            height=height,
        )

        if save:
            self.save_plot(fig, title if title else "sunburst_chart")

        del _df
        gc.collect()
        return fig

    def save_plot(self, fig, title_prefix: str) -> None: # Renamed title to title_prefix
        """Save the HTML file
        Args:
            fig (plotly.graph_objs.Figure): The Plotly figure object to save.
            title_prefix (str): A prefix for the filename. The final filename will be
                                `YYYY-MM-DD_title_prefix.html`. Special characters in
                                `title_prefix` will be replaced with underscores.
        Returns:
            None

        Notes:
            If `output_file_path` (set during `NLPlot` initialization) is not writable,
            a `PermissionError` will be caught, and an error message printed.
        """
        if not title_prefix or not isinstance(title_prefix, str):
            title_prefix = "plot"
        title_prefix = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in title_prefix)

        date_str = pd.to_datetime(datetime.datetime.now()).strftime('%Y-%m-%d')
        filename = f"{date_str}_{title_prefix}.html"
        full_path = os.path.join(self.output_file_path, filename)
        try:
            os.makedirs(self.output_file_path, exist_ok=True)
            plotly.offline.plot(fig, filename=full_path, auto_open=False)
            print(f"Plot saved to {full_path}")
            except PermissionError:
            print(f"Error: Permission denied to write plot to '{full_path}'. Please check directory permissions.")
        except Exception as e:
            print(f"Error saving plot to '{full_path}': {e}")
        return None

    def save_tables(self, prefix: str = "nlplot_output") -> None:
        """
        Saves the generated node and edge DataFrames to CSV files.

        The DataFrames `self.node_df` and `self.edge_df` (typically generated by
        `build_graph`) are saved.

        Args:
            prefix (str, optional): A prefix for the filenames. Files will be named
                                    `YYYY-MM-DD_prefix_node_df.csv` and
                                    `YYYY-MM-DD_prefix_edge_df.csv`.
                                    Defaults to "nlplot_output".
        Returns:
            None

        Notes:
            If `output_file_path` (set during `NLPlot` initialization) is not writable,
            a `PermissionError` will be caught, and an error message printed.
            If `node_df` or `edge_df` are not available or empty, they will not be saved,
            and a message will be printed.
        """
        if not hasattr(self, 'node_df') or not hasattr(self, 'edge_df'): # Check if attributes exist
            print("Warning: node_df or edge_df attributes not found. Ensure build_graph() has been called. Cannot save tables.")
            return

        date_str = pd.to_datetime(datetime.datetime.now()).strftime('%Y-%m-%d')
        sanitized_prefix = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in prefix)

        try:
            os.makedirs(self.output_file_path, exist_ok=True)

            if hasattr(self, 'node_df') and isinstance(self.node_df, pd.DataFrame) and not self.node_df.empty:
                node_filename = os.path.join(self.output_file_path, f"{date_str}_{sanitized_prefix}_node_df.csv")
                self.node_df.to_csv(node_filename, index=False)
                print(f'Saved nodes to {node_filename}')
            else:
                print('Node DataFrame is empty or not available. Not saved.')

            if hasattr(self, 'edge_df') and isinstance(self.edge_df, pd.DataFrame) and not self.edge_df.empty:
                edge_filename = os.path.join(self.output_file_path, f"{date_str}_{sanitized_prefix}_edge_df.csv")
                self.edge_df.to_csv(edge_filename, index=False)
                print(f'Saved edges to {edge_filename}')
            else:
                print('Edge DataFrame is empty or not available. Not saved.')

        except PermissionError:
            print(f"Error: Permission denied to write tables in '{self.output_file_path}'. Please check directory permissions.")
        except Exception as e:
            print(f"Error saving tables: {e}")
        return None
