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
from typing import Optional, List, Any
import asyncio # Added for asynchronous operations

# Default path for Japanese Font, can be overridden
DEFAULT_FONT_PATH = None

try:
    from janome.tokenizer import Tokenizer as JanomeTokenizer
    JANOME_AVAILABLE = True
except ImportError:
    JANOME_AVAILABLE = False
    class JanomeTokenizer: # type: ignore
        def tokenize(self, text: str, stream=False, wakati=False): # type: ignore
            return []
    # class JanomeToken: pass

# LiteLLM and Langchain TextSplitters
try:
    import litellm
    from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
    LITELLM_AVAILABLE = True
    LANGCHAIN_SPLITTERS_AVAILABLE = True
    import diskcache
    DISKCACHE_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    LANGCHAIN_SPLITTERS_AVAILABLE = False
    DISKCACHE_AVAILABLE = False
    # Dummy class for litellm if not installed, to prevent NameError on checks
    class litellm_dummy: # type: ignore
        def completion(self, *args, **kwargs): raise ImportError("litellm is not installed.")
        class exceptions: # type: ignore
            class APIConnectionError(Exception): pass
            class AuthenticationError(Exception): pass
            class RateLimitError(Exception): pass
            # Add other exceptions as needed for robust error handling
    litellm = litellm_dummy() # type: ignore

    # Dummy splitters if Langchain splitters are not available
    class RecursiveCharacterTextSplitter: # type: ignore
        def __init__(self, chunk_size=None, chunk_overlap=None, length_function=None, **kwargs): pass
        def split_text(self, text: str) -> List[str]: return [text] if text else []
    class CharacterTextSplitter: # type: ignore
        def __init__(self, separator=None, chunk_size=None, chunk_overlap=None, length_function=None, **kwargs): pass
        def split_text(self, text: str) -> List[str]: return [text] if text else []


def _ranked_topics_for_edges(batch_list: list) -> list:
    return sorted(list(map(str, batch_list)))

def _unique_combinations_for_edges(batch_list: list) -> list:
    return list(itertools.combinations(_ranked_topics_for_edges(batch_list), 2))

def _add_unique_combinations_to_dict(unique_combs: list, combo_dict: dict) -> dict:
    for combination in unique_combs:
        combo_dict[combination] = combo_dict.get(combination, 0) + 1
    return combo_dict

def get_colorpalette(colorpalette: str, n_colors: int) -> list:
    if not isinstance(n_colors, int) or n_colors <= 0:
        raise ValueError("n_colors must be a positive integer")
    palette = sns.color_palette(colorpalette, n_colors)
    return ['rgb({},{},{})'.format(*[x*256 for x in rgb_val]) for rgb_val in palette]

def generate_freq_df(value: pd.Series, n_gram: int = 1, top_n: int = 50, stopwords: list = [],
                     verbose: bool = True) -> pd.DataFrame:
    if not isinstance(n_gram, int) or n_gram <= 0:
        raise ValueError("n_gram must be a positive integer")
    if not isinstance(top_n, int) or top_n < 0:
        raise ValueError("top_n must be a non-negative integer")
    def generate_ngrams(text: str, n_gram_val: int = 1) -> list: # Renamed n_gram to n_gram_val
        token = [t for t in str(text).lower().split(" ") if t != "" and t not in stopwords]
        ngrams_zip = zip(*[token[i:] for i in range(n_gram_val)])
        return [" ".join(ngram) for ngram in ngrams_zip]
    freq_dict = defaultdict(int)
    iterable_value = tqdm(value) if verbose else value
    for sent in iterable_value:
        for word in generate_ngrams(sent, n_gram): freq_dict[word] += 1
    output_df = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    if not output_df.empty: output_df.columns = ['word', 'word_count']
    else: output_df = pd.DataFrame(columns=['word', 'word_count'])
    return output_df.head(top_n)

class NLPlotLLM():
    def __init__( self, df: pd.DataFrame, target_col: str, output_file_path: str = './',
                  default_stopwords_file_path: str = '', font_path: str = None,
                  use_cache: bool = True,
                  cache_dir: Optional[str] = None,
                  cache_expire: Optional[int] = None, # seconds, None for no expiration
                  cache_size_limit: Optional[int] = int(1e9) # bytes, 1GB default
                ):
        """
        Initializes the NLPlotLLM instance.

        Args:
            df (pd.DataFrame): DataFrame containing the text data.
            target_col (str): Name of the column in `df` that contains the text to analyze.
                              The contents of this column are typically tokenized into a list of words
                              during initialization if they are not already lists.
            output_file_path (str, optional): Default path to save generated plots and tables.
                                              Defaults to './'.
            default_stopwords_file_path (str, optional): Path to a custom file containing default
                                                         stopwords (one per line). Defaults to ''.
            font_path (str, optional): Path to a .ttf font file to be used for word clouds.
                                       If None, WordCloud attempts to use system default fonts.
                                       Defaults to None.
            use_cache (bool, optional): Whether to enable caching for LLM responses by default.
                                        Can be overridden per LLM method call. Defaults to True.
            cache_dir (Optional[str], optional): Directory to store cache files.
                                                 If None, defaults to a platform-specific user cache directory
                                                 (e.g., ~/.cache/nlplot_llm). Defaults to None.
            cache_expire (Optional[int], optional): Time in seconds after which cached items expire.
                                                    If None, items do not expire. Defaults to None.
            cache_size_limit (Optional[int], optional): Maximum size of the cache in bytes.
                                                        Defaults to 1GB (1e9 bytes).
        """
        self.df = df.copy()
        self.target_col = target_col
        self.df.dropna(subset=[self.target_col], inplace=True)
        if not self.df.empty and not pd.isna(self.df[self.target_col].iloc[0]) and \
           not isinstance(self.df[self.target_col].iloc[0], list):
            self.df.loc[:, self.target_col] = self.df[self.target_col].astype(str).map(lambda x: x.split())
        self.output_file_path = output_file_path
        # Determine initial font_path based on user input and DEFAULT_FONT_PATH
        if font_path and os.path.exists(font_path):
            self.font_path = font_path
        else:
            if font_path: # User specified a font_path but it wasn't found
                print(f"Warning: Specified font_path '{font_path}' not found. Falling back to default.")
            self.font_path = DEFAULT_FONT_PATH

        # Now, self.font_path is either a valid path from user, DEFAULT_FONT_PATH, or None (if DEFAULT_FONT_PATH is None)
        # Only check for existence if self.font_path is not None
        if self.font_path and not os.path.exists(self.font_path):
            print(f"Warning: The determined font path '{self.font_path}' does not exist. WordCloud may fail if a valid font is not provided at runtime or if the default font is missing.")
        elif not self.font_path: # self.font_path is None
             print(f"Info: No font path provided or determined (self.font_path is None). WordCloud will use its default system font.")

        self.default_stopwords = []
        if default_stopwords_file_path and os.path.exists(default_stopwords_file_path):
            try:
                with open(default_stopwords_file_path, 'r', encoding='utf-8') as f:
                    self.default_stopwords = [line.strip() for line in f if line.strip()]
            except PermissionError: print(f"Warning: Permission denied to read stopwords file '{default_stopwords_file_path}'. Continuing without these default stopwords.")
            except IOError as e: print(f"Warning: Could not read stopwords file '{default_stopwords_file_path}' due to an IO error: {e}. Continuing without these default stopwords.")
            except Exception as e: print(f"Warning: An unexpected error occurred while reading stopwords file '{default_stopwords_file_path}': {e}. Continuing without these default stopwords.")

        self._janome_tokenizer = None
        if JANOME_AVAILABLE:
            try: self._janome_tokenizer = JanomeTokenizer()
            except Exception as e: print(f"Warning: Failed to initialize Janome Tokenizer. Japanese text features may not be available. Error: {e}")

        # Cache initialization
        self.use_cache_default = use_cache
        self.cache = None
        if self.use_cache_default and DISKCACHE_AVAILABLE:
            cache_path = cache_dir if cache_dir else os.path.join(os.path.expanduser("~"), ".cache", "nlplot_llm")
            try:
                self.cache = diskcache.Cache(cache_path, size_limit=cache_size_limit, expire=cache_expire)
                print(f"NLPlotLLM cache initialized at: {cache_path}")
            except Exception as e:
                print(f"Warning: Failed to initialize diskcache at {cache_path}. Cache will be disabled. Error: {e}")
                self.cache = None
                self.use_cache_default = False # Disable cache if init fails
        elif self.use_cache_default and not DISKCACHE_AVAILABLE:
            print("Warning: diskcache library is not installed, but use_cache=True. Caching will be disabled. Please install diskcache.")
            self.use_cache_default = False

    def _tokenize_japanese_text(self, text: str) -> list:
        if not JANOME_AVAILABLE: print("Warning: Janome is not installed. Japanese tokenization is not available. Please install Janome (e.g., pip install janome)."); return []
        if self._janome_tokenizer is None: print("Warning: Janome Tokenizer is not available (it may have failed to initialize). Japanese tokenization cannot be performed."); return []
        if not isinstance(text, str) or not text.strip(): return []
        try: return list(self._janome_tokenizer.tokenize(text))
        except Exception as e: print(f"Error during Janome tokenization for text '{text[:30]}...': {e}"); return []

    def get_japanese_text_features(self, japanese_text_series: pd.Series) -> pd.DataFrame:
        results = []
        expected_columns = ['text', 'total_tokens', 'avg_token_length', 'noun_ratio', 'verb_ratio', 'adj_ratio', 'punctuation_count']
        if not isinstance(japanese_text_series, pd.Series): print("Warning: Input must be a pandas Series. Returning empty DataFrame."); return pd.DataFrame(columns=expected_columns)
        if japanese_text_series.empty: return pd.DataFrame(columns=expected_columns)
        for text_input in japanese_text_series:
            original_text = str(text_input) if pd.notna(text_input) else ""
            tokens = self._tokenize_japanese_text(text_input) if pd.notna(text_input) and isinstance(text_input, str) else []
            total_tokens = len(tokens)
            if total_tokens == 0:
                results.append({'text': original_text, 'total_tokens': 0, 'avg_token_length': 0.0, 'noun_ratio': 0.0, 'verb_ratio': 0.0, 'adj_ratio': 0.0, 'punctuation_count': 0})
                continue
            non_punctuation_tokens = [t for t in tokens if not (t.part_of_speech.startswith('記号,句点') or t.part_of_speech.startswith('記号,読点'))]
            num_non_punctuation_tokens = len(non_punctuation_tokens)
            avg_token_length = sum(len(t.surface) for t in non_punctuation_tokens) / num_non_punctuation_tokens if num_non_punctuation_tokens > 0 else 0.0
            nouns = [t for t in tokens if t.part_of_speech.startswith('名詞')]
            verbs = [t for t in tokens if t.part_of_speech.startswith('動詞,自立')]
            adjectives = [t for t in tokens if t.part_of_speech.startswith('形容詞,自立')]
            punctuations_list = [t for t in tokens if t.part_of_speech.startswith('記号,句点') or t.part_of_speech.startswith('記号,読点')]
            results.append({'text': original_text, 'total_tokens': total_tokens, 'avg_token_length': avg_token_length,
                            'noun_ratio': len(nouns) / total_tokens, 'verb_ratio': len(verbs) / total_tokens,
                            'adj_ratio': len(adjectives) / total_tokens, 'punctuation_count': len(punctuations_list)})
        return pd.DataFrame(results, columns=expected_columns)

    def get_stopword(self, top_n: int = 10, min_freq: int = 5) -> list:
        if not isinstance(top_n, int) or top_n < 0: raise ValueError("top_n must be a non-negative integer.")
        if not isinstance(min_freq, int) or min_freq < 0: raise ValueError("min_freq must be a non-negative integer.")
        fdist = Counter()
        for doc in self.df[self.target_col]:
            if isinstance(doc, list):
                for word in doc: fdist[word] += 1
        common_words = {word for word, freq in fdist.most_common(top_n)}
        rare_words = {word for word, freq in fdist.items() if freq <= min_freq}
        stopwords = list(common_words.union(rare_words))
        stopwords.extend([sw for sw in self.default_stopwords if sw not in stopwords])
        return stopwords

    def bar_ngram(self, title: str = None, xaxis_label: str = '', yaxis_label: str = '', ngram: int = 1, top_n: int = 50, width: int = 800, height: int = 1100, color: str = None, horizon: bool = True, stopwords: list = [], verbose: bool = True, save: bool = False) -> plotly.graph_objs.Figure:
        current_stopwords = list(set(stopwords + self.default_stopwords))
        temp_series = self.df[self.target_col].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
        self.ngram_df = generate_freq_df(temp_series, n_gram=ngram, top_n=top_n, stopwords=current_stopwords, verbose=verbose)
        if self.ngram_df.empty: print("Warning: No data to plot for bar_ngram after processing. Empty DataFrame."); return go.Figure()
        fig = px.bar(self.ngram_df.sort_values('word_count') if horizon else self.ngram_df,
                     y='word' if horizon else 'word_count', x='word_count' if horizon else 'word',
                     text='word_count', orientation='h' if horizon else 'v')
        fig.update_traces(texttemplate='%{text:.2s}', textposition='auto', marker_color=color)
        fig.update_layout(title=str(title) if title else 'N-gram Bar Chart', xaxis_title=str(xaxis_label), yaxis_title=str(yaxis_label), width=width, height=height)
        if save: self.save_plot(fig, title if title else "bar_ngram")
        return fig

    def treemap(self, title: str = None, ngram: int = 1, top_n: int = 50, width: int = 1300, height: int = 600, stopwords: list = [], verbose: bool = True, save: bool = False) -> plotly.graph_objs.Figure:
        current_stopwords = list(set(stopwords + self.default_stopwords))
        temp_series = self.df[self.target_col].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
        self.treemap_df = generate_freq_df(temp_series, n_gram=ngram, top_n=top_n, stopwords=current_stopwords, verbose=verbose)
        if self.treemap_df.empty or 'word' not in self.treemap_df.columns or 'word_count' not in self.treemap_df.columns:
            print("Warning: No data to plot for treemap after processing. Empty or malformed DataFrame."); return go.Figure()
        fig = px.treemap(self.treemap_df, path=[px.Constant("all"), 'word'], values='word_count')
        fig.update_layout(title=str(title) if title else 'Treemap', width=width, height=height)
        if save: self.save_plot(fig, title if title else "treemap")
        return fig

    def word_distribution(self, title: str = None, xaxis_label: str = '', yaxis_label: str = '', width: int = 1000, height: int = 600, color: str = None, template: str = 'plotly', bins: int = None, save: bool = False) -> plotly.graph_objs.Figure:
        col_name = self.target_col + '_length'
        self.df.loc[:, col_name] = self.df[self.target_col].apply(lambda x: len(x) if isinstance(x, list) else 0)
        if self.df[col_name].empty: print("Warning: No data to plot for word_distribution."); return go.Figure()
        fig = px.histogram(self.df, x=col_name, color=color, template=template, nbins=bins)
        fig.update_layout(title=str(title) if title else 'Word Distribution',
                          xaxis_title=str(xaxis_label) if xaxis_label else 'Number of Words',
                          yaxis_title=str(yaxis_label) if yaxis_label else 'Frequency',
                          width=width, height=height)
        if save: self.save_plot(fig, title if title else "word_distribution")
        return fig

    def wordcloud(self, width: int = 800, height: int = 500, max_words: int = 100, max_font_size: int = 80, stopwords: list = [], colormap: str = None, mask_file: str = None, font_path: str = None, save: bool = False) -> None:
        # Determine the font path to use
        wc_font_path = None
        if font_path is not None: # User explicitly provided a font_path for this call
            if os.path.exists(font_path):
                wc_font_path = font_path
            else:
                print(f"Warning: Specified font_path '{font_path}' for wordcloud not found. Will attempt to use WordCloud default.")
        elif self.font_path is not None: # Font path was set at NLPlotLLM instantiation
            if os.path.exists(self.font_path):
                wc_font_path = self.font_path
            else:
                print(f"Warning: Instance font_path '{self.font_path}' not found. Will attempt to use WordCloud default.")

        # If wc_font_path is still None here, WordCloud will try its own defaults (usually system fonts).
        if wc_font_path is None:
            print("Info: No valid font_path provided. WordCloud will attempt to use its default system font. Ensure a font is available for your language.")

        mask = None
        if mask_file and os.path.exists(mask_file):
            try: mask = np.array(Image.open(mask_file))
            except PermissionError: print(f"Warning: Permission denied to read mask file {mask_file}. Proceeding without mask."); mask = None
            except IOError as e: print(f"Warning: Could not load mask file {mask_file} due to an IO error: {e}. Proceeding without mask."); mask = None
            except Exception as e: print(f"Warning: Could not load mask file {mask_file}: {e}. Proceeding without mask."); mask = None
        elif mask_file: print(f"Warning: Mask file {mask_file} not found. Proceeding without mask.")

        processed_texts = [' '.join(map(str, item)) if isinstance(item, list) else (item if isinstance(item, str) else "") for item in self.df[self.target_col]]
        if not processed_texts: print("Warning: No text data available for wordcloud after processing."); return
        text_corpus = ' '.join(processed_texts)
        current_stopwords = set(stopwords + self.default_stopwords)
        if not text_corpus.strip(): print("Warning: Text corpus is empty after processing stopwords for wordcloud."); return

        try:
            wordcloud_instance = WordCloud(
                font_path=wc_font_path, # This can be None
                stopwords=current_stopwords,
                max_words=max_words,
                max_font_size=max_font_size,
                random_state=42,
                width=width,
                height=height,
                mask=mask,
                collocations=False,
                prefer_horizontal=1,
                colormap=colormap,
                background_color='white',
                font_step=1,
                contour_width=0,
                contour_color='steelblue'
            )
            wordcloud_instance.generate(text_corpus)
        except OSError as e: # Typically font not found or unreadable by WordCloud/FreeType
             print(f"Error during WordCloud generation (possibly font-related): {e}. If no font_path was specified, ensure system fonts are available. If a font_path was specified ('{wc_font_path}'), check its validity.")
             return
        except ValueError as e: # Often from empty corpus after stopword removal
            if "empty" in str(e).lower() or "zero" in str(e).lower():
                print(f"Warning: WordCloud could not be generated. All words might have been filtered out or corpus is empty. Details: {e}")
            else:
                print(f"An unexpected ValueError occurred during WordCloud generation: {e}")
            return
        except Exception as e:
            print(f"An unexpected error occurred during WordCloud generation: {e}")
            return

        img_array = wordcloud_instance.to_array()
        def show_array(img_array_to_show, save_flag, output_path, filename_prefix_wc):
            stream = BytesIO(); pil_img = Image.fromarray(img_array_to_show)
            if save_flag:
                date_str = pd.to_datetime(datetime.datetime.now()).strftime('%Y-%m-%d'); filename = f"{date_str}_{filename_prefix_wc}_wordcloud.png"; full_save_path = os.path.join(output_path, filename)
                try: os.makedirs(output_path, exist_ok=True); pil_img.save(full_save_path); print(f"Wordcloud image saved to {full_save_path}")
                except PermissionError: print(f"Error: Permission denied to save wordcloud image to '{full_save_path}'. Please check directory permissions.")
                except Exception as e_save: print(f"Error saving wordcloud image to '{full_save_path}': {e_save}")
            pil_img.save(stream, 'png'); IPython.display.display(IPython.display.Image(data=stream.getvalue()))
        show_array(img_array, save, self.output_file_path, "wordcloud_plot"); return None

    def get_edges_nodes(self, batches: list, min_edge_frequency: int) -> None:
        if not isinstance(min_edge_frequency, int) or min_edge_frequency < 0: raise ValueError("min_edge_frequency must be a non-negative integer.")
        edge_dict = {}
        for batch in batches:
            if isinstance(batch, list) and batch:
                 unique_elements_in_batch = list(set(batch))
                 if len(unique_elements_in_batch) >= 2: edge_dict = _add_unique_combinations_to_dict(_unique_combinations_for_edges(unique_elements_in_batch), edge_dict)
        source, target, edge_frequency_list = [], [], []
        for key, value in edge_dict.items(): source.append(key[0]); target.append(key[1]); edge_frequency_list.append(value)
        edge_df = pd.DataFrame({'source': source, 'target': target, 'edge_frequency': edge_frequency_list})
        edge_df = edge_df[edge_df['edge_frequency'] > min_edge_frequency].sort_values(by='edge_frequency', ascending=False).reset_index(drop=True)
        if edge_df.empty:
            self.edge_df = pd.DataFrame(columns=['source', 'target', 'edge_frequency', 'source_code', 'target_code'])
            self.node_df = pd.DataFrame(columns=['id', 'id_code']); self.node_dict = {}; self.edge_dict = edge_dict; return
        unique_nodes = list(set(edge_df['source']).union(set(edge_df['target'])))
        node_df = pd.DataFrame({'id': unique_nodes})
        if not node_df.empty:
            node_df['id_code'] = node_df.index; node_dict = dict(zip(node_df['id'], node_df['id_code']))
            edge_df['source_code'] = edge_df['source'].map(node_dict); edge_df['target_code'] = edge_df['target'].map(node_dict)
            edge_df.dropna(subset=['source_code', 'target_code'], inplace=True)
        else: node_dict = {}; edge_df = pd.DataFrame(columns=['source', 'target', 'edge_frequency', 'source_code', 'target_code'])
        self.edge_df = edge_df; self.node_df = node_df; self.node_dict = node_dict; self.edge_dict = edge_dict; return None

    def get_graph(self) -> nx.Graph:
        G = nx.Graph()
        if not hasattr(self, 'node_df') or self.node_df.empty:
            print("Warning: Node DataFrame is not initialized or empty. Cannot build graph.")
            return G
        G.add_nodes_from(self.node_df.id_code)
        if not hasattr(self, 'edge_df') or self.edge_df.empty: return G
        edge_tuples = [(self.edge_df['source_code'].iloc[i], self.edge_df['target_code'].iloc[i]) for i in range(len(self.edge_df))]
        G.add_edges_from(edge_tuples); return G

    def build_graph(self, stopwords: list = [], min_edge_frequency: int = 10) -> None:
        self._prepare_data_for_graph(stopwords); self.get_edges_nodes(self._batches, min_edge_frequency)
        if self.node_df.empty: self._initialize_empty_graph_attributes(); print('Warning: No nodes found after processing for build_graph. Co-occurrence network cannot be built.'); print('node_size:0, edge_size:0'); return
        self.G = self.get_graph()
        if not self.G.nodes(): self._initialize_empty_graph_attributes(graph_exists_but_no_nodes=True); print('Warning: Graph has no nodes. Further calculations for co-occurrence network will be skipped.'); print(f'node_size:{len(self.node_df)}, edge_size:{len(self.edge_df if hasattr(self, "edge_df") else [])}'); return
        self._calculate_graph_metrics(); self._detect_communities()
        print(f'node_size:{len(self.node_df)}, edge_size:{len(self.edge_df if hasattr(self, "edge_df") else [])}'); return None

    def _prepare_data_for_graph(self, stopwords_param: list):
        current_stopwords = list(set(stopwords_param + self.default_stopwords))
        self.df_edit = self.df.copy()
        self.df_edit.loc[:, self.target_col] = self.df_edit[self.target_col].apply(lambda doc: list(set(w for w in doc if w not in current_stopwords)) if isinstance(doc, list) else [])
        self._batches = self.df_edit[self.target_col].tolist()

    def _initialize_empty_graph_attributes(self, graph_exists_but_no_nodes=False):
        self.G = nx.Graph(); self.adjacencies = {}; self.betweeness = {}; self.clustering_coeff = {}; self.communities = []; self.communities_dict = {}
        if not graph_exists_but_no_nodes and hasattr(self, 'node_df') and not self.node_df.empty : self.node_df['community'] = -1

    def _calculate_graph_metrics(self):
        if not hasattr(self, 'G') or not self.G.nodes(): print("Warning: Graph not available for metric calculation."); return
        self.adjacencies = dict(self.G.adjacency()); self.betweeness = nx.betweenness_centrality(self.G); self.clustering_coeff = nx.clustering(self.G)
        self.node_df['adjacency_frequency'] = self.node_df['id_code'].map(lambda x: len(self.adjacencies.get(x, {})))
        self.node_df['betweeness_centrality'] = self.node_df['id_code'].map(lambda x: self.betweeness.get(x, 0.0))
        self.node_df['clustering_coefficient'] = self.node_df['id_code'].map(lambda x: self.clustering_coeff.get(x, 0.0))

    def _detect_communities(self):
        if not hasattr(self, 'G') or not self.G.nodes() or self.node_df.empty:
            print("Warning: Graph or node_df not available for community detection."); self.communities = []; self.communities_dict = {}
            if hasattr(self, 'node_df') and not self.node_df.empty: self.node_df['community'] = -1; return
        raw_communities = community.greedy_modularity_communities(self.G); self.communities = [list(comm) for comm in raw_communities if comm]
        self.communities_dict = {i: comm_nodes for i, comm_nodes in enumerate(self.communities)}
        def community_allocation(id_code):
            for k, v_list in self.communities_dict.items():
                if id_code in v_list: return k
            return -1
        self.node_df['community'] = self.node_df['id_code'].map(community_allocation)

    def _create_network_trace(self, trace_type: str, **kwargs) -> go.Scatter:
        if trace_type == "edge": return go.Scatter(x=kwargs['x'], y=kwargs['y'], mode='lines', line={'width': kwargs['width'], 'color': kwargs['color']}, line_shape='spline', opacity=kwargs['opacity'])
        elif trace_type == "node": return go.Scatter(x=kwargs['x'], y=kwargs['y'], text=kwargs['text'], mode='markers+text', textposition='bottom center', hoverinfo="text", marker=kwargs['marker'])
        raise ValueError(f"Unknown trace_type: {trace_type}")

    def co_network(self, title:str = None, sizing:int=100, node_size_col:str='adjacency_frequency', color_palette:str='hls', layout_func=nx.kamada_kawai_layout, light_theme:bool=True, width:int=1700, height:int=1200, save:bool=False) -> None:
        if not hasattr(self, 'G') or not self.G.nodes(): print("Warning: Graph not built or empty. Cannot plot co-occurrence network."); return
        if not hasattr(self, 'node_df') or self.node_df.empty: print("Warning: Node DataFrame not available or empty. Cannot plot co-occurrence network."); return
        if node_size_col not in self.node_df.columns:
            print(f"Warning: node_size column '{node_size_col}' not found in node_df. Using 'adjacency_frequency'."); node_size_col = 'adjacency_frequency'
            if node_size_col not in self.node_df.columns: print(f"Warning: Default node_size column 'adjacency_frequency' also not found. Node sizes will be uniform."); self.node_df['uniform_size'] = 10; node_size_col = 'uniform_size'
        back_col, edge_col = ('#ffffff', '#ece8e8') if light_theme else ('#000000', '#2d2b2b'); final_node_sizes = self._calculate_node_sizes(node_size_col, sizing)
        pos = layout_func(self.G)
        for node_id_code in self.G.nodes(): self.G.nodes[node_id_code]['pos'] = list(pos[node_id_code])
        edge_traces = [self._create_network_trace(trace_type="edge", x=[self.G.nodes[edge_nodes[0]]['pos'][0], self.G.nodes[edge_nodes[1]]['pos'][0], None], y=[self.G.nodes[edge_nodes[0]]['pos'][1], self.G.nodes[edge_nodes[1]]['pos'][1], None], width=1.2, color=edge_col, opacity=1) for edge_nodes in self.G.edges()]
        node_x, node_y, node_hover_text, node_marker_colors, node_marker_sizes = [], [], [], [], []
        if 'community' not in self.node_df.columns or not pd.api.types.is_numeric_dtype(self.node_df['community']): self.node_df['community_display'] = 0
        else: self.node_df['community_display'] = self.node_df['community']
        num_communities = self.node_df['community_display'].nunique(); palette_colors = get_colorpalette(color_palette, num_communities if num_communities > 0 else 1)
        id_code_to_info = self.node_df.set_index('id_code')
        for id_code_node in self.G.nodes():
            x, y = self.G.nodes[id_code_node]['pos']; node_x.append(x); node_y.append(y)
            node_specific_info = id_code_to_info.loc[id_code_node]; node_hover_text.append(node_specific_info['id'])
            community_val = int(node_specific_info['community_display']); node_marker_colors.append(palette_colors[community_val % len(palette_colors)])
            node_marker_sizes.append(final_node_sizes.loc[node_specific_info.name])
        node_trace = self._create_network_trace(trace_type="node", x=node_x, y=node_y, text=node_hover_text, marker={'size': node_marker_sizes, 'line': dict(width=0.5, color=edge_col), 'color': node_marker_colors})
        fig_data = edge_traces + [node_trace]
        fig_layout = go.Layout(title=str(title) if title else "Co-occurrence Network", font=dict(family='Arial', size=12), width=width, height=height, autosize=True, showlegend=False, xaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''), yaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''), margin=dict(l=40, r=40, b=85, t=100, pad=0), hovermode='closest', plot_bgcolor=back_col)
        fig = go.Figure(data=fig_data, layout=fig_layout); iplot(fig)
        if save: self.save_plot(fig, title if title else "co_network")
        gc.collect(); return None

    def _calculate_node_sizes(self, node_size_col: str, sizing_factor: int) -> pd.Series:
        if node_size_col not in self.node_df.columns or self.node_df[node_size_col].isnull().all():
            print(f"Warning: Node size column '{node_size_col}' not found or all nulls. Using uniform small size.")
            return pd.Series([sizing_factor * 0.1] * len(self.node_df), index=self.node_df.index)
        node_sizes_numeric = pd.to_numeric(self.node_df[node_size_col], errors='coerce').fillna(0)
        if len(node_sizes_numeric) == 0: return pd.Series(index=self.node_df.index, dtype=float)
        if node_sizes_numeric.nunique() <= 1:
            if node_sizes_numeric.iloc[0] == 0 : return pd.Series([sizing_factor * 0.1] * len(self.node_df), index=self.node_df.index)
            else: return pd.Series([sizing_factor * 0.5] * len(self.node_df), index=self.node_df.index)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 1.0))
        scaled_values = min_max_scaler.fit_transform(node_sizes_numeric.values.reshape(-1, 1)).flatten()
        return pd.Series(scaled_values, index=self.node_df.index) * sizing_factor

    def sunburst(self, title:str=None, colorscale:bool=False, color_col:str='betweeness_centrality', color_continuous_scale:str='Oryel', width:int=1100, height:int=1100, save:bool=False) -> plotly.graph_objs.Figure:
        if not hasattr(self, 'node_df') or self.node_df.empty: print("Warning: Node DataFrame not available or empty. Cannot plot sunburst chart."); return go.Figure()
        _df = self.node_df.copy()
        if 'community' not in _df.columns: _df['community'] = '0'
        else: _df['community'] = _df['community'].astype(str)
        if 'id' not in _df.columns: _df['id'] = "Unknown"
        if 'adjacency_frequency' not in _df.columns or _df['adjacency_frequency'].isnull().all():
            print("Warning: 'adjacency_frequency' column is missing or all nulls. Sunburst may be empty or error."); _df['adjacency_frequency'] = 1
        path_cols = ['community', 'id']
        try:
            if colorscale:
                if color_col not in _df.columns or _df[color_col].isnull().all():
                    print(f"Warning: color_col '{color_col}' for sunburst is missing or all nulls. Using default coloring."); fig = px.sunburst(_df, path=path_cols, values='adjacency_frequency', color='community')
                else:
                    _df[color_col] = pd.to_numeric(_df[color_col], errors='coerce').fillna(0)
                    fig = px.sunburst(_df, path=path_cols, values='adjacency_frequency', color=color_col, hover_data=None, color_continuous_scale=color_continuous_scale, color_continuous_midpoint=np.average(_df[color_col].fillna(0), weights=_df['adjacency_frequency'].fillna(1)))
            else: fig = px.sunburst(_df, path=path_cols, values='adjacency_frequency', color='community')
        except Exception as e: print(f"Error creating sunburst chart: {e}. Returning empty figure."); return go.Figure()
        fig.update_layout(title=str(title) if title else 'Sunburst Chart', width=width, height=height)
        if save: self.save_plot(fig, title if title else "sunburst_chart")
        del _df; gc.collect(); return fig

    def save_plot(self, fig, title_prefix: str) -> None:
        if not title_prefix or not isinstance(title_prefix, str): title_prefix = "plot"
        title_prefix = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in title_prefix)
        date_str = pd.to_datetime(datetime.datetime.now()).strftime('%Y-%m-%d')
        filename = f"{date_str}_{title_prefix}.html"
        full_path = os.path.join(self.output_file_path, filename)
        try: os.makedirs(self.output_file_path, exist_ok=True); plotly.offline.plot(fig, filename=full_path, auto_open=False); print(f"Plot saved to {full_path}")
        except PermissionError: print(f"Error: Permission denied to write plot to '{full_path}'. Please check directory permissions.")
        except Exception as e: print(f"Error saving plot to '{full_path}': {e}")
        return None

    def save_tables(self, prefix: str = "nlplot_output") -> None:
        if not hasattr(self, 'node_df') or not hasattr(self, 'edge_df'): print("Warning: node_df or edge_df attributes not found. Ensure build_graph() has been called. Cannot save tables."); return
        date_str = pd.to_datetime(datetime.datetime.now()).strftime('%Y-%m-%d')
        sanitized_prefix = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in prefix)
        try:
            os.makedirs(self.output_file_path, exist_ok=True)
            if hasattr(self, 'node_df') and isinstance(self.node_df, pd.DataFrame) and not self.node_df.empty:
                node_filename = os.path.join(self.output_file_path, f"{date_str}_{sanitized_prefix}_node_df.csv")
                self.node_df.to_csv(node_filename, index=False); print(f'Saved nodes to {node_filename}')
            else: print('Node DataFrame is empty or not available. Not saved.')
            if hasattr(self, 'edge_df') and isinstance(self.edge_df, pd.DataFrame) and not self.edge_df.empty:
                edge_filename = os.path.join(self.output_file_path, f"{date_str}_{sanitized_prefix}_edge_df.csv")
                self.edge_df.to_csv(edge_filename, index=False); print(f'Saved edges to {edge_filename}')
            else: print('Edge DataFrame is empty or not available. Not saved.')
        except PermissionError: print(f"Error: Permission denied to write tables in '{self.output_file_path}'. Please check directory permissions.")
        except Exception as e: print(f"Error saving tables: {e}")
        return None

    # --- LLM Related Methods ---
    # _get_llm_client method is removed as LiteLLM will be used directly.

    def analyze_sentiment_llm(
        self,
        text_series: pd.Series,
        model: str, # Changed from llm_provider and model_name to LiteLLM's model string
        prompt_template_str: Optional[str] = None,
        temperature: float = 0.0,
        use_cache: Optional[bool] = None, # Added for method-specific cache control
        **litellm_kwargs
    ) -> pd.DataFrame:
        """
        Analyzes sentiment of texts using a specified LLM via LiteLLM.

        Args:
            text_series (pd.Series): Series containing texts to analyze.
            model (str): The LiteLLM model string, identifying the provider and model
                         (e.g., "openai/gpt-3.5-turbo", "ollama/llama2", "azure/your-deployment-name").
            prompt_template_str (Optional[str], optional): Custom prompt template string.
                Must include "{text}" placeholder.
                Defaults to a standard sentiment analysis prompt:
                "Analyze the sentiment of the following text and classify it as 'positive', 'negative', or 'neutral'. Return only the single word classification for the sentiment. Text: {text}".
            temperature (float, optional): Temperature for LLM generation. Defaults to 0.0.
            use_cache (Optional[bool], optional): Whether to use caching for this specific call.
                                                 If None, uses the instance's default setting (`self.use_cache_default`).
                                                 Defaults to None.
            **litellm_kwargs: Additional keyword arguments to pass directly to `litellm.completion`.
                              This can include provider-specific parameters like `api_key`, `api_base`,
                              `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, etc.
                              Refer to LiteLLM documentation for details.
                              Example: `api_key="sk-..."`, `api_base="http://localhost:11434"`, `max_tokens=50`.

        Returns:
            pd.DataFrame: DataFrame with columns ["text", "sentiment", "raw_llm_output"].
                          "sentiment" can be "positive", "negative", "neutral", "unknown", or "error".
                          "raw_llm_output" contains the direct string content from the LLM response or error details.
        """
        if not LITELLM_AVAILABLE:
            print("Warning: LiteLLM is not installed. LLM-based sentiment analysis is not available.")
            return pd.DataFrame(columns=["text", "sentiment", "raw_llm_output"])

        # Langchain's PromptTemplate can still be useful if available
        # We'll need to import it specifically if we decide to keep using it.
        # For now, let's assume basic string formatting for prompts.
        # from langchain_core.prompts import PromptTemplate # Would need its own availability check or be always installed

        if not isinstance(text_series, pd.Series):
            print("Warning: Input 'text_series' must be a pandas Series. Returning empty DataFrame with expected columns.")
            return pd.DataFrame(columns=["text", "sentiment", "raw_llm_output"])
        if text_series.empty:
            return pd.DataFrame(columns=["text", "sentiment", "raw_llm_output"])

        # Determine if cache should be used for this call
        current_use_cache = use_cache if use_cache is not None else self.use_cache_default

        final_prompt_template_str = prompt_template_str
        if final_prompt_template_str is None:
            final_prompt_template_str = "Analyze the sentiment of the following text and classify it as 'positive', 'negative', or 'neutral'. Return only the single word classification for the sentiment. Text: {text}"

        if "{text}" not in final_prompt_template_str:
            print("Error: Prompt template must include '{text}' placeholder.")
            return pd.DataFrame(
                [{'text': str(txt) if pd.notna(txt) else "",
                  'sentiment': 'error',
                  'raw_llm_output': "Prompt template error: missing {text}"} for txt in text_series],
                columns=["text", "sentiment", "raw_llm_output"]
            )

        results = []
        for text_input in text_series:
            original_text_for_df = str(text_input) if pd.notna(text_input) else ""
            sentiment = "unknown"
            raw_llm_response_content = ""

            if not original_text_for_df.strip():
                sentiment = "neutral"
                raw_llm_response_content = "Input text was empty or whitespace."
            else:
                # --- Cache Key Generation ---
                # Sort kwargs to ensure consistent key order. Only include relevant ones for cache.
                # More sophisticated hashing might be needed for complex kwargs.
                cacheable_kwargs_items = sorted(
                    (k, v) for k, v in litellm_kwargs.items()
                    if k in ("temperature", "max_tokens", "top_p") # Add other relevant kwargs here
                )
                cache_key_tuple = (
                    "analyze_sentiment_llm", # Method identifier
                    model,
                    final_prompt_template_str,
                    original_text_for_df,
                    tuple(cacheable_kwargs_items)
                )

                cached_result = None
                if current_use_cache and self.cache:
                    try:
                        cached_result = self.cache.get(cache_key_tuple)
                    except Exception as e:
                        print(f"Warning: Cache get failed for key {cache_key_tuple}. Error: {e}")


                if cached_result is not None:
                    sentiment = cached_result["sentiment"]
                    raw_llm_response_content = cached_result["raw_llm_output"]
                    # print(f"Cache HIT for sentiment: {original_text_for_df[:30]}...") # For debugging
                else:
                    # print(f"Cache MISS for sentiment: {original_text_for_df[:30]}...") # For debugging
                    try:
                        messages = [{"role": "user", "content": final_prompt_template_str.format(text=original_text_for_df)}]
                        completion_kwargs = {k: v for k, v in litellm_kwargs.items()} # Pass all litellm_kwargs
                        if 'temperature' not in completion_kwargs:
                            completion_kwargs['temperature'] = temperature # Default if not in kwargs

                        response = litellm.completion(
                            model=model,
                            messages=messages,
                            **completion_kwargs
                        )
                        if response.choices and response.choices[0].message:
                            raw_llm_response_content = response.choices[0].message.content or ""
                        else:
                            raw_llm_response_content = str(response)

                        processed_output = raw_llm_response_content.strip().lower()
                        if "positive" in processed_output: sentiment = "positive"
                        elif "negative" in processed_output: sentiment = "negative"
                        elif "neutral" in processed_output: sentiment = "neutral"
                        # else: sentiment remains "unknown"

                        if current_use_cache and self.cache:
                            try:
                                self.cache.set(cache_key_tuple, {"sentiment": sentiment, "raw_llm_output": raw_llm_response_content})
                            except Exception as e:
                                print(f"Warning: Cache set failed for key {cache_key_tuple}. Error: {e}")

                    except ImportError:
                        print("LiteLLM is not installed. Cannot perform sentiment analysis.")
                        sentiment, raw_llm_response_content = "error", "LiteLLM not installed."
                    except litellm.exceptions.AuthenticationError as e:
                        print(f"LiteLLM Authentication Error: {e}"); sentiment, raw_llm_response_content = "error", f"AuthenticationError: {e}"
                    except litellm.exceptions.APIConnectionError as e:
                        print(f"LiteLLM API Connection Error: {e}"); sentiment, raw_llm_response_content = "error", f"APIConnectionError: {e}"
                    except litellm.exceptions.RateLimitError as e:
                        print(f"LiteLLM Rate Limit Error: {e}"); sentiment, raw_llm_response_content = "error", f"RateLimitError: {e}"
                    except Exception as e:
                        print(f"Error analyzing sentiment for text '{original_text_for_df[:50]}...': {e}"); sentiment, raw_llm_response_content = "error", str(e)

            results.append({"text": original_text_for_df, "sentiment": sentiment, "raw_llm_output": raw_llm_response_content})

        if self.cache: # Close the cache connection if it was opened by this instance
            try: self.cache.close()
            except Exception as e: print(f"Warning: Error closing cache: {e}")

        return pd.DataFrame(results, columns=["text", "sentiment", "raw_llm_output"])

    async def _analyze_sentiment_single_async(
        self,
        text_to_analyze: str,
        model: str,
        prompt_template_str: str,
        temperature: float,
        current_use_cache: bool,
        litellm_kwargs: dict
    ) -> dict:
        """
        Helper coroutine to process a single text for sentiment analysis asynchronously,
        including cache lookup and storage.

        Args:
            text_to_analyze (str): The text string to analyze.
            model (str): LiteLLM model string.
            prompt_template_str (str): The prompt template string.
            temperature (float): LLM temperature.
            current_use_cache (bool): Whether caching is enabled for this operation.
            litellm_kwargs (dict): Additional kwargs for litellm.acompletion.

        Returns:
            dict: A dictionary containing "text", "sentiment", and "raw_llm_output".
                  If an error occurs, "sentiment" will be "error" and "raw_llm_output" will contain the error message.
        """
        original_text_for_df = str(text_to_analyze) if pd.notna(text_to_analyze) else ""
        sentiment = "unknown"
        raw_llm_response_content = ""

        if not original_text_for_df.strip():
            sentiment = "neutral"
            raw_llm_response_content = "Input text was empty or whitespace."
        else:
            cacheable_kwargs_items = sorted(
                (k, v) for k, v in litellm_kwargs.items()
                if k in ("temperature", "max_tokens", "top_p")
            )
            cache_key_tuple = (
                "analyze_sentiment_llm", model, prompt_template_str, # Note: using the base method name for cache key consistency
                original_text_for_df, tuple(cacheable_kwargs_items)
            )

            cached_result = None
            if current_use_cache and self.cache:
                try: cached_result = self.cache.get(cache_key_tuple)
                except Exception as e: print(f"Warning: Async cache get failed. Error: {e}")

            if cached_result is not None:
                sentiment = cached_result["sentiment"]
                raw_llm_response_content = cached_result["raw_llm_output"]
            else:
                try:
                    messages = [{"role": "user", "content": prompt_template_str.format(text=original_text_for_df)}]
                    completion_kwargs = {**litellm_kwargs} # Ensure all kwargs are passed
                    if 'temperature' not in completion_kwargs: completion_kwargs['temperature'] = temperature

                    response = await litellm.acompletion(model=model, messages=messages, **completion_kwargs)

                    if response.choices and response.choices[0].message:
                        raw_llm_response_content = response.choices[0].message.content or ""
                    else: raw_llm_response_content = str(response)

                    processed_output = raw_llm_response_content.strip().lower()
                    if "positive" in processed_output: sentiment = "positive"
                    elif "negative" in processed_output: sentiment = "negative"
                    elif "neutral" in processed_output: sentiment = "neutral"

                    if current_use_cache and self.cache:
                        try: self.cache.set(cache_key_tuple, {"sentiment": sentiment, "raw_llm_output": raw_llm_response_content})
                        except Exception as e: print(f"Warning: Async cache set failed. Error: {e}")
                except Exception as e:
                    print(f"Error in _analyze_sentiment_single_async for '{original_text_for_df[:30]}...': {e}")
                    sentiment, raw_llm_response_content = "error", str(e)

        return {"text": original_text_for_df, "sentiment": sentiment, "raw_llm_output": raw_llm_response_content}

    async def analyze_sentiment_llm_async(
        self,
        text_series: pd.Series,
        model: str,
        prompt_template_str: Optional[str] = None,
        temperature: float = 0.0,
        use_cache: Optional[bool] = None,
        concurrency_limit: Optional[int] = None, # Max concurrent requests
        return_exceptions: bool = True, # For asyncio.gather
        **litellm_kwargs
    ) -> pd.DataFrame:
        """
        Asynchronously analyzes sentiment of texts using a specified LLM via LiteLLM.
        Results are returned in the same order as the input text_series.

        Args:
            text_series (pd.Series): Series containing texts to analyze.
            model (str): The LiteLLM model string (e.g., "openai/gpt-4o", "ollama/gemma").
            prompt_template_str (Optional[str], optional): Custom prompt template.
                Must include "{text}". Defaults to a standard sentiment analysis prompt.
            temperature (float, optional): Temperature for LLM generation. Defaults to 0.0.
            use_cache (Optional[bool], optional): Whether to use caching. Overrides instance default.
                                                 Defaults to None (use instance default).
            concurrency_limit (Optional[int], optional): Maximum number of concurrent LLM API calls.
                                                       Defaults to None (no limit).
            return_exceptions (bool, optional): If True, exceptions during individual API calls
                                                are caught and returned as part of the result for that item.
                                                If False, the first exception will stop all processing.
                                                Defaults to True.
            **litellm_kwargs: Additional arguments for `litellm.acompletion`
                              (e.g., `api_key`, `max_tokens`).

        Returns:
            pd.DataFrame: DataFrame with columns ["text", "sentiment", "raw_llm_output"].
                          If `return_exceptions` is True, failed items will have "error" as sentiment
                          and the error message in "raw_llm_output".
        """
        if not LITELLM_AVAILABLE:
            print("Warning: LiteLLM is not installed. Async LLM-based sentiment analysis is not available.")
            return pd.DataFrame(columns=["text", "sentiment", "raw_llm_output"])
        if not isinstance(text_series, pd.Series):
            print("Warning: Input 'text_series' must be a pandas Series.")
            return pd.DataFrame(columns=["text", "sentiment", "raw_llm_output"])
        if text_series.empty:
            return pd.DataFrame(columns=["text", "sentiment", "raw_llm_output"])

        current_use_cache = use_cache if use_cache is not None else self.use_cache_default
        final_prompt_template_str = prompt_template_str
        if final_prompt_template_str is None:
            final_prompt_template_str = "Analyze the sentiment of the following text and classify it as 'positive', 'negative', or 'neutral'. Return only the single word classification for the sentiment. Text: {text}"
        if "{text}" not in final_prompt_template_str:
            print("Error: Prompt template must include '{text}' placeholder.")
            # Consider how to handle this for async - maybe return a list of error dicts
            results = [{"text": str(txt), "sentiment": "error", "raw_llm_output": "Prompt template error: missing {text}"} for txt in text_series]
            return pd.DataFrame(results, columns=["text", "sentiment", "raw_llm_output"])

        tasks = []
        semaphore = asyncio.Semaphore(concurrency_limit) if concurrency_limit and concurrency_limit > 0 else None

        async def task_wrapper(text_input):
            if semaphore:
                async with semaphore:
                    return await self._analyze_sentiment_single_async(
                        text_input, model, final_prompt_template_str, temperature, current_use_cache, litellm_kwargs
                    )
            else:
                return await self._analyze_sentiment_single_async(
                    text_input, model, final_prompt_template_str, temperature, current_use_cache, litellm_kwargs
                )

        for text_input in text_series:
            tasks.append(task_wrapper(text_input))

        all_results_raw = await asyncio.gather(*tasks, return_exceptions=return_exceptions)

        # Process results, handling potential exceptions from gather
        processed_results = []
        for i, res_or_exc in enumerate(all_results_raw):
            original_text = str(text_series.iloc[i]) if pd.notna(text_series.iloc[i]) else ""
            if isinstance(res_or_exc, Exception):
                print(f"Exception for text '{original_text[:30]}...': {res_or_exc}")
                processed_results.append({"text": original_text, "sentiment": "error", "raw_llm_output": str(res_or_exc)})
            else: # Should be a dict from _analyze_sentiment_single_async
                processed_results.append(res_or_exc)

        if self.cache: # Close cache if opened by this instance
            try: self.cache.close()
            except Exception as e: print(f"Warning: Error closing cache: {e}")

        return pd.DataFrame(processed_results, columns=["text", "sentiment", "raw_llm_output"])

    async def _categorize_text_single_async(
        self,
        text_to_categorize: str,
        categories: List[str],
        category_list_str: str, # Pre-formatted string of categories for prompt
        model: str,
        prompt_template_str: str,
        multi_label: bool,
        temperature: float,
        current_use_cache: bool,
        litellm_kwargs: dict
    ) -> dict:
        """
        Helper coroutine to process a single text for categorization asynchronously,
        including cache lookup and storage.

        Args:
            text_to_categorize (str): The text string to categorize.
            categories (List[str]): List of predefined categories.
            category_list_str (str): Comma-separated string of categories for the prompt.
            model (str): LiteLLM model string.
            prompt_template_str (str): The prompt template string.
            multi_label (bool): Whether multi-label categorization is enabled.
            temperature (float): LLM temperature.
            current_use_cache (bool): Whether caching is enabled for this operation.
            litellm_kwargs (dict): Additional kwargs for litellm.acompletion.

        Returns:
            dict: A dictionary containing "text", the category column ("category" or "categories"),
                  "raw_llm_output", and a temporary "_multi_label" flag.
                  If an error occurs, the category column will hold "error" (or [] for multi_label)
                  and "raw_llm_output" will contain the error message.
        """
        original_text_for_df = str(text_to_categorize) if pd.notna(text_to_categorize) else ""
        category_col_name = "categories" if multi_label else "category"
        parsed_categories_result: Any = [] if multi_label else "unknown"
        raw_llm_response_content = ""

        if not original_text_for_df.strip():
            raw_llm_response_content = "Input text was empty or whitespace."
            # parsed_categories_result remains default
        else:
            cacheable_kwargs_items = sorted(
                (k, v) for k, v in litellm_kwargs.items() if k in ("temperature", "max_tokens", "top_p")
            )
            cache_key_tuple = (
                "categorize_text_llm", model, prompt_template_str,
                original_text_for_df, tuple(sorted(categories)), multi_label,
                tuple(cacheable_kwargs_items)
            )

            cached_result = None
            if current_use_cache and self.cache:
                try: cached_result = self.cache.get(cache_key_tuple)
                except Exception as e: print(f"Warning: Async cache get failed for categorize. Error: {e}")

            if cached_result is not None:
                parsed_categories_result = cached_result["categories_result"]
                raw_llm_response_content = cached_result["raw_llm_output"]
            else:
                try:
                    format_args = {"text": original_text_for_df}
                    if "{categories}" in prompt_template_str: format_args["categories"] = category_list_str
                    formatted_content_for_llm = prompt_template_str.format(**format_args)
                    messages = [{"role": "user", "content": formatted_content_for_llm}]

                    completion_kwargs = {**litellm_kwargs}
                    if 'temperature' not in completion_kwargs: completion_kwargs['temperature'] = temperature

                    response = await litellm.acompletion(model=model, messages=messages, **completion_kwargs)

                    if response.choices and response.choices[0].message:
                        raw_llm_response_content = response.choices[0].message.content or ""
                    else: raw_llm_response_content = str(response)

                    processed_output = raw_llm_response_content.strip().lower()
                    if multi_label:
                        found_cats_raw = [cat.strip() for cat in processed_output.split(',')]
                        parsed_categories_result = [
                            og_cat for og_cat in categories
                            if og_cat.lower() in [fcr.lower() for fcr in found_cats_raw if fcr and fcr != 'none']
                        ]
                    else:
                        parsed_categories_result = "unknown"
                        exact_match_found = False
                        for cat_og in categories:
                            if cat_og.lower() == processed_output:
                                parsed_categories_result = cat_og; exact_match_found = True; break
                        if not exact_match_found:
                            for cat_og in categories:
                                if cat_og.lower() in processed_output:
                                    parsed_categories_result = cat_og; break
                        if parsed_categories_result == "unknown" and processed_output in ["unknown", "none"]: pass

                    if current_use_cache and self.cache:
                        try: self.cache.set(cache_key_tuple, {"categories_result": parsed_categories_result, "raw_llm_output": raw_llm_response_content})
                        except Exception as e: print(f"Warning: Async cache set failed for categorize. Error: {e}")
                except Exception as e:
                    print(f"Error in _categorize_text_single_async for '{original_text_for_df[:30]}...': {e}")
                    parsed_categories_result = [] if multi_label else "error"
                    raw_llm_response_content = str(e)

        return {"text": original_text_for_df, category_col_name: parsed_categories_result, "raw_llm_output": raw_llm_response_content, "_multi_label": multi_label}


    async def categorize_text_llm_async(
        self,
        text_series: pd.Series,
        categories: List[str],
        model: str,
        prompt_template_str: Optional[str] = None,
        multi_label: bool = False,
        temperature: float = 0.0,
        use_cache: Optional[bool] = None,
        concurrency_limit: Optional[int] = None,
        return_exceptions: bool = True,
        **litellm_kwargs
    ) -> pd.DataFrame:
        """
        Asynchronously categorizes texts into predefined categories using LiteLLM.

        Args:
            text_series (pd.Series): Series containing texts to categorize.
            categories (List[str]): A list of predefined category names.
            model (str): The LiteLLM model string (e.g., "openai/gpt-4o").
            prompt_template_str (Optional[str], optional): Custom prompt template.
                Must include "{text}" and can include "{categories}".
                Defaults to a standard categorization prompt.
            multi_label (bool, optional): If True, allows multiple categories per text.
                                        Output column becomes "categories" (List[str]).
                                        If False, "category" (str). Defaults to False.
            temperature (float, optional): LLM temperature. Defaults to 0.0.
            use_cache (Optional[bool], optional): Whether to use caching. Overrides instance default.
            concurrency_limit (Optional[int], optional): Max concurrent LLM calls. Defaults to None.
            return_exceptions (bool, optional): If True, exceptions are caught and returned in results.
                                                Defaults to True.
            **litellm_kwargs: Additional arguments for `litellm.acompletion`.

        Returns:
            pd.DataFrame: DataFrame with ["text", category_column_name, "raw_llm_output"].
                          Failed items will have "error" or [] in category_column_name.
        """
        category_col_name = "categories" if multi_label else "category"
        default_columns = ["text", category_col_name, "raw_llm_output"]

        if not LITELLM_AVAILABLE:
            print("Warning: LiteLLM is not available for async categorization.")
            return pd.DataFrame(columns=default_columns)
        if not isinstance(text_series, pd.Series):
            print("Warning: Input 'text_series' must be a pandas Series.")
            return pd.DataFrame(columns=default_columns)
        if text_series.empty: return pd.DataFrame(columns=default_columns)
        if not categories or not isinstance(categories, list) or not all(isinstance(c, str) and c for c in categories):
            raise ValueError("Categories list must be a non-empty list of non-empty strings.")

        current_use_cache = use_cache if use_cache is not None else self.use_cache_default
        category_list_str_for_prompt = ", ".join(f"'{c}'" for c in categories)

        final_prompt_template_str = prompt_template_str
        if final_prompt_template_str is None:
            if multi_label:
                final_prompt_template_str = (
                    f"Analyze the following text and classify it into one or more of these categories: {category_list_str_for_prompt}. "
                    f"Return a comma-separated list of the matching category names. If no categories match, return 'none'. Text: {{text}}"
                )
            else:
                final_prompt_template_str = (
                    f"Analyze the following text and classify it into exactly one of these categories: {category_list_str_for_prompt}. "
                    f"Return only the single matching category name. If no categories match, return 'unknown'. Text: {{text}}"
                )

        if "{text}" not in final_prompt_template_str:
            results = [{"text": str(txt), category_col_name: [] if multi_label else "error", "raw_llm_output": "Prompt template error: missing {text}"} for txt in text_series]
            return pd.DataFrame(results, columns=default_columns)

        tasks = []
        semaphore = asyncio.Semaphore(concurrency_limit) if concurrency_limit and concurrency_limit > 0 else None

        async def task_wrapper(text_input):
            if semaphore:
                async with semaphore:
                    return await self._categorize_text_single_async(
                        text_input, categories, category_list_str_for_prompt, model, final_prompt_template_str,
                        multi_label, temperature, current_use_cache, litellm_kwargs
                    )
            else:
                return await self._categorize_text_single_async(
                    text_input, categories, category_list_str_for_prompt, model, final_prompt_template_str,
                    multi_label, temperature, current_use_cache, litellm_kwargs
                )

        for text_input in text_series:
            tasks.append(task_wrapper(text_input))

        all_results_raw = await asyncio.gather(*tasks, return_exceptions=return_exceptions)

        processed_results = []
        for i, res_or_exc in enumerate(all_results_raw):
            original_text = str(text_series.iloc[i]) if pd.notna(text_series.iloc[i]) else ""
            if isinstance(res_or_exc, Exception):
                processed_results.append({"text": original_text, category_col_name: [] if multi_label else "error", "raw_llm_output": str(res_or_exc)})
            else: # Remove the temporary '_multi_label' key before creating DataFrame
                res_or_exc.pop("_multi_label", None)
                processed_results.append(res_or_exc)

        if self.cache:
            try: self.cache.close()
            except Exception as e: print(f"Warning: Error closing cache: {e}")

        return pd.DataFrame(processed_results, columns=default_columns)

    async def _categorize_text_single_async(
        self,
        text_to_categorize: str,
        categories: List[str],
        category_list_str: str, # Pre-formatted string of categories for prompt
        model: str,
        prompt_template_str: str,
        multi_label: bool,
        temperature: float,
        current_use_cache: bool,
        litellm_kwargs: dict
    ) -> dict:
        """Helper coroutine to process a single text for categorization asynchronously, with caching."""
        original_text_for_df = str(text_to_categorize) if pd.notna(text_to_categorize) else ""
        category_col_name = "categories" if multi_label else "category"
        parsed_categories_result: Any = [] if multi_label else "unknown"
        raw_llm_response_content = ""

        if not original_text_for_df.strip():
            raw_llm_response_content = "Input text was empty or whitespace."
        else:
            cacheable_kwargs_items = sorted(
                (k, v) for k, v in litellm_kwargs.items() if k in ("temperature", "max_tokens", "top_p")
            )
            cache_key_tuple = (
                "categorize_text_llm", model, prompt_template_str,
                original_text_for_df, tuple(sorted(categories)), multi_label,
                tuple(cacheable_kwargs_items)
            )

            cached_result = None
            if current_use_cache and self.cache:
                try: cached_result = self.cache.get(cache_key_tuple)
                except Exception as e: print(f"Warning: Async cache get failed for categorize. Error: {e}")

            if cached_result is not None:
                parsed_categories_result = cached_result["categories_result"]
                raw_llm_response_content = cached_result["raw_llm_output"]
            else:
                try:
                    format_args = {"text": original_text_for_df}
                    if "{categories}" in prompt_template_str: format_args["categories"] = category_list_str
                    formatted_content_for_llm = prompt_template_str.format(**format_args)
                    messages = [{"role": "user", "content": formatted_content_for_llm}]

                    completion_kwargs = {**litellm_kwargs}
                    if 'temperature' not in completion_kwargs: completion_kwargs['temperature'] = temperature

                    response = await litellm.acompletion(model=model, messages=messages, **completion_kwargs)

                    if response.choices and response.choices[0].message:
                        raw_llm_response_content = response.choices[0].message.content or ""
                    else: raw_llm_response_content = str(response)

                    processed_output = raw_llm_response_content.strip().lower()
                    if multi_label:
                        found_cats_raw = [cat.strip() for cat in processed_output.split(',')]
                        parsed_categories_result = [
                            og_cat for og_cat in categories
                            if og_cat.lower() in [fcr.lower() for fcr in found_cats_raw if fcr and fcr != 'none']
                        ]
                    else:
                        parsed_categories_result = "unknown"
                        exact_match_found = False
                        for cat_og in categories:
                            if cat_og.lower() == processed_output:
                                parsed_categories_result = cat_og; exact_match_found = True; break
                        if not exact_match_found: # Fallback to substring match if no exact match
                            for cat_og in categories:
                                if cat_og.lower() in processed_output: # Check if category is IN output
                                    parsed_categories_result = cat_og; break
                        if parsed_categories_result == "unknown" and processed_output in ["unknown", "none"]: pass

                    if current_use_cache and self.cache:
                        try: self.cache.set(cache_key_tuple, {"categories_result": parsed_categories_result, "raw_llm_output": raw_llm_response_content})
                        except Exception as e: print(f"Warning: Async cache set failed for categorize. Error: {e}")
                except Exception as e:
                    print(f"Error in _categorize_text_single_async for '{original_text_for_df[:30]}...': {e}")
                    parsed_categories_result = [] if multi_label else "error"
                    raw_llm_response_content = str(e)

        return {"text": original_text_for_df, category_col_name: parsed_categories_result, "raw_llm_output": raw_llm_response_content, "_multi_label": multi_label} # Temp key for main async method

    async def categorize_text_llm_async(
        self,
        text_series: pd.Series,
        categories: List[str],
        model: str,
        prompt_template_str: Optional[str] = None,
        multi_label: bool = False,
        temperature: float = 0.0,
        use_cache: Optional[bool] = None,
        concurrency_limit: Optional[int] = None,
        return_exceptions: bool = True,
        **litellm_kwargs
    ) -> pd.DataFrame:
        """Asynchronously categorizes texts into predefined categories."""
        category_col_name = "categories" if multi_label else "category"
        default_columns = ["text", category_col_name, "raw_llm_output"]

        if not LITELLM_AVAILABLE:
            print("Warning: LiteLLM is not available for async categorization.")
            return pd.DataFrame(columns=default_columns)
        if not isinstance(text_series, pd.Series):
            print("Warning: Input 'text_series' must be a pandas Series.")
            return pd.DataFrame(columns=default_columns)
        if text_series.empty: return pd.DataFrame(columns=default_columns)
        if not categories or not isinstance(categories, list) or not all(isinstance(c, str) and c for c in categories):
            raise ValueError("Categories list must be a non-empty list of non-empty strings.")

        current_use_cache = use_cache if use_cache is not None else self.use_cache_default
        category_list_str_for_prompt = ", ".join(f"'{c}'" for c in categories)

        final_prompt_template_str = prompt_template_str
        if final_prompt_template_str is None:
            if multi_label:
                final_prompt_template_str = (
                    f"Analyze the following text and classify it into one or more of these categories: {category_list_str_for_prompt}. "
                    f"Return a comma-separated list of the matching category names. If no categories match, return 'none'. Text: {{text}}"
                )
            else:
                final_prompt_template_str = (
                    f"Analyze the following text and classify it into exactly one of these categories: {category_list_str_for_prompt}. "
                    f"Return only the single matching category name. If no categories match, return 'unknown'. Text: {{text}}"
                )

        if "{text}" not in final_prompt_template_str: # Should also check for {categories} if prompt needs it
            results = [{"text": str(txt), category_col_name: [] if multi_label else "error", "raw_llm_output": "Prompt template error: missing {text}"} for txt in text_series]
            return pd.DataFrame(results, columns=default_columns)

        tasks = []
        semaphore = asyncio.Semaphore(concurrency_limit) if concurrency_limit and concurrency_limit > 0 else None

        async def task_wrapper(text_input):
            if semaphore:
                async with semaphore:
                    return await self._categorize_text_single_async(
                        text_input, categories, category_list_str_for_prompt, model, final_prompt_template_str,
                        multi_label, temperature, current_use_cache, litellm_kwargs
                    )
            else:
                return await self._categorize_text_single_async(
                    text_input, categories, category_list_str_for_prompt, model, final_prompt_template_str,
                    multi_label, temperature, current_use_cache, litellm_kwargs
                )

        for text_input in text_series:
            tasks.append(task_wrapper(text_input))

        all_results_raw = await asyncio.gather(*tasks, return_exceptions=return_exceptions)

        processed_results = []
        for i, res_or_exc in enumerate(all_results_raw):
            original_text = str(text_series.iloc[i]) if pd.notna(text_series.iloc[i]) else ""
            # Determine category_col_name based on _multi_label from result if it exists, else from input
            current_multi_label_for_row = multi_label # Default to input multi_label
            if isinstance(res_or_exc, dict) and "_multi_label" in res_or_exc:
                current_multi_label_for_row = res_or_exc["_multi_label"]

            col_name_for_this_row = "categories" if current_multi_label_for_row else "category"

            if isinstance(res_or_exc, Exception):
                # Ensure the column name matches what the DataFrame expects based on the input `multi_label`
                # This part is tricky if results can have mixed multi_label states due to errors.
                # For simplicity, we'll use the original multi_label to determine error column structure.
                error_val = [] if multi_label else "error"
                processed_results.append({"text": original_text, category_col_name: error_val, "raw_llm_output": str(res_or_exc)})
            else:
                res_or_exc.pop("_multi_label", None) # Clean up temp key
                # If the helper used a different key (e.g. processed single when expecting multi), adjust.
                # This primarily handles the structure for the DataFrame, assuming `category_col_name` is the target.
                if col_name_for_this_row != category_col_name and category_col_name in res_or_exc :
                     res_or_exc[col_name_for_this_row] = res_or_exc.pop(category_col_name)
                elif col_name_for_this_row != category_col_name and col_name_for_this_row in res_or_exc:
                    # If the result has the "correct" dynamic column name, but it's not the one expected by default_columns
                    # ensure it's placed in the column name that default_columns expects.
                     res_or_exc[category_col_name] = res_or_exc.pop(col_name_for_this_row)


                processed_results.append(res_or_exc)

        if self.cache:
            try: self.cache.close()
            except Exception as e: print(f"Warning: Error closing cache: {e}")

        return pd.DataFrame(processed_results, columns=default_columns)

    async def _summarize_text_single_async(
        self,
        text_to_summarize: str,
        model: str,
        direct_prompt_template_str: str,
        chunk_prompt_template_str_in: Optional[str],
        combine_prompt_template_str_in: Optional[str],
        temperature: float,
        use_chunking: bool,
        chunk_size: int,
        chunk_overlap: int,
        current_use_cache: bool,
        litellm_kwargs: dict
    ) -> dict:
        """
        Helper coroutine to process a single text for summarization asynchronously,
        handling chunking internally and caching the final summary.

        Args:
            text_to_summarize (str): The text string to summarize.
            model (str): LiteLLM model string.
            direct_prompt_template_str (str): Prompt for direct or fallback summarization.
            chunk_prompt_template_str_in (Optional[str]): Prompt for individual chunks.
            combine_prompt_template_str_in (Optional[str]): Prompt for combining chunk summaries.
            temperature (float): LLM temperature.
            use_chunking (bool): Whether chunking is enabled.
            chunk_size (int): Size of chunks for splitting.
            chunk_overlap (int): Overlap between chunks.
            current_use_cache (bool): Whether caching is enabled for this operation.
            litellm_kwargs (dict): Additional kwargs for litellm.acompletion.

        Returns:
            dict: A dictionary containing "original_text", "summary", and "raw_llm_output".
                  If an error occurs, "summary" will be "error" or contain error indicators,
                  and "raw_llm_output" will hold error details.
        """
        original_text_for_df = str(text_to_summarize) if pd.notna(text_to_summarize) else ""
        summary_text = ""
        raw_llm_response_agg = ""

        if not original_text_for_df.strip():
            summary_text = ""
            raw_llm_response_agg = "Input text was empty or whitespace."
        else:
            cacheable_kwargs_items = sorted(
                (k, v) for k, v in litellm_kwargs.items() if k in ("temperature", "max_tokens", "top_p")
            )
            _actual_chunk_prompt = chunk_prompt_template_str_in if chunk_prompt_template_str_in else "Concisely summarize this piece of text: {text}"
            _actual_combine_prompt = combine_prompt_template_str_in

            num_chunks_for_key = 0
            if use_chunking and LANGCHAIN_SPLITTERS_AVAILABLE:
                temp_chunks_for_key = self._chunk_text(original_text_for_df, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                num_chunks_for_key = len(temp_chunks_for_key)

            cache_key_tuple = (
                "summarize_text_llm", model,
                direct_prompt_template_str,
                _actual_chunk_prompt if use_chunking else None,
                _actual_combine_prompt if use_chunking and num_chunks_for_key > 1 else None,
                original_text_for_df,
                use_chunking, chunk_size if use_chunking else None, chunk_overlap if use_chunking else None,
                tuple(cacheable_kwargs_items)
            )

            cached_result = None
            if current_use_cache and self.cache:
                try: cached_result = self.cache.get(cache_key_tuple)
                except Exception as e: print(f"Warning: Async cache get failed for summarize. Error: {e}")

            if cached_result is not None:
                summary_text = cached_result["summary"]
                raw_llm_response_agg = cached_result["raw_llm_output"]
            else:
                _final_completion_kwargs = {**litellm_kwargs}
                if 'temperature' not in _final_completion_kwargs: _final_completion_kwargs['temperature'] = temperature

                if not use_chunking or not LANGCHAIN_SPLITTERS_AVAILABLE: # Fallback to direct if chunking selected but not available
                    try:
                        messages = [{"role": "user", "content": direct_prompt_template_str.format(text=original_text_for_df)}]
                        response = await litellm.acompletion(model=model, messages=messages, **_final_completion_kwargs)
                        summary_text = response.choices[0].message.content.strip() if response.choices and response.choices[0].message else ""
                        raw_llm_response_agg = str(response)
                        if use_chunking and not LANGCHAIN_SPLITTERS_AVAILABLE : raw_llm_response_agg = f"Chunking skipped (splitters unavailable). {raw_llm_response_agg}"
                    except Exception as e:
                        summary_text, raw_llm_response_agg = "error", f"Async direct/fallback summary error: {e}"
                else: # Proceed with chunking
                    chunks = self._chunk_text(original_text_for_df, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    if not chunks:
                        summary_text, raw_llm_response_agg = "", "Text was empty or too short after attempting to chunk."
                    else:
                        chunk_summaries_list, raw_chunk_outputs_list = [], []
                        current_chunk_prompt_str = _actual_chunk_prompt
                        if "{text}" not in current_chunk_prompt_str:
                            summary_text, raw_llm_response_agg = "error_prompt_chunk", "Chunk prompt template error: missing {text}"
                        else:
                            chunk_tasks = []
                            for chunk_item_text in chunks:
                                messages_chunk = [{"role": "user", "content": current_chunk_prompt_str.format(text=chunk_item_text)}]
                                chunk_tasks.append(litellm.acompletion(model=model, messages=messages_chunk, **_final_completion_kwargs))

                            try:
                                chunk_responses = await asyncio.gather(*chunk_tasks, return_exceptions=True)
                                for i_chunk, response_chunk_or_exc in enumerate(chunk_responses):
                                    if isinstance(response_chunk_or_exc, Exception):
                                        chunk_summaries_list.append(f"[Error in chunk {i_chunk+1}]")
                                        raw_chunk_outputs_list.append(f"Chunk {i_chunk+1} Error: {str(response_chunk_or_exc)}")
                                    elif response_chunk_or_exc.choices and response_chunk_or_exc.choices[0].message:
                                        chunk_summaries_list.append(response_chunk_or_exc.choices[0].message.content.strip())
                                        raw_chunk_outputs_list.append(f"Chunk {i_chunk+1} Raw: {str(response_chunk_or_exc)}")
                                    else:
                                        chunk_summaries_list.append("")
                                        raw_chunk_outputs_list.append(f"Chunk {i_chunk+1} Raw (empty): {str(response_chunk_or_exc)}")
                            except Exception as e_gather_chunks:
                                 summary_text, raw_llm_response_agg = "error", f"Error gathering chunk summaries: {e_gather_chunks}"

                        if not summary_text.startswith("error"): # If chunk summarization part was okay
                            intermediate_summary = "\n\n".join(chunk_summaries_list)
                            raw_llm_response_agg = "\n---\n".join(raw_chunk_outputs_list)

                            if _actual_combine_prompt and len(chunks) > 1:
                                if "{text}" not in _actual_combine_prompt:
                                    summary_text = intermediate_summary
                                    raw_llm_response_agg += "\n---\nWarning: Combine prompt invalid, using intermediate."
                                else:
                                    try:
                                        messages_combine = [{"role": "user", "content": _actual_combine_prompt.format(text=intermediate_summary)}]
                                        response_combine = await litellm.acompletion(model=model, messages=messages_combine, **_final_completion_kwargs)
                                        summary_text = response_combine.choices[0].message.content.strip() if response_combine.choices and response_combine.choices[0].message else ""
                                        raw_llm_response_agg += f"\n---\nFinal Combined Summary Raw: {str(response_combine)}"
                                    except Exception as e_combine:
                                        summary_text = f"[Error combining: {intermediate_summary[:100]}...]"
                                        raw_llm_response_agg += f"\n---\nError combining summaries: {e_combine}"
                            else:
                                summary_text = intermediate_summary

                if current_use_cache and self.cache and not summary_text.startswith("error_") and not summary_text.startswith("[Error"):
                    try: self.cache.set(cache_key_tuple, {"summary": summary_text, "raw_llm_output": raw_llm_response_agg})
                    except Exception as e: print(f"Warning: Async cache set failed for summarize. Error: {e}")

        return {"original_text": original_text_for_df, "summary": summary_text, "raw_llm_output": raw_llm_response_agg}

    async def summarize_text_llm_async(
        self, text_series: pd.Series, model: str, prompt_template_str: Optional[str]=None,
        chunk_prompt_template_str: Optional[str]=None, combine_prompt_template_str: Optional[str]=None,
        temperature: float=0.0, use_chunking: bool=True, chunk_size: int=1000, chunk_overlap: int=100,
        use_cache: Optional[bool]=None, concurrency_limit: Optional[int]=None, return_exceptions: bool=True,
        max_length: Optional[int]=None, min_length: Optional[int]=None, **litellm_kwargs
    ) -> pd.DataFrame:
        """
        Asynchronously summarizes texts using LiteLLM, with support for chunking.

        Args:
            text_series (pd.Series): Series containing texts to summarize.
            model (str): The LiteLLM model string.
            prompt_template_str (Optional[str], optional): Prompt for direct summarization
                (if `use_chunking` is False or as fallback). Defaults to a standard summarization prompt.
            chunk_prompt_template_str (Optional[str], optional): Prompt for individual chunks.
            combine_prompt_template_str (Optional[str], optional): Prompt for combining chunk summaries.
            temperature (float, optional): LLM temperature. Defaults to 0.0.
            use_chunking (bool, optional): Whether to use chunking. Defaults to True.
            chunk_size (int, optional): Character size for chunks. Defaults to 1000.
            chunk_overlap (int, optional): Character overlap for chunks. Defaults to 100.
            use_cache (Optional[bool], optional): Whether to use caching. Overrides instance default.
            concurrency_limit (Optional[int], optional): Max concurrent LLM calls. Defaults to None.
            return_exceptions (bool, optional): If True, exceptions are caught. Defaults to True.
            max_length (Optional[int], optional): If provided, passed as `max_tokens` to LiteLLM,
                                                  unless `max_tokens` is in `litellm_kwargs`.
            min_length (Optional[int], optional): Currently a placeholder, not directly used.
            **litellm_kwargs: Additional arguments for `litellm.acompletion`.

        Returns:
            pd.DataFrame: DataFrame with columns ["original_text", "summary", "raw_llm_output"].
                          Failed items will have "error" in "summary".
        """
        cols = ["original_text", "summary", "raw_llm_output"]
        if not LITELLM_AVAILABLE: return pd.DataFrame(columns=cols)
        if not isinstance(text_series, pd.Series): return pd.DataFrame(columns=cols)
        if text_series.empty: return pd.DataFrame(columns=cols)

        use_cache_flag = use_cache if use_cache is not None else self.use_cache_default
        kw_main = {**litellm_kwargs}
        if 'temperature' not in kw_main: kw_main['temperature'] = temperature
        if max_length and 'max_tokens' not in kw_main: kw_main['max_tokens'] = max_length

        direct_prompt = prompt_template_str if prompt_template_str else "Please summarize the following text concisely: {text}"
        if "{text}" not in direct_prompt:
            return pd.DataFrame([{"original_text":str(t),"summary":"error_prompt_direct","raw_llm_output":"Direct prompt error"} for t in text_series], columns=cols)
        if use_chunking and chunk_prompt_template_str and "{text}" not in chunk_prompt_template_str: # Check chunk_prompt only if used
            return pd.DataFrame([{"original_text":str(t),"summary":"error_prompt_chunk","raw_llm_output":"Chunk prompt error"} for t in text_series], columns=cols)

        tasks = []
        sem = asyncio.Semaphore(concurrency_limit) if concurrency_limit and concurrency_limit > 0 else None
        async def wrapper(text):
            if sem:
                async with sem:
                    return await self._summarize_text_single_async(text, model, direct_prompt, chunk_prompt_template_str, combine_prompt_template_str, temperature, use_chunking, chunk_size, chunk_overlap, use_cache_flag, kw_main)
            else:
                return await self._summarize_text_single_async(text, model, direct_prompt, chunk_prompt_template_str, combine_prompt_template_str, temperature, use_chunking, chunk_size, chunk_overlap, use_cache_flag, kw_main)
        for txt_in in text_series: tasks.append(wrapper(txt_in))
        raw_res = await asyncio.gather(*tasks, return_exceptions=return_exceptions)

        proc_res = []
        for i, r_exc in enumerate(raw_res):
            orig_txt = str(text_series.iloc[i]) if pd.notna(text_series.iloc[i]) else ""
            if isinstance(r_exc, Exception): proc_res.append({"original_text":orig_txt, "summary":"error", "raw_llm_output":str(r_exc)})
            else: proc_res.append(r_exc)
        if self.cache:
            try: self.cache.close()
            except Exception as e: print(f"Warning: Error closing cache: {e}")
        return pd.DataFrame(proc_res, columns=cols)

    def categorize_text_llm(
        self,
        text_series: pd.Series,
        categories: List[str],
        model: str,
        prompt_template_str: Optional[str] = None,
        multi_label: bool = False,
        temperature: float = 0.0,
        use_cache: Optional[bool] = None,
        **litellm_kwargs
    ) -> pd.DataFrame:
        """
        Categorizes texts using a specified LLM via LiteLLM.

        Args:
            text_series (pd.Series): Series containing texts to categorize.
            categories (List[str]): A list of predefined category names.
            model (str): The LiteLLM model string (e.g., "openai/gpt-3.5-turbo").
            prompt_template_str (Optional[str], optional): Custom prompt template.
                Must include "{text}". Can also include "{categories}" for the list of categories,
                which will be formatted as a comma-separated string of category names.
                Defaults to a standard categorization prompt that varies based on `multi_label`.
            multi_label (bool, optional): If True, allows multiple categories per text.
                                        The output 'categories' column will be a list of strings.
                                        If False, assigns a single category per text.
                                        The output 'category' column will be a string.
                                        Defaults to False.
            temperature (float, optional): Temperature for LLM generation. Defaults to 0.0.
            use_cache (Optional[bool], optional): Whether to use caching. Defaults to instance setting.
            **litellm_kwargs: Additional keyword arguments to pass directly to `litellm.completion`.
                              This can include provider-specific parameters like `api_key`, `api_base`,
                              `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, etc.
                              Refer to LiteLLM documentation for details.
                              Example: `api_key="sk-..."`, `api_base="http://localhost:11434"`, `max_tokens=100`.

        Returns:
            pd.DataFrame: DataFrame with columns ["text", category_col_name, "raw_llm_output"].
                          `category_col_name` is "categories" (List[str]) if multi_label is True,
                          else "category" (str).
                          Categorization can be one of the provided categories, "unknown", or "error".
        """
        category_col_name = "categories" if multi_label else "category"
        default_columns = ["text", category_col_name, "raw_llm_output"]

        if not LITELLM_AVAILABLE:
            print("Warning: LiteLLM is not installed. LLM-based categorization is not available.")
            return pd.DataFrame(columns=default_columns)

        if not isinstance(text_series, pd.Series):
            print("Warning: Input 'text_series' must be a pandas Series.")
            return pd.DataFrame(columns=default_columns)
        if text_series.empty:
            return pd.DataFrame(columns=default_columns)
        if not categories or not isinstance(categories, list) or not all(isinstance(c, str) and c for c in categories):
            raise ValueError("Categories list must be a non-empty list of non-empty strings.")

        category_list_str = ", ".join(f"'{c}'" for c in categories)

        final_prompt_template_str = prompt_template_str # Use user's if provided
        if final_prompt_template_str is None:
            if multi_label:
                final_prompt_template_str = (
                    f"Analyze the following text and classify it into one or more of these categories: {category_list_str}. "
                    f"Return a comma-separated list of the matching category names. If no categories match, return 'none'. Text: {{text}}"
                )
            else:
                final_prompt_template_str = (
                    f"Analyze the following text and classify it into exactly one of these categories: {category_list_str}. "
                    f"Return only the single matching category name. If no categories match, return 'unknown'. Text: {{text}}"
                )

        # Basic check for placeholders
        if "{text}" not in final_prompt_template_str:
            print("Error: Prompt template must include '{text}' placeholder.")
            return pd.DataFrame(
                [{'text': str(txt) if pd.notna(txt) else "",
                  category_col_name: [] if multi_label else 'error',
                  'raw_llm_output': "Prompt template error: missing {text}"} for txt in text_series],
                columns=default_columns
            )
        if "{categories}" in final_prompt_template_str and not category_list_str: # Should not happen due to earlier check
             print("Error: Prompt template includes '{categories}' but no categories were provided effectively.")
             # This case should ideally be caught by `if not categories` earlier.
             # Defensive return:
             return pd.DataFrame(
                [{'text': str(txt) if pd.notna(txt) else "",
                  category_col_name: [] if multi_label else 'error',
                  'raw_llm_output': "Prompt template error: {categories} placeholder used with empty category list."} for txt in text_series],
                columns=default_columns
            )


        results = []
        for text_input in text_series:
            original_text_for_df = str(text_input) if pd.notna(text_input) else ""
            raw_llm_response_content = ""
            parsed_categories: Any = [] if multi_label else "unknown"

            if not original_text_for_df.strip():
                raw_llm_response_content = "Input text was empty or whitespace."
                # parsed_categories remains as default (empty list or "unknown")
            else:
                # --- Cache Key Generation ---
                cacheable_kwargs_items = sorted(
                    (k, v) for k, v in litellm_kwargs.items()
                    if k in ("temperature", "max_tokens", "top_p")
                )
                cache_key_tuple = (
                    "categorize_text_llm", model, final_prompt_template_str,
                    original_text_for_df, tuple(sorted(categories)), multi_label, # Add categories and multi_label
                    tuple(cacheable_kwargs_items)
                )

                cached_result = None
                current_use_cache_for_call = use_cache if use_cache is not None else self.use_cache_default
                if current_use_cache_for_call and self.cache:
                    try:
                        cached_result = self.cache.get(cache_key_tuple)
                    except Exception as e:
                        print(f"Warning: Cache get failed for key {cache_key_tuple}. Error: {e}")

                if cached_result is not None:
                    parsed_categories = cached_result["categories_result"]
                    raw_llm_response_content = cached_result["raw_llm_output"]
                    # print(f"Cache HIT for categorize: {original_text_for_df[:30]}...") # For debugging
                else:
                    # print(f"Cache MISS for categorize: {original_text_for_df[:30]}...") # For debugging
                    try:
                        prompt_to_format = final_prompt_template_str
                        format_args = {"text": original_text_for_df}
                        if "{categories}" in prompt_to_format:
                            format_args["categories"] = category_list_str
                        formatted_content_for_llm = prompt_to_format.format(**format_args)
                        messages = [{"role": "user", "content": formatted_content_for_llm}]

                        completion_kwargs = {k: v for k, v in litellm_kwargs.items()}
                        if 'temperature' not in completion_kwargs:
                            completion_kwargs['temperature'] = temperature

                        response = litellm.completion(model=model, messages=messages, **completion_kwargs)

                        if response.choices and response.choices[0].message:
                            raw_llm_response_content = response.choices[0].message.content or ""
                        else:
                            raw_llm_response_content = str(response)

                        processed_output = raw_llm_response_content.strip().lower()
                        if multi_label:
                            found_cats_raw = [cat.strip() for cat in processed_output.split(',')]
                            parsed_categories = [
                                original_cat for original_cat in categories
                                if original_cat.lower() in [fcr.lower() for fcr in found_cats_raw if fcr and fcr != 'none']
                            ]
                        else:
                            parsed_categories = "unknown"
                            exact_match_found = False
                            for cat_original_case in categories:
                                if cat_original_case.lower() == processed_output:
                                    parsed_categories = cat_original_case
                                    exact_match_found = True; break
                            if not exact_match_found:
                                for cat_original_case in categories:
                                    if cat_original_case.lower() in processed_output:
                                        parsed_categories = cat_original_case; break
                            if parsed_categories == "unknown" and processed_output in ["unknown", "none"]: pass

                        if current_use_cache_for_call and self.cache:
                            try:
                                self.cache.set(cache_key_tuple, {"categories_result": parsed_categories, "raw_llm_output": raw_llm_response_content})
                            except Exception as e:
                                print(f"Warning: Cache set failed for key {cache_key_tuple}. Error: {e}")

                    except ImportError:
                        parsed_categories = [] if multi_label else "error"; raw_llm_response_content = "LiteLLM not installed."
                        print("LiteLLM is not installed. Cannot perform categorization.")
                    except litellm.exceptions.AuthenticationError as e:
                        parsed_categories = [] if multi_label else "error"; raw_llm_response_content = f"AuthenticationError: {e}"; print(f"LiteLLM Authentication Error: {e}")
                    except litellm.exceptions.APIConnectionError as e:
                        parsed_categories = [] if multi_label else "error"; raw_llm_response_content = f"APIConnectionError: {e}"; print(f"LiteLLM API Connection Error: {e}")
                    except litellm.exceptions.RateLimitError as e:
                        parsed_categories = [] if multi_label else "error"; raw_llm_response_content = f"RateLimitError: {e}"; print(f"LiteLLM Rate Limit Error: {e}")
                    except Exception as e:
                        parsed_categories = [] if multi_label else "error"; raw_llm_response_content = str(e); print(f"Error categorizing text '{original_text_for_df[:50]}...': {e}")

            results.append({"text": original_text_for_df, category_col_name: parsed_categories, "raw_llm_output": raw_llm_response_content})

        if self.cache: # Close cache if opened by this instance
            try: self.cache.close()
            except Exception as e: print(f"Warning: Error closing cache: {e}")

        return pd.DataFrame(results, columns=default_columns)

    def _chunk_text(
        self,
        text_to_chunk: str,
        strategy: str = "recursive_char",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        **splitter_kwargs
    ) -> List[str]:
        """
        (TDD Cycle 4 - Stub/Initial Implementation)
        Splits a long text into smaller chunks using Langchain TextSplitters.
        """
        if not LANGCHAIN_SPLITTERS_AVAILABLE:
            print("Warning: Langchain text splitter components are not installed. Chunking will not be performed; returning original text as a single chunk.")
            return [text_to_chunk] if text_to_chunk else [] # Return list for consistency

        if not text_to_chunk: # Handles empty string, None, etc.
            return []

        # --- Actual implementation will follow ---
        # Placeholder to make the "method exists" test pass and "basic functionality" tests fail.
        # For example, a simple split that doesn't use Langchain or proper chunking yet.
        # Or just return the original text in a list if it's short enough based on chunk_size.

        # Simplified stub logic for now:
        # if len(text_to_chunk) > chunk_size:
        #     # This is a very naive split, just to have something that might change based on input
        #     # and allow tests for real splitter logic to fail.
        #     return [text_to_chunk[:chunk_size], text_to_chunk[chunk_size:]]
        # else:
        # return [text_to_chunk]

        # Per the original stub's intention for TDD:
        # "For now, the stub returns the original text as a single chunk if not empty.
        # This will make tests for actual chunking fail as expected (Red state for functionality)."
        # print(f"_chunk_text called with strategy: {strategy}, chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}, splitter_kwargs: {splitter_kwargs}") # Debug print
        # return [text_to_chunk] # Original stub logic

        # (TDD Cycle 4 - Green Phase Implementation)
        if strategy == "recursive_char":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len, # Default, but explicit
                **splitter_kwargs
            )
        elif strategy == "character":
            # Default separator if not provided in splitter_kwargs
            separator = splitter_kwargs.pop("separator", "\n\n")
            splitter = CharacterTextSplitter(
                separator=separator,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len, # Default, but explicit
                **splitter_kwargs # Pass remaining kwargs
            )
        # Add more strategies here if needed, e.g., "sentence_transformers_token"
        # elif strategy == "sentence_transformers_token":
        #     from langchain.text_splitter import SentenceTransformersTokenTextSplitter
        #     splitter = SentenceTransformersTokenTextSplitter(
        #         chunk_overlap=chunk_overlap, # chunk_size is often model-specific for tokenizers
        #         **splitter_kwargs
        #     )
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}. Supported strategies: 'recursive_char', 'character'.")

        try:
            return splitter.split_text(text_to_chunk)
        except Exception as e:
            print(f"Error during text chunking with strategy '{strategy}': {e}")
            # Fallback: return original text as a single chunk if splitting fails
            return [text_to_chunk]

    def summarize_text_llm(
        self,
        text_series: pd.Series,
        model: str, # Unified model string for LiteLLM
        prompt_template_str: Optional[str] = None,
        chunk_prompt_template_str: Optional[str] = None, # For individual chunks
        combine_prompt_template_str: Optional[str] = None, # For combining summaries of chunks
        max_length: Optional[int] = None, # Placeholder for max summary length if LLM supports
        min_length: Optional[int] = None, # Placeholder for min summary length
        temperature: float = 0.0,
        use_chunking: bool = True, # Whether to use chunking for long texts
        chunk_size: int = 1000,    # Default chunk size for summarization
        chunk_overlap: int = 100,  # Default chunk overlap
        use_cache: Optional[bool] = None, # Added for method-specific cache control
        **litellm_kwargs
    ) -> pd.DataFrame:
        """
        Summarizes texts using a specified LLM via LiteLLM.
        Can handle long texts by chunking, summarizing parts, and optionally combining those summaries.

        Args:
            text_series (pd.Series): Series containing texts to summarize.
            model (str): The LiteLLM model string (e.g., "openai/gpt-3.5-turbo", "ollama/llama2", "azure/your-deployment").
            prompt_template_str (Optional[str], optional): Prompt template for summarizing
                texts directly (when `use_chunking` is False). Must include "{text}".
                Defaults to "Please summarize the following text concisely: {text}".
            chunk_prompt_template_str (Optional[str], optional): Prompt template for
                summarizing individual chunks (when `use_chunking` is True). Must include "{text}".
                Defaults to "Summarize this text: {text}".
            combine_prompt_template_str (Optional[str], optional): Prompt template for
                combining summaries of multiple chunks into a final summary
                (when `use_chunking` is True and more than one chunk exists). Must include "{text}"
                which will be filled with the concatenation of chunk summaries.
                If None, chunk summaries are simply joined with newlines. Defaults to None.
            max_length (Optional[int], optional): Intended to influence the maximum length of the generated summary.
                If provided, it is passed as `max_tokens` to `litellm.completion` unless `max_tokens` is already
                present in `litellm_kwargs`. Note that the exact behavior is LLM provider-dependent.
                It is generally recommended to use `max_tokens` directly within `litellm_kwargs` for more explicit control.
                Defaults to None.
            min_length (Optional[int], optional): Placeholder for desired minimum summary length.
                Currently not directly enforced by this method or easily mapped to a generic LiteLLM parameter.
                Behavior is LLM provider-dependent if such a parameter exists for the chosen model.
                Defaults to None.
            temperature (float, optional): Temperature for LLM generation. Defaults to 0.0.
            use_chunking (bool, optional): Whether to split long texts into chunks for summarization.
                Defaults to True.
            chunk_size (int, optional): The character size for each text chunk when `use_chunking` is True.
                Defaults to 1000.
            chunk_overlap (int, optional): The character overlap between text chunks when `use_chunking` is True.
                Defaults to 100.
            use_cache (Optional[bool], optional): Controls caching for this call.
            **litellm_kwargs: Additional keyword arguments to pass directly to `litellm.completion`.
                              This can include provider-specific parameters like `api_key`, `api_base`,
                              `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, etc.
                              Refer to LiteLLM documentation for details.
                              Example: `api_key="sk-..."`, `api_base="http://localhost:11434"`, `max_tokens=150`.

        Returns:
            pd.DataFrame: DataFrame with columns ["original_text", "summary", "raw_llm_output"].
                          "summary" contains the generated summary, or "error" / "error_prompt_..."
                          if an issue occurred. "raw_llm_output" contains the direct output from
                          the LLM and any intermediate outputs or error messages from chunking/combining.
        """
        default_columns = ["original_text", "summary", "raw_llm_output"]
        if not LITELLM_AVAILABLE: # Check for LiteLLM
            print("Warning: LiteLLM is not installed. LLM-based summarization is not available.")
            return pd.DataFrame(columns=default_columns)

        # LANGCHAIN_SPLITTERS_AVAILABLE check is for _chunk_text, which will raise its own error if needed.
        # No need to check it here directly unless we stop using _chunk_text.

        if not isinstance(text_series, pd.Series):
            print("Warning: Input 'text_series' must be a pandas Series.")
            return pd.DataFrame(columns=default_columns)
        if text_series.empty:
            return pd.DataFrame(columns=default_columns)

        if text_series.empty:
            return pd.DataFrame(columns=default_columns)

        current_use_cache_for_call = use_cache if use_cache is not None else self.use_cache_default

        _litellm_kwargs = {**litellm_kwargs} # Use a local copy
        if 'temperature' not in _litellm_kwargs:
            _litellm_kwargs['temperature'] = temperature
        if max_length is not None and 'max_tokens' not in _litellm_kwargs:
             _litellm_kwargs['max_tokens'] = max_length

        final_direct_prompt_template_str = prompt_template_str if prompt_template_str else "Please summarize the following text concisely: {text}"
        if "{text}" not in final_direct_prompt_template_str:
            print("Error: Direct summarization prompt template must include '{text}' placeholder.")
            return pd.DataFrame(
                [{'original_text': str(txt) if pd.notna(txt) else "", 'summary': 'error_prompt_direct', 'raw_llm_output': "Direct prompt error"} for txt in text_series],
                columns=default_columns
            )

        results = []
        for text_input in text_series:
            original_text_for_df = str(text_input) if pd.notna(text_input) else ""
            summary_text = ""
            raw_llm_response_agg = "" # Aggregate raw responses

            if not original_text_for_df.strip():
                summary_text = ""
                raw_llm_response_agg = "Input text was empty or whitespace."
            else:
                # --- Cache Key Generation ---
                cacheable_kwargs_items = sorted(
                    (k, v) for k, v in _litellm_kwargs.items()
                    if k in ("temperature", "max_tokens", "top_p")
                )
                _effective_chunk_prompt = chunk_prompt_template_str if chunk_prompt_template_str else "Concisely summarize this piece of text: {text}"
                _effective_combine_prompt = combine_prompt_template_str

                # Estimate number of chunks for cache key consistency if combine_prompt is used
                num_chunks_for_key = 0
                if use_chunking and LANGCHAIN_SPLITTERS_AVAILABLE:
                    temp_chunks_for_key = self._chunk_text(original_text_for_df, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    num_chunks_for_key = len(temp_chunks_for_key)

                cache_key_tuple = (
                    "summarize_text_llm", model,
                    final_direct_prompt_template_str,
                    _effective_chunk_prompt if use_chunking else None,
                    _effective_combine_prompt if use_chunking and num_chunks_for_key > 1 else None,
                    original_text_for_df,
                    use_chunking, chunk_size if use_chunking else None, chunk_overlap if use_chunking else None,
                    tuple(cacheable_kwargs_items)
                )

                cached_result = None
                if current_use_cache_for_call and self.cache:
                    try:
                        cached_result = self.cache.get(cache_key_tuple)
                    except Exception as e:
                        print(f"Warning: Cache get failed for key {cache_key_tuple}. Error: {e}")

                if cached_result is not None:
                    summary_text = cached_result["summary"]
                    raw_llm_response_agg = cached_result["raw_llm_output"]
                else:
                    if not use_chunking:
                        try:
                            messages = [{"role": "user", "content": final_direct_prompt_template_str.format(text=original_text_for_df)}]
                            response = litellm.completion(model=model, messages=messages, **_litellm_kwargs)
                            summary_text = response.choices[0].message.content.strip() if response.choices and response.choices[0].message else ""
                            raw_llm_response_agg = str(response)
                        except Exception as e:
                            print(f"Error during direct summarization for text '{original_text_for_df[:50]}...': {e}")
                            summary_text, raw_llm_response_agg = "error", f"Direct summarization error: {e}"
                    else: # use_chunking is True
                        if not LANGCHAIN_SPLITTERS_AVAILABLE:
                            print("Warning: Langchain text splitters not available. Attempting direct summarization.")
                            try:
                                messages = [{"role": "user", "content": final_direct_prompt_template_str.format(text=original_text_for_df)}]
                                response = litellm.completion(model=model, messages=messages, **_litellm_kwargs)
                                summary_text = response.choices[0].message.content.strip() if response.choices and response.choices[0].message else ""
                                raw_llm_response_agg = f"Chunking skipped. Direct summary raw: {str(response)}"
                            except Exception as e:
                                print(f"Error during fallback direct summarization: {e}"); summary_text, raw_llm_response_agg = "error", f"Fallback direct summarization error: {e}"
                        else:
                            chunks = self._chunk_text(original_text_for_df, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                            if not chunks:
                                summary_text, raw_llm_response_agg = "", "Text too short after chunking."
                            else:
                                chunk_summaries_list, raw_chunk_outputs_list = [], []
                                current_chunk_prompt = _effective_chunk_prompt
                                if "{text}" not in current_chunk_prompt: # Should be caught by main async method too
                                    summary_text, raw_llm_response_agg = "error_prompt_chunk", "Chunk prompt error"
                                    # This early exit was missing, adding it:
                                    results.append({"original_text": original_text_for_df, "summary": summary_text, "raw_llm_output": raw_llm_response_agg}); continue


                                for i, chunk_item_text in enumerate(chunks):
                                    try:
                                        messages_chunk = [{"role": "user", "content": current_chunk_prompt.format(text=chunk_item_text)}]
                                        response_chunk = litellm.completion(model=model, messages=messages_chunk, **_litellm_kwargs)
                                        chunk_summary = response_chunk.choices[0].message.content.strip() if response_chunk.choices and response_chunk.choices[0].message else ""
                                        chunk_summaries_list.append(chunk_summary)
                                        raw_chunk_outputs_list.append(f"Chunk {i+1}/{len(chunks)} Raw: {str(response_chunk)}")
                                    except Exception as e_chunk:
                                        chunk_summaries_list.append(f"[Err chunk {i+1}]"); raw_chunk_outputs_list.append(f"Chunk {i+1} Err: {e_chunk}")

                                intermediate_summary = "\n\n".join(chunk_summaries_list)
                                raw_llm_response_agg = "\n---\n".join(raw_chunk_outputs_list)

                                if _effective_combine_prompt and len(chunks) > 1:
                                    if "{text}" not in _effective_combine_prompt:
                                        summary_text = intermediate_summary
                                        raw_llm_response_agg += "\n---\nWarn: Combine prompt invalid."
                                    else:
                                        try:
                                            messages_combine = [{"role": "user", "content": _effective_combine_prompt.format(text=intermediate_summary)}]
                                            response_combine = litellm.completion(model=model, messages=messages_combine, **_litellm_kwargs)
                                            summary_text = response_combine.choices[0].message.content.strip() if response_combine.choices and response_combine.choices[0].message else ""
                                            raw_llm_response_agg += f"\n---\nCombined Raw: {str(response_combine)}"
                                        except Exception as e_combine:
                                            summary_text = f"[Err combine: {intermediate_summary[:100]}...]"; raw_llm_response_agg += f"\n---\nErr combine: {e_combine}"
                                else:
                                    summary_text = intermediate_summary

                    if current_use_cache_for_call and self.cache and not summary_text.startswith("error_") and not summary_text.startswith("[Err"):
                        try:
                            self.cache.set(cache_key_tuple, {"summary": summary_text, "raw_llm_output": raw_llm_response_agg})
                        except Exception as e:
                            print(f"Warning: Cache set failed for key {cache_key_tuple}. Error: {e}")

            results.append({
                "original_text": original_text_for_df,
                "summary": summary_text,
                "raw_llm_output": raw_llm_response_agg
            })

        if self.cache:
            try: self.cache.close()
            except Exception as e: print(f"Warning: Error closing cache: {e}")

        return pd.DataFrame(results, columns=default_columns)


    def plot_japanese_text_features(
        self,
        features_df: pd.DataFrame,
        target_feature: str,
        title: Optional[str] = None,
        save: bool = False,
        **kwargs
    ) -> Optional[plotly.graph_objs.Figure]:
        # Temporarily removing docstring to isolate syntax error
        if not isinstance(features_df, pd.DataFrame) or features_df.empty:
            raise ValueError("Input DataFrame 'features_df' is empty or not a DataFrame.")

        if target_feature not in features_df.columns:
            raise ValueError(f"Target feature '{target_feature}' not found in DataFrame.")

        try:
            numeric_feature_series = pd.to_numeric(features_df[target_feature], errors='coerce')
        except Exception as e:
            raise ValueError(f"Column '{target_feature}' could not be converted to numeric due to: {e}")

        if not pd.api.types.is_numeric_dtype(numeric_feature_series) or numeric_feature_series.isnull().all():
            raise ValueError(f"Column '{target_feature}' is not numeric, contains only NaN values, or could not be coerced to numeric for plotting.")

        plot_title = title if title else f"Distribution of {target_feature}"

        hist_kwargs = { "x": target_feature, "title": plot_title, "marginal": "box" }
        df_to_plot = features_df.copy()
        df_to_plot[target_feature] = numeric_feature_series
        hist_kwargs.update(kwargs)

        try:
            fig = px.histogram(df_to_plot, **hist_kwargs)
            fig.update_layout(xaxis_title=target_feature, yaxis_title="Frequency")
        except Exception as e:
            print(f"Error during histogram generation for feature '{target_feature}': {e}")
            return None

        if save:
            filename_prefix = f"jp_feature_{target_feature.replace(' ','_').replace('/','_')}"
            self.save_plot(fig, filename_prefix)

        return fig

# Example usage (for testing or direct script execution):
if __name__ == '__main__':
    # Create a sample DataFrame for NLPlotLLM initialization
    sample_df_main = pd.DataFrame({
        'text_column': [
            "This is the first test text for basic functionality.",
            "Another example sentence to see how NLPlotLLM handles it."
        ]
    })
    npt_main = NLPlotLLM(sample_df_main, target_col='text_column')

    # --- Test Sentiment Analysis ---
    print("\\n--- Testing Sentiment Analysis ---")
    sentiment_texts = pd.Series([
        "I love this product, it's absolutely fantastic!",
        "This is the worst experience I have ever had.",
        "The weather today is just okay, nothing special."
    ])
    if LITELLM_AVAILABLE:
        try:
            # Test with OpenAI (requires OPENAI_API_KEY env var or passed in litellm_kwargs)
            # sentiment_results_openai = npt_main.analyze_sentiment_llm(
            #     sentiment_texts, model="openai/gpt-3.5-turbo"
            # )
            # print("\\nOpenAI Sentiment Results:")
            # print(sentiment_results_openai)

            # Test with Ollama (requires Ollama server running with 'llama2' model)
            # Make sure to start your Ollama server: `ollama serve`
            # And pull a model if you haven't: `ollama pull llama2`
            print("\\nAttempting Ollama Sentiment Analysis (ensure Ollama is running)...")
            sentiment_results_ollama = npt_main.analyze_sentiment_llm(
                 sentiment_texts, model="ollama/llama2" # or another model like "ollama/mistral"
            )
            print("\\nOllama Sentiment Results:")
            print(sentiment_results_ollama)

        except Exception as e:
            print(f"Error during sentiment analysis test: {e}")
    else:
        print("Skipping sentiment analysis test as LiteLLM is not available.")

    # --- Test Text Categorization ---
    print("\\n--- Testing Text Categorization ---")
    categorization_texts = pd.Series([
        "The new iPhone 15 Pro Max features a titanium design and A17 Bionic chip.",
        "Manchester United won the match with a last-minute goal.",
        "The Federal Reserve announced an increase in interest rates to combat inflation."
    ])
    categories_list = ["technology", "sports", "finance", "health"]
    if LITELLM_AVAILABLE:
        try:
            # Test with Ollama
            print("\\nAttempting Ollama Categorization (ensure Ollama is running)...")
            categorization_results_ollama = npt_main.categorize_text_llm(
                categorization_texts, categories_list, model="ollama/llama2", multi_label=True
            )
            print("\\nOllama Categorization Results (Multi-label):")
            print(categorization_results_ollama)
        except Exception as e:
            print(f"Error during categorization test: {e}")
    else:
        print("Skipping categorization test as LiteLLM is not available.")

    # --- Test Text Summarization ---
    print("\\n--- Testing Text Summarization ---")
    summarization_texts = pd.Series([
        "The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. Modern AI predicates on the formalization of reasoning by philosophers in the first millennium BCE. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.",
        "The quick brown fox jumps over the lazy dog. This sentence is famous because it contains all the letters of the English alphabet. It is often used for testing typewriters and keyboards. The origin of the sentence is not entirely clear, but it has been in use for a long time. It's a classic pangram."
    ])
    if LITELLM_AVAILABLE:
        try:
            # Test with Ollama (chunking enabled by default)
            print("\\nAttempting Ollama Summarization with Chunking (ensure Ollama is running)...")
            summarization_results_ollama_chunked = npt_main.summarize_text_llm(
                summarization_texts, model="ollama/llama2",
                chunk_size=200, chunk_overlap=50 # Smaller chunks for testing
            )
            print("\\nOllama Summarization Results (Chunked):")
            print(summarization_results_ollama_chunked)

            # Test direct summarization (no chunking)
            print("\\nAttempting Ollama Summarization (Direct, No Chunking)...")
            summarization_results_ollama_direct = npt_main.summarize_text_llm(
                pd.Series([summarization_texts.iloc[1]]), # Take a shorter one for direct
                model="ollama/llama2",
                use_chunking=False
            )
            print("\\nOllama Summarization Results (Direct):")
            print(summarization_results_ollama_direct)

        except Exception as e:
            print(f"Error during summarization test: {e}")
    else:
        print("Skipping summarization test as LiteLLM is not available.")

    print("\\n--- NLPlotLLM Basic Tests Complete ---")
    # Example of a traditional nlplot feature
    # npt_main.bar_ngram(title="Sample N-gram") # This would require target_col to be list of words
    # For this to work, the constructor logic for splitting string target_col needs to be active
    # or the input df must already have tokenized lists.
    # Let's re-initialize with a string that will be split by the constructor for this test.
    sample_df_for_plot = pd.DataFrame({'text_plot': ["this is a test text for plotting", "another plot example text here"]})
    npt_plotter = NLPlotLLM(sample_df_for_plot, target_col='text_plot')
    if not npt_plotter.df[npt_plotter.target_col].empty:
         print("\\n--- Testing a traditional plotting feature (N-gram Bar Chart) ---")
         fig = npt_plotter.bar_ngram(title="Sample N-gram from NLPlotLLM")
         # In a script, fig.show() or saving might be needed. Here, just checking it runs.
         if fig: print("N-gram bar chart generated.")
         else: print("N-gram bar chart generation failed or returned empty.")
    else:
        print("DataFrame for plotting is empty, skipping bar_ngram test.")
