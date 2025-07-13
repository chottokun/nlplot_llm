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
from typing import Optional, List, Any, Dict # Added Dict
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


from .utils.common import (
    _ranked_topics_for_edges,
    _unique_combinations_for_edges,
    _add_unique_combinations_to_dict,
    get_colorpalette,
    generate_freq_df
)

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

        if not self.df.empty and self.target_col in self.df.columns:
            first_item = self.df[self.target_col].iloc[0]
            # Process if the first item is a string and not already a list (for auto-splitting)
            if pd.notna(first_item):
                # This check ensures we only try to split if it's a non-NA value.
                # Convert to string and split for any type, to ensure list of tokens.
                self.df.loc[:, self.target_col] = self.df[self.target_col].astype(str).map(lambda x: x.split())
            # If first_item is already a list,
            # we assume it's either correctly pre-processed or not intended for splitting here.
        elif self.df.empty and self.target_col not in self.df.columns :
             print(f"Warning: DataFrame is empty and target column '{self.target_col}' not found. Initializing with an empty column.")
             self.df = pd.DataFrame({self.target_col: pd.Series([], dtype=object)})
        elif self.df.empty :
             print(f"Warning: DataFrame is empty after processing. Target column '{self.target_col}' might be present but with no data.")
             # Ensure column exists even if df is empty, matching constructor expectation for plotting methods
             if self.target_col not in self.df.columns:
                 self.df[self.target_col] = pd.Series([], dtype=object)


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

    def bar_ngram(self, *args, **kwargs):
        from .plot.standard import bar_ngram
        return bar_ngram(self, *args, **kwargs)

    def treemap(self, *args, **kwargs):
        from .plot.standard import treemap
        return treemap(self, *args, **kwargs)

    def word_distribution(self, *args, **kwargs):
        from .plot.standard import word_distribution
        return word_distribution(self, *args, **kwargs)

    def wordcloud(self, *args, **kwargs):
        from .plot.standard import wordcloud
        return wordcloud(self, *args, **kwargs)

    def get_tfidf_top_features(self, *args, **kwargs):
        from .utils.text import get_tfidf_top_features
        return get_tfidf_top_features(self, *args, **kwargs)

    def get_kwic_results(self, *args, **kwargs):
        from .utils.text import get_kwic_results
        return get_kwic_results(self, *args, **kwargs)

    def get_edges_nodes(self, *args, **kwargs):
        from .network.graph import get_edges_nodes
        return get_edges_nodes(self, *args, **kwargs)

    def get_graph(self, *args, **kwargs):
        from .network.graph import get_graph
        return get_graph(self, *args, **kwargs)

    def build_graph(self, *args, **kwargs):
        from .network.graph import build_graph
        return build_graph(self, *args, **kwargs)

    def _prepare_data_for_graph(self, *args, **kwargs):
        from .network.graph import _prepare_data_for_graph
        return _prepare_data_for_graph(self, *args, **kwargs)

    def _initialize_empty_graph_attributes(self, *args, **kwargs):
        from .network.graph import _initialize_empty_graph_attributes
        return _initialize_empty_graph_attributes(self, *args, **kwargs)

    def _calculate_graph_metrics(self, *args, **kwargs):
        from .network.graph import _calculate_graph_metrics
        return _calculate_graph_metrics(self, *args, **kwargs)

    def _detect_communities(self, *args, **kwargs):
        from .network.graph import _detect_communities
        return _detect_communities(self, *args, **kwargs)

    def _create_network_trace(self, *args, **kwargs):
        from .network.graph import _create_network_trace
        return _create_network_trace(self, *args, **kwargs)

    def co_network(self, *args, **kwargs):
        from .network.graph import co_network
        return co_network(self, *args, **kwargs)

    def _calculate_node_sizes(self, *args, **kwargs):
        from .network.graph import _calculate_node_sizes
        return _calculate_node_sizes(self, *args, **kwargs)

    def sunburst(self, *args, **kwargs):
        from .network.graph import sunburst
        return sunburst(self, *args, **kwargs)

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
    def analyze_sentiment_llm(self, *args, **kwargs):
        from .llm.sentiment import analyze_sentiment_llm
        return analyze_sentiment_llm(self, *args, **kwargs)

    async def analyze_sentiment_llm_async(self, *args, **kwargs):
        from .llm.sentiment import analyze_sentiment_llm_async
        return await analyze_sentiment_llm_async(self, *args, **kwargs)

    def categorize_text_llm(self, *args, **kwargs):
        from .llm.categorize import categorize_text_llm
        return categorize_text_llm(self, *args, **kwargs)

    async def categorize_text_llm_async(self, *args, **kwargs):
        from .llm.categorize import categorize_text_llm_async
        return await categorize_text_llm_async(self, *args, **kwargs)

    def summarize_text_llm(self, *args, **kwargs):
        from .llm.summarize import summarize_text_llm
        return summarize_text_llm(self, *args, **kwargs)

    async def summarize_text_llm_async(self, *args, **kwargs):
        from .llm.summarize import summarize_text_llm_async
        return await summarize_text_llm_async(self, *args, **kwargs)


    def plot_japanese_text_features(self, *args, **kwargs):
        from .plot.standard import plot_japanese_text_features
        return plot_japanese_text_features(self, *args, **kwargs)

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
