"""Visualization Module for Natural Language Processing"""

import os
import datetime as datetime
from collections import Counter
from typing import Optional, List

import pandas as pd
import plotly


# Default path for Japanese Font, can be overridden
DEFAULT_FONT_PATH = None

try:
    from janome.tokenizer import Tokenizer as JanomeTokenizer

    JANOME_AVAILABLE = True
except ImportError:
    JANOME_AVAILABLE = False
    # Dummy class for JanomeTokenizer if not installed

    class JanomeTokenizer:
        def tokenize(self, text: str, stream=False, wakati=False):
            return []


# LiteLLM and Langchain TextSplitters
try:
    import litellm
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        CharacterTextSplitter,
    )

    LITELLM_AVAILABLE = True
    LANGCHAIN_SPLITTERS_AVAILABLE = True
    import diskcache

    DISKCACHE_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    LANGCHAIN_SPLITTERS_AVAILABLE = False
    DISKCACHE_AVAILABLE = False
    # Dummy class for litellm if not installed

    class litellm_dummy:
        def completion(self, *args, **kwargs):
            raise ImportError("litellm is not installed.")

        class exceptions:
            class APIConnectionError(Exception):
                pass

            class AuthenticationError(Exception):
                pass

            class RateLimitError(Exception):
                pass

    litellm = litellm_dummy()

    # Dummy splitters if Langchain splitters are not available
    class RecursiveCharacterTextSplitter:
        def __init__(self, *args, **kwargs):
            pass

        def split_text(self, text: str) -> List[str]:
            return [text] if text else []

    class CharacterTextSplitter:
        def __init__(self, *args, **kwargs):
            pass

        def split_text(self, text: str) -> List[str]:
            return [text] if text else []


class NLPlotLLM:
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        output_file_path: str = "./",
        default_stopwords_file_path: str = "",
        font_path: Optional[str] = None,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        cache_expire: Optional[int] = None,
        cache_size_limit: int = 1_000_000_000,
    ):
        """
        Initializes the NLPlotLLM instance.

        Args:
            df (pd.DataFrame): The DataFrame containing the text data.
            target_col (str): The name of the column with text data.
            output_file_path (str): Path to save plots and tables.
            default_stopwords_file_path (str): Path to a stopwords file.
            font_path (Optional[str]): Path to a .ttf font file.
            use_cache (bool): Enable/disable caching for LLM responses.
            cache_dir (Optional[str]): Directory for cache files.
            cache_expire (Optional[int]): Cache expiration in seconds.
            cache_size_limit (int): Cache size limit in bytes.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.df.dropna(subset=[self.target_col], inplace=True)

        if not self.df.empty and self.target_col in self.df.columns:
            first_item = self.df[self.target_col].iloc[0]
            if not isinstance(first_item, list) and pd.notna(first_item):
                self.df[self.target_col] = (
                    self.df[self.target_col].astype(str).map(lambda x: x.split())
                )
        elif self.df.empty:
            print(
                "Warning: DataFrame is empty. Target column "
                f"'{self.target_col}' might be missing or has no data."
            )
            if self.target_col not in self.df.columns:
                self.df[self.target_col] = pd.Series([], dtype=object)

        self.output_file_path = output_file_path
        self.font_path = self._validate_font_path(font_path)

        self.default_stopwords = self._load_stopwords(default_stopwords_file_path)

        self._janome_tokenizer = None
        if JANOME_AVAILABLE:
            try:
                self._janome_tokenizer = JanomeTokenizer()
            except Exception as e:
                print(
                    "Warning: Failed to initialize Janome Tokenizer. "
                    "Japanese text features may not be available. "
                    f"Error: {e}"
                )

        self.use_cache_default = use_cache
        self.cache = self._initialize_cache(cache_dir, cache_size_limit, cache_expire)

    def _validate_font_path(self, font_path: Optional[str]) -> Optional[str]:
        if font_path and os.path.exists(font_path):
            return font_path
        if font_path:
            print(
                f"Warning: Specified font_path '{font_path}' not found. "
                "Falling back to default."
            )
        if DEFAULT_FONT_PATH and os.path.exists(DEFAULT_FONT_PATH):
            return DEFAULT_FONT_PATH
        if DEFAULT_FONT_PATH:
            print(
                f"Warning: Default font path '{DEFAULT_FONT_PATH}' not found. "
                "WordCloud may fail."
            )
        return None

    def _load_stopwords(self, filepath: str) -> list:
        if not filepath or not os.path.exists(filepath):
            return []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except (IOError, PermissionError) as e:
            print(
                f"Warning: Could not read stopwords file '{filepath}': {e}. "
                "Continuing without default stopwords."
            )
        return []

    def _initialize_cache(self, cache_dir, size_limit, expire):
        if not (self.use_cache_default and DISKCACHE_AVAILABLE):
            if self.use_cache_default:
                print(
                    "Warning: diskcache not installed. Caching disabled. "
                    "Please install with: pip install diskcache"
                )
            return None

        cache_path = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "nlplot_llm"
        )
        try:
            cache = diskcache.Cache(cache_path, size_limit=size_limit, expire=expire)
            print(f"NLPlotLLM cache initialized at: {cache_path}")
            return cache
        except Exception as e:
            print(
                f"Warning: Failed to initialize diskcache at {cache_path}. "
                f"Cache disabled. Error: {e}"
            )
            self.use_cache_default = False
            return None

    def _tokenize_japanese_text(self, text: str) -> list:
        if not JANOME_AVAILABLE:
            print(
                "Warning: Janome not installed. Japanese tokenization "
                "unavailable. Please install with: pip install janome"
            )
            return []
        if self._janome_tokenizer is None:
            print(
                "Warning: Janome Tokenizer not available. "
                "Tokenization cannot be performed."
            )
            return []
        if not isinstance(text, str) or not text.strip():
            return []
        try:
            return list(self._janome_tokenizer.tokenize(text, wakati=True))
        except Exception as e:
            print(
                f"Error during Janome tokenization for text '{text[:30]}...':" f" {e}"
            )
            return []

    def get_japanese_text_features(
        self, japanese_text_series: pd.Series
    ) -> pd.DataFrame:
        expected_columns = [
            "text",
            "total_tokens",
            "avg_token_length",
            "noun_ratio",
            "verb_ratio",
            "adj_ratio",
            "punctuation_count",
        ]
        if not isinstance(japanese_text_series, pd.Series):
            print("Warning: Input must be a pandas Series.")
            return pd.DataFrame(columns=expected_columns)
        if japanese_text_series.empty:
            return pd.DataFrame(columns=expected_columns)

        results = []
        for text_input in japanese_text_series:
            original_text = str(text_input) if pd.notna(text_input) else ""
            tokens = (
                self._tokenize_japanese_text(original_text) if original_text else []
            )
            total_tokens = len(tokens)

            if total_tokens == 0:
                results.append({col: 0.0 for col in expected_columns[1:]})
                results[-1]["text"] = original_text
                continue

            results.append(
                {
                    "text": original_text,
                    "total_tokens": total_tokens,
                    "avg_token_length": 0.0,
                    "noun_ratio": 0.0,
                    "verb_ratio": 0.0,
                    "adj_ratio": 0.0,
                    "punctuation_count": 0,
                }
            )
        return pd.DataFrame(results, columns=expected_columns)

    def get_stopword(self, top_n: int = 10, min_freq: int = 5) -> list:
        if not isinstance(top_n, int) or top_n < 0:
            raise ValueError("top_n must be a non-negative integer.")
        if not isinstance(min_freq, int) or min_freq < 0:
            raise ValueError("min_freq must be a non-negative integer.")

        fdist = Counter(
            word
            for doc in self.df[self.target_col]
            if isinstance(doc, list)
            for word in doc
        )
        common_words = {word for word, _ in fdist.most_common(top_n)}
        rare_words = {word for word, freq in fdist.items() if freq <= min_freq}
        stopwords = list(common_words.union(rare_words))
        stopwords.extend(sw for sw in self.default_stopwords if sw not in stopwords)
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
        if not title_prefix or not isinstance(title_prefix, str):
            title_prefix = "plot"
        title_prefix = "".join(
            c if c.isalnum() or c in ("_", "-") else "_" for c in title_prefix
        )
        date_str = pd.to_datetime(datetime.datetime.now()).strftime("%Y-%m-%d")
        filename = f"{date_str}_{title_prefix}.html"
        full_path = os.path.join(self.output_file_path, filename)
        try:
            os.makedirs(self.output_file_path, exist_ok=True)
            plotly.offline.plot(fig, filename=full_path, auto_open=False)
            print(f"Plot saved to {full_path}")
        except (PermissionError, IOError) as e:
            print(f"Error: Could not save plot to '{full_path}'. Reason: {e}")

    def save_tables(self, prefix: str = "nlplot_output") -> None:
        if not hasattr(self, "node_df") or not hasattr(self, "edge_df"):
            print("Warning: node_df or edge_df not found. " "Run build_graph() first.")
            return

        date_str = pd.to_datetime(datetime.datetime.now()).strftime("%Y-%m-%d")
        sanitized_prefix = "".join(
            c if c.isalnum() or c in ("_", "-") else "_" for c in prefix
        )
        os.makedirs(self.output_file_path, exist_ok=True)

        for name, df in [("node", self.node_df), ("edge", self.edge_df)]:
            if isinstance(df, pd.DataFrame) and not df.empty:
                filename = os.path.join(
                    self.output_file_path,
                    f"{date_str}_{sanitized_prefix}_{name}_df.csv",
                )
                try:
                    df.to_csv(filename, index=False)
                    print(f"Saved {name}s to {filename}")
                except (PermissionError, IOError) as e:
                    print(
                        f"Error: Could not save {name} table to "
                        f"'{filename}'. Reason: {e}"
                    )
            else:
                print(f"{name.capitalize()} DataFrame is empty or not available.")

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


if __name__ == "__main__":
    sample_df_main = pd.DataFrame(
        {
            "text_column": [
                "This is a test text for basic functionality.",
                "Another example sentence to see how NLPlotLLM handles it.",
            ]
        }
    )
    npt_main = NLPlotLLM(sample_df_main, target_col="text_column")

    print("\n--- Testing Sentiment Analysis ---")
    sentiment_texts = pd.Series(
        [
            "I love this product, it's absolutely fantastic!",
            "This is the worst experience I have ever had.",
            "The weather today is just okay, nothing special.",
        ]
    )
    if LITELLM_AVAILABLE:
        try:
            print(
                "\nAttempting Ollama Sentiment Analysis "
                "(ensure Ollama is running)..."
            )
            sentiment_results_ollama = npt_main.analyze_sentiment_llm(
                sentiment_texts, model="ollama/llama2"
            )
            print("\nOllama Sentiment Results:")
            print(sentiment_results_ollama)

        except Exception as e:
            print(f"Error during sentiment analysis test: {e}")
    else:
        print("Skipping sentiment analysis test as LiteLLM is not available.")

    print("\n--- Testing Text Categorization ---")
    categorization_texts = pd.Series(
        [
            "The new iPhone 15 Pro features a titanium design.",
            "Manchester United won the match with a last-minute goal.",
            "The Federal Reserve announced an increase in interest rates.",
        ]
    )
    categories_list = ["technology", "sports", "finance", "health"]
    if LITELLM_AVAILABLE:
        try:
            print("\nAttempting Ollama Categorization " "(ensure Ollama is running)...")
            categorization_results_ollama = npt_main.categorize_text_llm(
                categorization_texts,
                categories_list,
                model="ollama/llama2",
                multi_label=True,
            )
            print("\nOllama Categorization Results (Multi-label):")
            print(categorization_results_ollama)
        except Exception as e:
            print(f"Error during categorization test: {e}")
    else:
        print("Skipping categorization test as LiteLLM is not available.")

    print("\n--- Testing Text Summarization ---")
    summarization_texts = pd.Series(
        [
            (
                "Artificial intelligence (AI) has a long history, starting "
                "from ancient myths about intelligent beings. Modern AI is "
                "based on formalizing reasoning, which led to the invention "
                "of computers."
            ),
            (
                "The quick brown fox jumps over the lazy dog. This sentence "
                "contains all letters of the English alphabet and is used "
                "for testing keyboards."
            ),
        ]
    )
    if LITELLM_AVAILABLE:
        try:
            print(
                "\nAttempting Ollama Summarization with Chunking "
                "(ensure Ollama is running)..."
            )
            summarization_results = npt_main.summarize_text_llm(
                summarization_texts,
                model="ollama/llama2",
                chunk_size=200,
                chunk_overlap=50,
            )
            print("\nOllama Summarization Results (Chunked):")
            print(summarization_results)

        except Exception as e:
            print(f"Error during summarization test: {e}")
    else:
        print("Skipping summarization test as LiteLLM is not available.")

    print("\n--- NLPlotLLM Basic Tests Complete ---")
    sample_df_for_plot = pd.DataFrame(
        {
            "text_plot": [
                "this is a test text for plotting",
                "another plot example text here",
            ]
        }
    )
    npt_plotter = NLPlotLLM(sample_df_for_plot, target_col="text_plot")
    if not npt_plotter.df[npt_plotter.target_col].empty:
        print("\n--- Testing a traditional plotting feature " "(N-gram Bar Chart) ---")
        fig = npt_plotter.bar_ngram(title="Sample N-gram from NLPlotLLM")
        if fig:
            print("N-gram bar chart generated.")
        else:
            print("N-gram bar chart generation failed or returned empty.")
    else:
        print("DataFrame for plotting is empty, skipping bar_ngram test.")
