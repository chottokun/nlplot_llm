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

# Default path for Japanese Font, can be overridden
DEFAULT_FONT_PATH = str(os.path.dirname(__file__)) + '/data/mplus-1c-regular.ttf'

try:
    from janome.tokenizer import Tokenizer as JanomeTokenizer
    JANOME_AVAILABLE = True
except ImportError:
    JANOME_AVAILABLE = False
    class JanomeTokenizer: # type: ignore
        def tokenize(self, text: str, stream=False, wakati=False): # type: ignore
            return []
    # class JanomeToken: pass

# Langchain related imports
try:
    from langchain_openai import ChatOpenAI
    from langchain_community.chat_models.ollama import OllamaChat
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.prompts import PromptTemplate
    from langchain_core.outputs import AIMessage
    from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    class ChatOpenAI: pass # type: ignore
    class OllamaChat: pass # type: ignore
    class BaseChatModel: pass # type: ignore
    class PromptTemplate: pass # type: ignore
    class AIMessage: # type: ignore
        def __init__(self, content=""): self.content = content
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

class NLPlot():
    def __init__( self, df: pd.DataFrame, target_col: str, output_file_path: str = './',
                  default_stopwords_file_path: str = '', font_path: str = None):
        self.df = df.copy()
        self.target_col = target_col
        self.df.dropna(subset=[self.target_col], inplace=True)
        if not self.df.empty and not pd.isna(self.df[self.target_col].iloc[0]) and \
           not isinstance(self.df[self.target_col].iloc[0], list):
            self.df.loc[:, self.target_col] = self.df[self.target_col].astype(str).map(lambda x: x.split())
        self.output_file_path = output_file_path
        self.font_path = font_path if font_path and os.path.exists(font_path) else DEFAULT_FONT_PATH
        if font_path and not os.path.exists(font_path):
            print(f"Warning: Specified font_path '{font_path}' not found. Falling back to default: {self.font_path}")
        if not os.path.exists(self.font_path):
            print(f"Warning: The determined font path '{self.font_path}' does not exist. WordCloud may fail if a valid font is not provided at runtime or if the default font is missing.")
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
        current_font_path = font_path if font_path and os.path.exists(font_path) else self.font_path
        if font_path and not os.path.exists(font_path): print(f"Warning: Specified font_path '{font_path}' for wordcloud not found. Falling back to instance/default: {current_font_path}")
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
            if not os.path.exists(current_font_path): raise OSError(f"Font file not found at {current_font_path}")
            wordcloud_instance = WordCloud(font_path=current_font_path, stopwords=current_stopwords, max_words=max_words,max_font_size=max_font_size, random_state=42, width=width, height=height,mask=mask, collocations=False, prefer_horizontal=1, colormap=colormap, background_color='white', font_step=1, contour_width=0, contour_color='steelblue')
            wordcloud_instance.generate(text_corpus)
        except (OSError, TypeError) as e:
            print(f"Warning: Error processing font at '{current_font_path}': {e}.")
            if current_font_path != DEFAULT_FONT_PATH and os.path.exists(DEFAULT_FONT_PATH):
                print(f"Attempting to fallback to default font: {DEFAULT_FONT_PATH}")
                try:
                    current_font_path = DEFAULT_FONT_PATH
                    wordcloud_instance = WordCloud(font_path=current_font_path, stopwords=current_stopwords, max_words=max_words,max_font_size=max_font_size, random_state=42, width=width, height=height,mask=mask, collocations=False, prefer_horizontal=1, colormap=colormap, background_color='white', font_step=1, contour_width=0, contour_color='steelblue')
                    wordcloud_instance.generate(text_corpus)
                except Exception as fallback_e: print(f"Error: Fallback to default font ('{DEFAULT_FONT_PATH}') also failed: {fallback_e}. WordCloud cannot be generated."); return
            elif current_font_path == DEFAULT_FONT_PATH: print(f"Error: Default font at '{DEFAULT_FONT_PATH}' seems to be an issue. WordCloud cannot be generated. Details: {e}"); return
            else: print(f"Error: Default font not found at '{DEFAULT_FONT_PATH}' and custom font failed. WordCloud cannot be generated."); return
        except ValueError as e:
            if "empty" in str(e).lower() or "zero" in str(e).lower(): print(f"Warning: WordCloud could not be generated. All words might have been filtered out or corpus is empty. Details: {e}"); return
            else: print(f"An unexpected ValueError occurred during WordCloud generation: {e}"); return
        except Exception as e: print(f"An unexpected error occurred during WordCloud generation: {e}"); return
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
        if not hasattr(self, 'node_df') or self.node_df.empty: print("Warning: Node DataFrame is not initialized or empty. Cannot build graph."); return G
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
    def _get_llm_client(self, llm_provider: str, model_name: str, **kwargs) -> BaseChatModel:
        if not LANGCHAIN_AVAILABLE: raise ImportError("Langchain or related packages are not installed. Please install them to use LLM features (e.g., pip install langchain langchain-openai langchain-community openai).")
        provider = llm_provider.lower()
        if provider == "openai":
            api_key = kwargs.pop("openai_api_key", os.getenv("OPENAI_API_KEY"))
            if not api_key: raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or pass 'openai_api_key' as a keyword argument.")
            try: return ChatOpenAI(model=model_name, openai_api_key=api_key, **kwargs)
            except Exception as e: raise ValueError(f"Failed to initialize OpenAI client for model '{model_name}': {e}")
        elif provider == "ollama":
            base_url = kwargs.pop("base_url", "http://localhost:11434")
            ollama_kwargs = {k: v for k, v in kwargs.items() if k != 'openai_api_key'}
            try: return OllamaChat(model=model_name, base_url=base_url, **ollama_kwargs)
            except Exception as e: raise ValueError(f"Failed to initialize Ollama client for model '{model_name}' with base_url '{base_url}': {e}")
        else: raise ValueError(f"Unsupported LLM provider: '{llm_provider}'. Supported providers are 'openai', 'ollama'.")

    def analyze_sentiment_llm(
        self,
        text_series: pd.Series,
        llm_provider: str,
        model_name: str,
        prompt_template_str: Optional[str] = None,
        temperature: float = 0.0,
        **llm_config
    ) -> pd.DataFrame:
        if not LANGCHAIN_AVAILABLE:
            print("Warning: Langchain or its dependencies are not installed. LLM-based sentiment analysis is not available.")
            return pd.DataFrame(columns=["text", "sentiment", "raw_llm_output"])
        if not isinstance(text_series, pd.Series):
            print("Warning: Input 'text_series' must be a pandas Series. Returning empty DataFrame with expected columns.")
            return pd.DataFrame(columns=["text", "sentiment", "raw_llm_output"])
        if text_series.empty: return pd.DataFrame(columns=["text", "sentiment", "raw_llm_output"])
        try:
            llm_client_config = llm_config.copy()
            if 'temperature' not in llm_client_config: llm_client_config['temperature'] = temperature
            llm = self._get_llm_client(llm_provider, model_name, **llm_client_config)
        except (ImportError, ValueError) as e:
            print(f"Error initializing LLM client: {e}")
            return pd.DataFrame([{'text': str(txt) if pd.notna(txt) else "", 'sentiment': 'error', 'raw_llm_output': str(e)} for txt in text_series], columns=["text", "sentiment", "raw_llm_output"])
        if prompt_template_str is None: prompt_template_str = "Analyze the sentiment of the following text and classify it as 'positive', 'negative', or 'neutral'. Return only the single word classification for the sentiment. Text: {text}"
        try: prompt = PromptTemplate.from_template(prompt_template_str)
        except Exception as e: print(f"Error creating prompt template from string \"{prompt_template_str}\": {e}. Using a basic pass-through."); prompt = PromptTemplate.from_template("Text: {text}\nSentiment:")
        results = []
        for text_input in text_series:
            original_text_for_df = str(text_input) if pd.notna(text_input) else ""; sentiment = "unknown"; raw_output = ""
            if not original_text_for_df.strip(): sentiment = "neutral"; raw_output = "Input text was empty or whitespace."
            else:
                try:
                    formatted_prompt_content = prompt.format_prompt(text=original_text_for_df).to_string()
                    response_message = llm.invoke(formatted_prompt_content)
                    raw_output = response_message.content if hasattr(response_message, 'content') else str(response_message)
                    processed_output = raw_output.strip().lower()
                    if "positive" in processed_output: sentiment = "positive"
                    elif "negative" in processed_output: sentiment = "negative"
                    elif "neutral" in processed_output: sentiment = "neutral"
                except Exception as e: print(f"Error analyzing sentiment for text '{original_text_for_df[:50]}...': {e}"); sentiment = "error"; raw_output = str(e)
            results.append({"text": original_text_for_df, "sentiment": sentiment, "raw_llm_output": raw_output})
        return pd.DataFrame(results, columns=["text", "sentiment", "raw_llm_output"])

    def categorize_text_llm(
        self,
        text_series: pd.Series,
        categories: List[str],
        llm_provider: str,
        model_name: str,
        prompt_template_str: Optional[str] = None,
        multi_label: bool = False,
        temperature: float = 0.0,
        **llm_config
    ) -> pd.DataFrame:
        """
        (TDD Cycle 3 - Implementation)
        Categorizes texts using a specified LLM via Langchain.
        """
        category_col_name = "categories" if multi_label else "category"
        default_columns = ["text", category_col_name, "raw_llm_output"]
        if not LANGCHAIN_AVAILABLE: print("Warning: Langchain or its dependencies are not installed. LLM-based categorization is not available."); return pd.DataFrame(columns=default_columns)
        if not isinstance(text_series, pd.Series): print("Warning: Input 'text_series' must be a pandas Series."); return pd.DataFrame(columns=default_columns)
        if text_series.empty: return pd.DataFrame(columns=default_columns)
        if not categories or not isinstance(categories, list) or not all(isinstance(c, str) for c in categories): raise ValueError("Categories list must be a non-empty list of strings.")
        try:
            llm_client_config = llm_config.copy();
            if 'temperature' not in llm_client_config: llm_client_config['temperature'] = temperature
            llm = self._get_llm_client(llm_provider, model_name, **llm_client_config)
        except (ImportError, ValueError) as e:
            print(f"Error initializing LLM client: {e}")
            return pd.DataFrame([{'text': str(txt) if pd.notna(txt) else "", category_col_name: [] if multi_label else 'error', 'raw_llm_output': str(e)} for txt in text_series], columns=default_columns)
        category_list_str = ", ".join(f"'{c}'" for c in categories)
        if prompt_template_str is None:
            if multi_label: prompt_template_str = (f"Analyze the following text and classify it into one or more of these categories: {category_list_str}. Return a comma-separated list of the matching category names. If no categories match, return 'none'. Text: {{text}}")
            else: prompt_template_str = (f"Analyze the following text and classify it into exactly one of these categories: {category_list_str}. Return only the single matching category name. If no categories match, return 'unknown'. Text: {{text}}")
        try:
            if "{categories}" in prompt_template_str: prompt = PromptTemplate(template=prompt_template_str, input_variables=["text", "categories"])
            else: prompt = PromptTemplate.from_template(prompt_template_str)
        except Exception as e:
            print(f"Error creating prompt template: {e}.")
            return pd.DataFrame([{'text': str(txt) if pd.notna(txt) else "", category_col_name: [] if multi_label else 'error', 'raw_llm_output': f"Prompt template error: {e}"} for txt in text_series], columns=default_columns)
        results = []
        for text_input in text_series:
            original_text_for_df = str(text_input) if pd.notna(text_input) else ""; raw_llm_resp_content = ""
            if not original_text_for_df.strip(): parsed_categories = [] if multi_label else "unknown"; raw_llm_resp_content = "Input text was empty or whitespace."
            else:
                try:
                    prompt_args = {"text": original_text_for_df}
                    if "{categories}" in prompt.template: prompt_args["categories"] = category_list_str
                    formatted_prompt_content = prompt.format_prompt(**prompt_args).to_string()
                    response_message = llm.invoke(formatted_prompt_content)
                    raw_llm_resp_content = response_message.content if hasattr(response_message, 'content') else str(response_message)
                    processed_output = raw_llm_resp_content.strip().lower()
                    if multi_label:
                        found_cats_raw = [cat.strip() for cat in processed_output.split(',')]; parsed_categories = [original_cat for original_cat in categories if original_cat.lower() in [fcr.lower() for fcr in found_cats_raw]]
                    else:
                        parsed_categories = "unknown"
                        for cat_original_case in categories:
                            if cat_original_case.lower() == processed_output: parsed_categories = cat_original_case; break
                        if parsed_categories == "unknown":
                             for cat_original_case in categories:
                                 if cat_original_case.lower() in processed_output: parsed_categories = cat_original_case; break
                except Exception as e: print(f"Error categorizing text '{original_text_for_df[:50]}...': {e}"); parsed_categories = [] if multi_label else "error"; raw_llm_resp_content = str(e)
            results.append({"text": original_text_for_df, category_col_name: parsed_categories, "raw_llm_output": raw_llm_resp_content})
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
        (TDD Cycle 4 - Initial Stub)
        Splits a long text into smaller chunks using Langchain TextSplitters.
        """
        if not LANGCHAIN_AVAILABLE:
            print("Warning: Langchain or its text splitter components are not installed. Chunking will not be performed; returning original text as a single chunk.")
            return [text_to_chunk] if text_to_chunk else []

        # print("_chunk_text (stub) called.")
        if not text_to_chunk: return [] # If input is empty, return empty list of chunks

        # Actual implementation will go here in Green phase for chunking
        # For now, the stub returns the original text as a single chunk if not empty.
        # This will make tests for actual chunking fail as expected (Red state for functionality).
        return [text_to_chunk]


    def plot_japanese_text_features(
        self,
        features_df: pd.DataFrame,
        target_feature: str,
        title: Optional[str] = None,
        save: bool = False,
        **kwargs
    ) -> Optional[plotly.graph_objs.Figure]:
        """
        Plots a histogram of a specified feature from the Japanese text features DataFrame.

        Args:
            features_df (pd.DataFrame): DataFrame containing Japanese text features,
                                        typically an output from `get_japanese_text_features`.
            target_feature (str): The name of the column in `features_df` to plot.
                                  This column should contain numeric data.
            title (Optional[str], optional): Title of the plot. If None, a default title
                                             based on `target_feature` is used. Defaults to None.
            save (bool, optional): Whether to save the generated plot as an HTML file.
                                   Defaults to False.
            **kwargs: Additional keyword arguments to be passed to `plotly.express.histogram`.
                      (e.g., nbins, color, template, width, height).

        Returns:
            Optional[plotly.graph_objs.Figure]: The generated Plotly Figure object, or None if plotting fails.

        Raises:
            ValueError: If `features_df` is empty or not a DataFrame,
                        or `target_feature` is not found in `features_df`,
                        or if the `target_feature` column cannot be treated as numeric
                        or contains only NaN values after conversion.
        """
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

[end of nlplot/nlplot.py]
