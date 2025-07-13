import gc
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import community
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import iplot
from sklearn import preprocessing
from ..utils.common import (
    _add_unique_combinations_to_dict,
    _unique_combinations_for_edges,
    get_colorpalette
)


def get_edges_nodes(nlplot_instance, batches: list, min_edge_frequency: int) -> None:
    if not isinstance(min_edge_frequency, int) or min_edge_frequency < 0:
        raise ValueError("min_edge_frequency must be a non-negative integer.")
    edge_dict = {}
    # print(f"DEBUG: Initial edge_dict: {edge_dict}")
    for i, batch in enumerate(batches):
        # print(f"DEBUG: Processing batch {i}: {batch}")
        if isinstance(batch, list) and batch:
            unique_elements_in_batch = list(set(batch))
            # print(f"DEBUG: Unique elements in batch {i}: {unique_elements_in_batch}")
            if len(unique_elements_in_batch) >= 2:
                combinations = _unique_combinations_for_edges(unique_elements_in_batch)
                #print(f"DEBUG: Combinations for batch {i}: {combinations}")
                edge_dict = _add_unique_combinations_to_dict(combinations, edge_dict)
                #print(f"DEBUG: edge_dict after batch {i}: {edge_dict}")
            else:
                #print(f"DEBUG: Batch {i} has less than 2 unique elements.")
                pass
        else:
            # print(f"DEBUG: Batch {i} is not a valid list.")
            pass
    source, target, edge_frequency_list = [], [], []
    for key, value in edge_dict.items():
        source.append(key[0])
        target.append(key[1])
        edge_frequency_list.append(value)
    edge_df = pd.DataFrame({'source': source, 'target': target, 'edge_frequency': edge_frequency_list})
    edge_df = edge_df[edge_df['edge_frequency'] > min_edge_frequency].sort_values(by='edge_frequency', ascending=False).reset_index(drop=True)
    if edge_df.empty:
        nlplot_instance.edge_df = pd.DataFrame(columns=['source', 'target', 'edge_frequency', 'source_code', 'target_code'])
        nlplot_instance.node_df = pd.DataFrame(columns=['id', 'id_code'])
        nlplot_instance.node_dict = {}
        nlplot_instance.edge_dict = edge_dict
        return
    unique_nodes = list(set(edge_df['source']).union(set(edge_df['target'])))
    node_df = pd.DataFrame({'id': unique_nodes})
    if not node_df.empty:
        node_df['id_code'] = node_df.index
        node_dict = dict(zip(node_df['id'], node_df['id_code']))
        edge_df['source_code'] = edge_df['source'].map(node_dict)
        edge_df['target_code'] = edge_df['target'].map(node_dict)
        edge_df.dropna(subset=['source_code', 'target_code'], inplace=True)
    else:
        node_dict = {}
        edge_df = pd.DataFrame(columns=['source', 'target', 'edge_frequency', 'source_code', 'target_code'])
    nlplot_instance.edge_df = edge_df
    nlplot_instance.node_df = node_df
    nlplot_instance.node_dict = node_dict
    nlplot_instance.edge_dict = edge_dict
    return None

def get_graph(nlplot_instance) -> nx.Graph:
    G = nx.Graph()
    if not hasattr(nlplot_instance, 'node_df') or nlplot_instance.node_df.empty:
        print("Warning: Node DataFrame is not initialized or empty. Cannot build graph.")
        return G
    G.add_nodes_from(nlplot_instance.node_df.id_code)
    if not hasattr(nlplot_instance, 'edge_df') or nlplot_instance.edge_df.empty:
        return G
    edge_tuples = [(nlplot_instance.edge_df['source_code'].iloc[i], nlplot_instance.edge_df['target_code'].iloc[i]) for i in range(len(nlplot_instance.edge_df))]
    G.add_edges_from(edge_tuples)
    return G

def build_graph(nlplot_instance, stopwords: list = [], min_edge_frequency: int = 10) -> None:
    _prepare_data_for_graph(nlplot_instance, stopwords)
    get_edges_nodes(nlplot_instance, nlplot_instance._batches, min_edge_frequency)
    if nlplot_instance.node_df.empty:
        _initialize_empty_graph_attributes(nlplot_instance)
        print('Warning: No nodes found after processing for build_graph. Co-occurrence network cannot be built.')
        print('node_size:0, edge_size:0')
        return
    nlplot_instance.G = get_graph(nlplot_instance)
    if not nlplot_instance.G.nodes():
        _initialize_empty_graph_attributes(nlplot_instance, graph_exists_but_no_nodes=True)
        print('Warning: Graph has no nodes. Further calculations for co-occurrence network will be skipped.')
        print(f'node_size:{len(nlplot_instance.node_df)}, edge_size:{len(nlplot_instance.edge_df if hasattr(nlplot_instance, "edge_df") else [])}')
        return
    _calculate_graph_metrics(nlplot_instance)
    _detect_communities(nlplot_instance)
    # print(f'node_size:{len(nlplot_instance.node_df)}, edge_size:{len(nlplot_instance.edge_df if hasattr(nlplot_instance, "edge_df") else [])}')
    return None

import re

def _prepare_data_for_graph(nlplot_instance, stopwords_param: list):
    current_stopwords = set(sw.lower() for sw in stopwords_param + nlplot_instance.default_stopwords)
    nlplot_instance.df_edit = nlplot_instance.df.copy()

    def process_doc(doc):
        if not isinstance(doc, list):
            return []

        words = [re.sub(r'[^\w\s]', '', w).lower() for w in doc]

        filtered_words = [w for w in words if w and w not in current_stopwords]

        return filtered_words

    nlplot_instance.df_edit.loc[:, nlplot_instance.target_col] = nlplot_instance.df_edit[nlplot_instance.target_col].apply(process_doc)
    nlplot_instance._batches = nlplot_instance.df_edit[nlplot_instance.target_col].tolist()

def _initialize_empty_graph_attributes(nlplot_instance, graph_exists_but_no_nodes=False):
    nlplot_instance.G = nx.Graph()
    nlplot_instance.adjacencies = {}
    nlplot_instance.betweeness = {}
    nlplot_instance.clustering_coeff = {}
    nlplot_instance.communities = []
    nlplot_instance.communities_dict = {}
    if not graph_exists_but_no_nodes and hasattr(nlplot_instance, 'node_df') and not nlplot_instance.node_df.empty:
        nlplot_instance.node_df['community'] = -1

def _calculate_graph_metrics(nlplot_instance):
    if not hasattr(nlplot_instance, 'G') or not nlplot_instance.G.nodes():
        print("Warning: Graph not available for metric calculation.")
        return
    nlplot_instance.adjacencies = dict(nlplot_instance.G.adjacency())
    nlplot_instance.betweeness = nx.betweenness_centrality(nlplot_instance.G)
    nlplot_instance.clustering_coeff = nx.clustering(nlplot_instance.G)
    nlplot_instance.node_df['adjacency_frequency'] = nlplot_instance.node_df['id_code'].map(lambda x: len(nlplot_instance.adjacencies.get(x, {})))
    nlplot_instance.node_df['betweeness_centrality'] = nlplot_instance.node_df['id_code'].map(lambda x: nlplot_instance.betweeness.get(x, 0.0))
    nlplot_instance.node_df['clustering_coefficient'] = nlplot_instance.node_df['id_code'].map(lambda x: nlplot_instance.clustering_coeff.get(x, 0.0))

def _detect_communities(nlplot_instance):
    if not hasattr(nlplot_instance, 'G') or not nlplot_instance.G.nodes() or nlplot_instance.node_df.empty:
        print("Warning: Graph or node_df not available for community detection.")
        nlplot_instance.communities = []
        nlplot_instance.communities_dict = {}
        if hasattr(nlplot_instance, 'node_df') and not nlplot_instance.node_df.empty:
            nlplot_instance.node_df['community'] = -1
        return
    raw_communities = community.greedy_modularity_communities(nlplot_instance.G)
    nlplot_instance.communities = [list(comm) for comm in raw_communities if comm]
    nlplot_instance.communities_dict = {i: comm_nodes for i, comm_nodes in enumerate(nlplot_instance.communities)}
    def community_allocation(id_code):
        for k, v_list in nlplot_instance.communities_dict.items():
            if id_code in v_list:
                return k
        return -1
    nlplot_instance.node_df['community'] = nlplot_instance.node_df['id_code'].map(community_allocation)

def _create_network_trace(trace_type: str, **kwargs) -> go.Scatter:
    if trace_type == "edge":
        return go.Scatter(x=kwargs['x'], y=kwargs['y'], mode='lines', line={'width': kwargs['width'], 'color': kwargs['color']}, line_shape='spline', opacity=kwargs['opacity'])
    elif trace_type == "node":
        return go.Scatter(x=kwargs['x'], y=kwargs['y'], text=kwargs['text'], mode='markers+text', textposition='bottom center', hoverinfo="text", marker=kwargs['marker'])
    raise ValueError(f"Unknown trace_type: {trace_type}")

def co_network(nlplot_instance, title:str = None, sizing:int=100, node_size_col:str='adjacency_frequency', color_palette:str='hls', layout_func=nx.kamada_kawai_layout, light_theme:bool=True, width:int=1700, height:int=1200, save:bool=False) -> None:
    if not hasattr(nlplot_instance, 'G') or not nlplot_instance.G.nodes():
        print("Warning: Graph not built or empty. Cannot plot co-occurrence network.")
        return
    if not hasattr(nlplot_instance, 'node_df') or nlplot_instance.node_df.empty:
        print("Warning: Node DataFrame not available or empty. Cannot plot co-occurrence network.")
        return
    if node_size_col not in nlplot_instance.node_df.columns:
        print(f"Warning: node_size column '{node_size_col}' not found in node_df. Using 'adjacency_frequency'.")
        node_size_col = 'adjacency_frequency'
        if node_size_col not in nlplot_instance.node_df.columns:
            print(f"Warning: Default node_size column 'adjacency_frequency' also not found. Node sizes will be uniform.")
            nlplot_instance.node_df['uniform_size'] = 10
            node_size_col = 'uniform_size'
    back_col, edge_col = ('#ffffff', '#ece8e8') if light_theme else ('#000000', '#2d2b2b')
    final_node_sizes = _calculate_node_sizes(nlplot_instance, node_size_col, sizing)
    pos = layout_func(nlplot_instance.G)
    for node_id_code in nlplot_instance.G.nodes():
        nlplot_instance.G.nodes[node_id_code]['pos'] = list(pos[node_id_code])
    edge_traces = [_create_network_trace(trace_type="edge", x=[nlplot_instance.G.nodes[edge_nodes[0]]['pos'][0], nlplot_instance.G.nodes[edge_nodes[1]]['pos'][0], None], y=[nlplot_instance.G.nodes[edge_nodes[0]]['pos'][1], nlplot_instance.G.nodes[edge_nodes[1]]['pos'][1], None], width=1.2, color=edge_col, opacity=1) for edge_nodes in nlplot_instance.G.edges()]
    node_x, node_y, node_hover_text, node_marker_colors, node_marker_sizes = [], [], [], [], []
    if 'community' not in nlplot_instance.node_df.columns or not pd.api.types.is_numeric_dtype(nlplot_instance.node_df['community']):
        nlplot_instance.node_df['community_display'] = 0
    else:
        nlplot_instance.node_df['community_display'] = nlplot_instance.node_df['community']
    num_communities = nlplot_instance.node_df['community_display'].nunique()
    palette_colors = get_colorpalette(color_palette, num_communities if num_communities > 0 else 1)
    id_code_to_info = nlplot_instance.node_df.set_index('id_code')
    for id_code_node in nlplot_instance.G.nodes():
        x, y = nlplot_instance.G.nodes[id_code_node]['pos']
        node_x.append(x)
        node_y.append(y)
        node_specific_info = id_code_to_info.loc[id_code_node]
        node_hover_text.append(node_specific_info['id'])
        community_val = int(node_specific_info['community_display'])
        node_marker_colors.append(palette_colors[community_val % len(palette_colors)])
        node_marker_sizes.append(final_node_sizes.loc[node_specific_info.name])
    node_trace = _create_network_trace(trace_type="node", x=node_x, y=node_y, text=node_hover_text, marker={'size': node_marker_sizes, 'line': dict(width=0.5, color=edge_col), 'color': node_marker_colors})
    fig_data = edge_traces + [node_trace]
    fig_layout = go.Layout(title=str(title) if title else "Co-occurrence Network", font=dict(family='Arial', size=12), width=width, height=height, autosize=True, showlegend=False, xaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''), yaxis=dict(showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''), margin=dict(l=40, r=40, b=85, t=100, pad=0), hovermode='closest', plot_bgcolor=back_col)
    fig = go.Figure(data=fig_data, layout=fig_layout)
    iplot(fig)
    if save:
        nlplot_instance.save_plot(fig, title if title else "co_network")
    gc.collect()
    return fig

def _calculate_node_sizes(nlplot_instance, node_size_col: str, sizing_factor: int) -> pd.Series:
    if node_size_col not in nlplot_instance.node_df.columns or nlplot_instance.node_df[node_size_col].isnull().all():
        print(f"Warning: Node size column '{node_size_col}' not found or all nulls. Using uniform small size.")
        return pd.Series([sizing_factor * 0.1] * len(nlplot_instance.node_df), index=nlplot_instance.node_df.index)
    node_sizes_numeric = pd.to_numeric(nlplot_instance.node_df[node_size_col], errors='coerce').fillna(0)
    if len(node_sizes_numeric) == 0:
        return pd.Series(index=nlplot_instance.node_df.index, dtype=float)
    if node_sizes_numeric.nunique() <= 1:
        if node_sizes_numeric.iloc[0] == 0:
            return pd.Series([sizing_factor * 0.1] * len(nlplot_instance.node_df), index=nlplot_instance.node_df.index)
        else:
            return pd.Series([sizing_factor * 0.5] * len(nlplot_instance.node_df), index=nlplot_instance.node_df.index)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 1.0))
    scaled_values = min_max_scaler.fit_transform(node_sizes_numeric.values.reshape(-1, 1)).flatten()
    return pd.Series(scaled_values, index=nlplot_instance.node_df.index) * sizing_factor

def sunburst(nlplot_instance, title:str=None, colorscale:bool=False, color_col:str='betweeness_centrality', color_continuous_scale:str='Oryel', width:int=1100, height:int=1100, save:bool=False) -> go.Figure:
    if not hasattr(nlplot_instance, 'node_df') or nlplot_instance.node_df.empty:
        print("Warning: Node DataFrame not available or empty. Cannot plot sunburst chart.")
        return go.Figure()
    _df = nlplot_instance.node_df.copy()
    if 'community' not in _df.columns:
        _df['community'] = '0'
    else:
        _df['community'] = _df['community'].astype(str)
    if 'id' not in _df.columns:
        _df['id'] = "Unknown"
    if 'adjacency_frequency' not in _df.columns or _df['adjacency_frequency'].isnull().all():
        print("Warning: 'adjacency_frequency' column is missing or all nulls. Sunburst may be empty or error.")
        _df['adjacency_frequency'] = 1
    path_cols = ['community', 'id']
    try:
        if colorscale:
            if color_col not in _df.columns or _df[color_col].isnull().all():
                print(f"Warning: color_col '{color_col}' for sunburst is missing or all nulls. Using default coloring.")
                fig = px.sunburst(_df, path=path_cols, values='adjacency_frequency', color='community')
            else:
                _df[color_col] = pd.to_numeric(_df[color_col], errors='coerce').fillna(0)
                fig = px.sunburst(_df, path=path_cols, values='adjacency_frequency', color=color_col, hover_data=None, color_continuous_scale=color_continuous_scale, color_continuous_midpoint=np.average(_df[color_col].fillna(0), weights=_df['adjacency_frequency'].fillna(1)))
        else:
            fig = px.sunburst(_df, path=path_cols, values='adjacency_frequency', color='community')
    except Exception as e:
        print(f"Error creating sunburst chart: {e}. Returning empty figure.")
        return go.Figure()
    fig.update_layout(title=str(title) if title else 'Sunburst Chart', width=width, height=height)
    if save:
        nlplot_instance.save_plot(fig, title if title else "sunburst_chart")
    del _df
    gc.collect()
    return fig
