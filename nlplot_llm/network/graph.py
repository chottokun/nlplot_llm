import community as community_louvain
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Tuple, List

# Type hint for a Plotly Figure object for clarity
Figure = go.Figure


def get_edges_nodes(
    nlplot_instance,
    top_n: int = 20,
    min_edge_frequency: int = 10,
    stopwords: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes nodes and edges for a co-occurrence network graph.

    Args:
        nlplot_instance: An instance of the NLPlot class.
        top_n (int): The number of top co-occurring words to consider.
        min_edge_frequency (int): Minimum frequency for an edge to be
                                  included.
        stopwords (Optional[List[str]]): A list of stopwords to ignore.

    Returns:
        A tuple containing two pandas DataFrames: (edges_df, node_df).
    """
    if not hasattr(nlplot_instance, "df") or not hasattr(
        nlplot_instance, "target_col"
    ):
        raise AttributeError(
            "The nlplot_instance must have 'df' and 'target_col' attributes."
        )

    # Use a copy to avoid modifying the original DataFrame
    df = nlplot_instance.df.copy()
    stop_words = stopwords or []

    # Filter out stopwords from the target column
    df[nlplot_instance.target_col] = df[nlplot_instance.target_col].apply(
        lambda words: [word for word in words if word not in stop_words]
    )

    # Create a list of all words
    all_words = [
        word for words in df[nlplot_instance.target_col] for word in words
    ]

    # Create a frequency distribution of words
    fdist = pd.Series(all_words).value_counts()
    top_words = fdist.iloc[:top_n].index

    # Create a DataFrame of co-occurrences
    co_occurrence_list = []
    for words in df[nlplot_instance.target_col]:
        # Consider only words from the top_n list
        words_in_top = [word for word in words if word in top_words]
        # Create pairs of co-occurring words
        for i in range(len(words_in_top)):
            for j in range(i + 1, len(words_in_top)):
                co_occurrence_list.append(
                    tuple(sorted((words_in_top[i], words_in_top[j])))
                )

    # Create a DataFrame from the co-occurrence list
    co_occurrence_df = (
        pd.DataFrame(co_occurrence_list, columns=["word1", "word2"])
        .value_counts()
        .reset_index(name="edge_frequency")
    )

    # Filter edges by minimum frequency
    edges_df = co_occurrence_df[
        co_occurrence_df["edge_frequency"] >= min_edge_frequency
    ]

    # Create the node DataFrame
    node_df = (
        pd.DataFrame(all_words, columns=["word"])
        .value_counts()
        .reset_index(name="node_frequency")
    )
    node_df = node_df[node_df["word"].isin(top_words)]

    return edges_df, node_df


def build_graph(
    nlplot_instance,
    stopwords: Optional[List[str]] = None,
    min_edge_frequency: int = 10,
) -> Optional[nx.Graph]:
    """
    Builds a networkx graph from co-occurrence data.

    Args:
        nlplot_instance: An instance of the NLPlot class.
        stopwords (Optional[List[str]]): A list of stopwords to exclude.
        min_edge_frequency (int): The minimum frequency for an edge.

    Returns:
        A networkx Graph object, or None if data is insufficient.
    """
    stop_words = stopwords or []
    nlplot_instance.df[nlplot_instance.target_col] = nlplot_instance.df[
        nlplot_instance.target_col
    ].apply(lambda words: [word for word in words if word not in stop_words])

    if nlplot_instance.df[nlplot_instance.target_col].apply(len).sum() == 0:
        print(
            "Warning: No words left after removing stopwords. "
            "Graph cannot be built."
        )
        return None

    # Get edges and nodes
    edges_df, node_df = get_edges_nodes(
        nlplot_instance, min_edge_frequency=min_edge_frequency
    )

    if edges_df.empty or node_df.empty:
        print("Warning: No edges or nodes to build the graph.")
        return None

    # Create a graph from the edges
    graph = nx.from_pandas_edgelist(
        edges_df,
        "word1",
        "word2",
        ["edge_frequency"],
        create_using=nx.Graph(),
    )

    # Add node attributes
    for _, row in node_df.iterrows():
        if row["word"] in graph:
            graph.nodes[row["word"]]["node_frequency"] = row["node_frequency"]

    return graph


def _prepare_data_for_graph(
    nlplot_instance, G: nx.Graph
) -> Tuple[go.Scatter, go.Scatter, List[str], List[str]]:
    """
    Prepares data for plotting a network graph.

    Args:
        nlplot_instance: An instance of the NLPlot class.
        G (nx.Graph): The networkx graph to plot.

    Returns:
        A tuple containing edge_trace, node_trace, node_adjacencies,
        and node_text.
    """
    pos = nx.spring_layout(G, k=0.7, iterations=50)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_adjacencies, node_text = [], []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(
            f"{adjacencies[0]} - # of connections: {len(adjacencies[1])}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
    )

    return edge_trace, node_trace, node_adjacencies, node_text


def _initialize_empty_graph_attributes(nlplot_instance) -> None:
    """Initializes graph-related attributes to empty or default values."""
    nlplot_instance.node_df = pd.DataFrame()
    nlplot_instance.edge_df = pd.DataFrame()
    nlplot_instance.G = None
    nlplot_instance.node_trace = go.Scatter()
    nlplot_instance.edge_trace = go.Scatter()
    nlplot_instance.node_adjacencies = []
    nlplot_instance.node_text = []


def _calculate_graph_metrics(nlplot_instance, G: nx.Graph) -> None:
    """Calculates and stores various graph metrics."""
    if not G or not isinstance(G, nx.Graph):
        return

    # Centrality measures
    nlplot_instance.node_df["degree_centrality"] = pd.Series(
        nx.degree_centrality(G)
    )
    nlplot_instance.node_df["eigenvector_centrality"] = pd.Series(
        nx.eigenvector_centrality(G, max_iter=500)
    )
    nlplot_instance.node_df["betweenness_centrality"] = pd.Series(
        nx.betweenness_centrality(G)
    )


def _detect_communities(nlplot_instance, G: nx.Graph) -> None:
    """Detects and assigns community partitions to nodes."""
    if not G or not isinstance(G, nx.Graph):
        return

    try:
        partition = community_louvain.best_partition(G)
        nlplot_instance.node_df["community"] = nlplot_instance.node_df[
            "word"
        ].map(partition)
    except Exception as e:
        print(f"Community detection failed: {e}")
        nlplot_instance.node_df["community"] = 0


def _create_network_trace(
    nlplot_instance, G: nx.Graph
) -> Tuple[go.Scatter, go.Scatter]:
    """Creates Plotly traces for nodes and edges."""
    if not G:
        return go.Scatter(), go.Scatter()

    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )
    node_trace = go.Scatter(
        x=[],
        y=[],
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
    )

    # Populate edge trace
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace["x"] += (x0, x1, None)
        edge_trace["y"] += (y0, y1, None)

    # Populate node trace
    for node in G.nodes():
        x, y = pos[node]
        node_trace["x"] += (x,)
        node_trace["y"] += (y,)

    return edge_trace, node_trace


def co_network(
    nlplot_instance,
    title: str = "Co-occurrence network",
    color_palette: str = "hls",
    width: int = 1100,
    height: int = 700,
    save: bool = False,
) -> Figure:
    """
    Generates and displays a co-occurrence network graph.

    Args:
        nlplot_instance: An instance of the NLPlot class.
        title (str): The title of the graph.
        color_palette (str): The color palette for the graph.
        width (int): The width of the figure.
        height (int): The height of the figure.
        save (bool): Whether to save the plot as an HTML file.

    Returns:
        A Plotly Figure object.
    """
    _initialize_empty_graph_attributes(nlplot_instance)
    build_graph(nlplot_instance)

    if not nlplot_instance.G:
        print("Graph could not be created.")
        return go.Figure()

    edge_trace, node_trace, node_adj, node_txt = _prepare_data_for_graph(
        nlplot_instance, nlplot_instance.G
    )
    nlplot_instance.node_adjacencies.extend(node_adj)
    nlplot_instance.node_text.extend(node_txt)

    # Update node trace with colors and text
    node_trace.marker.color = nlplot_instance.node_adjacencies
    node_trace.text = nlplot_instance.node_text

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                    text="Powered by: Plotly",
                )
            ],
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False
            ),
        ),
    )
    fig.update_layout(width=width, height=height)

    if save:
        nlplot_instance.save_plot(fig, "co_occurrence_network")

    return fig


def _calculate_node_sizes(
    nlplot_instance, size_metric: str = "node_frequency"
) -> List[float]:
    """
    Calculates node sizes based on a specified metric.

    Args:
        nlplot_instance: An instance of the NLPlot class.
        size_metric (str): The metric to use for node sizes.

    Returns:
        A list of node sizes.
    """
    if size_metric not in nlplot_instance.node_df.columns:
        raise ValueError(
            f"Metric '{size_metric}' not found in node DataFrame."
        )

    # Normalize the metric to a suitable size range
    sizes = nlplot_instance.node_df[size_metric]
    return (sizes - sizes.min() + 1) / (sizes.max() - sizes.min() + 1) * 50


def sunburst(
    nlplot_instance,
    title: str = "Sunburst Chart of Word Communities",
    width: int = 800,
    height: int = 800,
    save: bool = False,
) -> Figure:
    """
    Creates a sunburst chart to visualize word communities.

    Args:
        nlplot_instance: An instance of the NLPlot class.
        title (str): The title of the chart.
        width (int): The width of the figure.
        height (int): The height of the figure.
        save (bool): Whether to save the plot as an HTML file.

    Returns:
        A Plotly Figure object.
    """
    if "community" not in nlplot_instance.node_df.columns:
        print(
            "Community information is not available. Please run "
            "build_graph first."
        )
        return go.Figure()

    sunburst_df = nlplot_instance.node_df.copy()
    sunburst_df["community_str"] = (
        "community " + sunburst_df["community"].astype(str)
    )

    fig = go.Figure(
        go.Sunburst(
            labels=sunburst_df["word"],
            parents=sunburst_df["community_str"],
            values=sunburst_df["node_frequency"],
            branchvalues="total",
        )
    )
    fig.update_layout(
        title=title,
        margin=dict(t=50, l=0, r=0, b=0),
        width=width,
        height=height,
    )

    if save:
        nlplot_instance.save_plot(fig, "sunburst_chart")

    return fig
