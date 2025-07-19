import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from typing import Optional, List

# Assuming these are in the same package or accessible
from ..utils.common import get_colorpalette, generate_freq_df

# Type hint for a Plotly Figure object
Figure = go.Figure


def bar_ngram(
    nlplot_instance,
    title: str = "N-gram bar chart",
    xaxis_title: str = "word",
    yaxis_title: str = "count",
    ngram: int = 1,
    top_n: int = 20,
    width: int = 800,
    height: int = 500,
    color: Optional[str] = None,
    color_palette: str = "hls",
    save: bool = False,
    stopwords: Optional[List[str]] = None,
) -> Figure:
    """
    Generates a bar chart for n-gram frequencies.

    Args:
        nlplot_instance: An instance of the NLPlot class.
        title (str): The title of the chart.
        xaxis_title (str): The title for the x-axis.
        yaxis_title (str): The title for the y-axis.
        ngram (int): The 'n' in n-gram.
        top_n (int): The number of top n-grams to display.
        width (int): The width of the figure.
        height (int): The height of the figure.
        color (Optional[str]): A specific color for the bars.
        color_palette (str): The color palette to use if `color` is not
                             set.
        save (bool): Whether to save the plot as an HTML file.

    Returns:
        A Plotly Figure object.
    """
    # Get n-gram data
    df_ngram = generate_freq_df(
        nlplot_instance.df[nlplot_instance.target_col],
        n_gram=ngram,
        top_n=top_n,
        stopwords=stopwords,
    )

    # Determine colors for the chart
    if color:
        colors = [color] * len(df_ngram)
    else:
        colors = get_colorpalette(color_palette, len(df_ngram))

    # Create the figure
    fig = go.Figure(
        [
            go.Bar(
                x=df_ngram["word"],
                y=df_ngram["word_count"],
                marker_color=colors,
            )
        ]
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        width=width,
        height=height,
    )

    if save:
        nlplot_instance.save_plot(fig, f"{ngram}-gram_barchart")

    return fig


def treemap(
    nlplot_instance,
    title: str = "Treemap",
    ngram: int = 1,
    top_n: int = 20,
    width: int = 800,
    height: int = 500,
    color_palette: str = "hls",
    save: bool = False,
    stopwords: Optional[List[str]] = None,
) -> Figure:
    """
    Generates a treemap for n-gram frequencies.

    Args:
        nlplot_instance: An instance of the NLPlot class.
        title (str): The title of the chart.
        ngram (int): The 'n' in n-gram.
        top_n (int): The number of top n-grams to display.
        width (int): The width of the figure.
        height (int): The height of the figure.
        color_palette (str): The color palette to use.
        save (bool): Whether to save the plot as an HTML file.

    Returns:
        A Plotly Figure object.
    """
    # Get n-gram data
    df_ngram = generate_freq_df(
        nlplot_instance.df[nlplot_instance.target_col],
        n_gram=ngram,
        top_n=top_n,
        stopwords=stopwords,
    )

    # Create the figure
    fig = px.treemap(
        df_ngram,
        path=[px.Constant(title), "word"],
        values="word_count",
        color_discrete_sequence=get_colorpalette(
            color_palette, len(df_ngram)
        ),
    )

    # Update layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
    )

    if save:
        nlplot_instance.save_plot(fig, f"{ngram}-gram_treemap")

    return fig


def plot_japanese_text_features(
    nlplot_instance,
    df_features: pd.DataFrame,
    title: str = "Distribution of Japanese Text Features",
    width: int = 1000,
    height: int = 600,
    save: bool = False,
) -> Figure:
    """
    Plots the distribution of Japanese text features.

    Args:
        nlplot_instance: An instance of the NLPlot class.
        df_features (pd.DataFrame): DataFrame with text features.
        title (str): The title of the plot.
        width (int): The width of the figure.
        height (int): The height of the figure.
        save (bool): Whether to save the plot as an HTML file.

    Returns:
        A Plotly Figure object.
    """
    if df_features.empty:
        print("Warning: Feature DataFrame is empty. Cannot create plot.")
        return go.Figure()

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Total Tokens",
            "Avg Token Length",
            "Noun Ratio",
            "Verb Ratio",
            "Adjective Ratio",
            "Punctuation Count",
        ),
    )

    # Add histograms for each feature
    features_to_plot = [
        ("total_tokens", 1, 1),
        ("avg_token_length", 1, 2),
        ("noun_ratio", 1, 3),
        ("verb_ratio", 2, 1),
        ("adj_ratio", 2, 2),
        ("punctuation_count", 2, 3),
    ]
    for feature, row, col in features_to_plot:
        if feature in df_features.columns:
            fig.add_trace(
                go.Histogram(x=df_features[feature], name=feature),
                row=row,
                col=col,
            )

    # Update layout
    fig.update_layout(
        title_text=title,
        showlegend=False,
        width=width,
        height=height,
    )

    if save:
        nlplot_instance.save_plot(fig, "japanese_text_features_dist")

    return fig
