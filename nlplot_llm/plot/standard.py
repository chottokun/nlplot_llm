import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import matplotlib.pyplot as plt

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

    Returns:
        A Plotly Figure object.
    """
    # Get n-gram data
    df_ngram = generate_freq_df(
        nlplot_instance.df[nlplot_instance.target_col],
        n_gram=ngram,
        top_n=top_n,
        stopwords=stopwords if stopwords is not None else [],
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


def wordcloud(
    nlplot_instance,
    width: int = 800,
    height: int = 500,
    max_words: int = 100,
    max_font_size: int = 80,
    stopwords: Optional[List[str]] = None,
    colormap: Optional[str] = None,
    mask_file: Optional[str] = None,
    font_path: Optional[str] = None,
    save: bool = False,
) -> Optional[Image.Image]:
    # Prepare stopwords set
    stopwords_set = set(stopwords) if stopwords else set(STOPWORDS)

    # Get text data from nlplot_instance
    text_data = " ".join(nlplot_instance.df[nlplot_instance.target_col].dropna().astype(str).tolist())

    if not text_data.strip():
        print("Could not generate Word Cloud. Input text might be empty or all words filtered out.")
        return None

    # Prepare mask image if provided
    mask = None
    if mask_file and os.path.exists(mask_file):
        mask = np.array(Image.open(mask_file))

    # Determine font path
    wc_font_path = None
    if font_path and os.path.exists(font_path):
        wc_font_path = font_path
    elif nlplot_instance.font_path and os.path.exists(nlplot_instance.font_path):
        wc_font_path = nlplot_instance.font_path

    # Create WordCloud object
    wc = WordCloud(
        width=width,
        height=height,
        max_words=max_words,
        max_font_size=max_font_size,
        stopwords=stopwords_set,
        colormap=colormap,
        mask=mask,
        font_path=wc_font_path,
        background_color="white",
        random_state=42,
    )

    # Generate word cloud
    wc.generate(text_data)

    # Convert to PIL Image
    image = wc.to_image()

    if save:
        save_path = f"wordcloud_{nlplot_instance.target_col}.png"
        wc.to_file(save_path)
        print(f"Word Cloud saved to {save_path}")

    return image


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
    """
    # Placeholder return to avoid errors; replace with actual implementation
    return go.Figure()
