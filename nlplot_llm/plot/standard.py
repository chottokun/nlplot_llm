import os
import datetime
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
import plotly.express as px
from wordcloud import WordCloud
from PIL import Image
from typing import Optional

from ..utils.common import generate_freq_df

def bar_ngram(nlplot_instance, title: str = None, xaxis_label: str = '', yaxis_label: str = '', ngram: int = 1, top_n: int = 50, width: int = 800, height: int = 1100, color: str = None, horizon: bool = True, stopwords: list = [], verbose: bool = True, save: bool = False) -> plotly.graph_objs.Figure:
    current_stopwords = list(set(stopwords + nlplot_instance.default_stopwords))
    temp_series = nlplot_instance.df[nlplot_instance.target_col].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    nlplot_instance.ngram_df = generate_freq_df(temp_series, n_gram=ngram, top_n=top_n, stopwords=current_stopwords, verbose=verbose)
    if nlplot_instance.ngram_df.empty:
        print("Warning: No data to plot for bar_ngram after processing. Empty DataFrame.")
        return go.Figure()
    fig = px.bar(nlplot_instance.ngram_df.sort_values('word_count') if horizon else nlplot_instance.ngram_df,
                 y='word' if horizon else 'word_count', x='word_count' if horizon else 'word',
                 text='word_count', orientation='h' if horizon else 'v')
    fig.update_traces(texttemplate='%{text:.2s}', textposition='auto', marker_color=color)
    fig.update_layout(title=str(title) if title else 'N-gram Bar Chart', xaxis_title=str(xaxis_label), yaxis_title=str(yaxis_label), width=width, height=height)
    if save: nlplot_instance.save_plot(fig, title if title else "bar_ngram")
    return fig

def treemap(nlplot_instance, title: str = None, ngram: int = 1, top_n: int = 50, width: int = 1300, height: int = 600, stopwords: list = [], verbose: bool = True, save: bool = False) -> plotly.graph_objs.Figure:
    current_stopwords = list(set(stopwords + nlplot_instance.default_stopwords))
    temp_series = nlplot_instance.df[nlplot_instance.target_col].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
    nlplot_instance.treemap_df = generate_freq_df(temp_series, n_gram=ngram, top_n=top_n, stopwords=current_stopwords, verbose=verbose)
    if nlplot_instance.treemap_df.empty or 'word' not in nlplot_instance.treemap_df.columns or 'word_count' not in nlplot_instance.treemap_df.columns:
        print("Warning: No data to plot for treemap after processing. Empty or malformed DataFrame.")
        return go.Figure()
    fig = px.treemap(nlplot_instance.treemap_df, path=[px.Constant("all"), 'word'], values='word_count')
    fig.update_layout(title=str(title) if title else 'Treemap', width=width, height=height)
    if save: nlplot_instance.save_plot(fig, title if title else "treemap")
    return fig

def word_distribution(nlplot_instance, title: str = None, xaxis_label: str = '', yaxis_label: str = '', width: int = 1000, height: int = 600, color: str = None, template: str = 'plotly', bins: int = None, save: bool = False) -> plotly.graph_objs.Figure:
    col_name = nlplot_instance.target_col + '_length'
    nlplot_instance.df.loc[:, col_name] = nlplot_instance.df[nlplot_instance.target_col].apply(lambda x: len(x) if isinstance(x, list) else 0)
    if nlplot_instance.df[col_name].empty:
        print("Warning: No data to plot for word_distribution.")
        return go.Figure()
    fig = px.histogram(nlplot_instance.df, x=col_name, color=color, template=template, nbins=bins)
    fig.update_layout(title=str(title) if title else 'Word Distribution',
                      xaxis_title=str(xaxis_label) if xaxis_label else 'Number of Words',
                      yaxis_title=str(yaxis_label) if yaxis_label else 'Frequency',
                      width=width, height=height)
    if save: nlplot_instance.save_plot(fig, title if title else "word_distribution")
    return fig

def wordcloud(nlplot_instance, width: int = 800, height: int = 500, max_words: int = 100, max_font_size: int = 80, stopwords: list = [], colormap: str = None, mask_file: str = None, font_path: str = None, save: bool = False) -> None:
    wc_font_path = None
    if font_path is not None:
        if os.path.exists(font_path):
            wc_font_path = font_path
        else:
            print(f"Warning: Specified font_path '{font_path}' for wordcloud not found. Will attempt to use WordCloud default.")
    elif nlplot_instance.font_path is not None:
        if os.path.exists(nlplot_instance.font_path):
            wc_font_path = nlplot_instance.font_path
        else:
            print(f"Warning: Instance font_path '{nlplot_instance.font_path}' not found. Will attempt to use WordCloud default.")

    if wc_font_path is None:
        print("Info: No valid font_path provided. WordCloud will attempt to use its default system font. Ensure a font is available for your language.")

    mask = None
    if mask_file and os.path.exists(mask_file):
        try: mask = np.array(Image.open(mask_file))
        except PermissionError: print(f"Warning: Permission denied to read mask file {mask_file}. Proceeding without mask."); mask = None
        except IOError as e: print(f"Warning: Could not load mask file {mask_file} due to an IO error: {e}. Proceeding without mask."); mask = None
        except Exception as e: print(f"Warning: Could not load mask file {mask_file}: {e}. Proceeding without mask."); mask = None
    elif mask_file: print(f"Warning: Mask file {mask_file} not found. Proceeding without mask.")

    processed_texts = [' '.join(map(str, item)) if isinstance(item, list) else (item if isinstance(item, str) else "") for item in nlplot_instance.df[nlplot_instance.target_col]]
    if not processed_texts:
        print("Warning: No text data available for wordcloud after processing.")
        return
    text_corpus = ' '.join(processed_texts)
    current_stopwords = set(stopwords + nlplot_instance.default_stopwords)
    if not text_corpus.strip():
        print("Warning: Text corpus is empty after processing stopwords for wordcloud.")
        return

    try:
        wordcloud_instance = WordCloud(
            font_path=wc_font_path,
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
    except OSError as e:
         print(f"Error during WordCloud generation (possibly font-related): {e}. If no font_path was specified, ensure system fonts are available. If a font_path was specified ('{wc_font_path}'), check its validity.")
         return
    except ValueError as e:
        if "empty" in str(e).lower() or "zero" in str(e).lower():
            print(f"Warning: WordCloud could not be generated. All words might have been filtered out or corpus is empty. Details: {e}")
        else:
            print(f"An unexpected ValueError occurred during WordCloud generation: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during WordCloud generation: {e}")
        return

    img_array = wordcloud_instance.to_array()
    pil_img = Image.fromarray(img_array)

    if save:
        date_str = pd.to_datetime(datetime.datetime.now()).strftime('%Y-%m-%d')
        filename = f"{date_str}_wordcloud_plot_wordcloud.png"
        full_save_path = os.path.join(nlplot_instance.output_file_path, filename)
        try:
            os.makedirs(nlplot_instance.output_file_path, exist_ok=True)
            pil_img.save(full_save_path)
            print(f"Wordcloud image saved to {full_save_path}")
        except PermissionError:
            print(f"Error: Permission denied to save wordcloud image to '{full_save_path}'. Please check directory permissions.")
        except Exception as e_save:
            print(f"Error saving wordcloud image to '{full_save_path}': {e_save}")

    try:
        from IPython.display import display
        display(pil_img)
    except ImportError:
        pass

    return pil_img

def plot_japanese_text_features(
    nlplot_instance,
    features_df: pd.DataFrame,
    target_feature: str,
    title: Optional[str] = None,
    save: bool = False,
    **kwargs
) -> Optional[plotly.graph_objs.Figure]:
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
        nlplot_instance.save_plot(fig, filename_prefix)

    return fig
