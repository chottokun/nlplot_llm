# 📝 nlplot
nlplot: Analysis and visualization module for Natural Language Processing 📈

## Description
Facilitates the visualization of natural language processing and provides quicker analysis

You can draw the following graph

1. [N-gram bar chart](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_uni-gram.html)
2. [N-gram tree Map](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_Tree%20of%20Most%20Common%20Words.html)
3. [Histogram of the word count](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_number%20of%20words%20distribution.html)
4. [wordcloud](https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/wordcloud.png)
5. [co-occurrence networks](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_Co-occurrence%20network.html)
6. [sunburst chart](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_sunburst%20chart.html)

（Tested in English and Japanese）

## Requirement
- [python package](https://github.com/takapy0210/nlplot/blob/master/requirements.txt)

## Installation
```sh
pip install nlplot
```

I've posted on [this blog](https://www.takapy.work/entry/2020/05/17/192947) about the specific use. (Japanese)

And, The sample code is also available [in the kernel of kaggle](https://www.kaggle.com/takanobu0210/twitter-sentiment-eda-using-nlplot). (English)

## Quick start - Data Preparation

The column to be analyzed must be a space-delimited string

```python
# sample data
target_col = "text"
texts = [
    "Think rich look poor",
    "When you come to a roadblock, take a detour",
    "When it is dark enough, you can see the stars",
    "Never let your memories be greater than your dreams",
    "Victory is sweetest when you’ve known defeat"
    ]
df = pd.DataFrame({target_col: texts})
df.head()
```

|    |  text  |
| ---- | ---- |
|  0  |  Think rich look poor |
|  1  |  When you come to a roadblock, take a detour |
|  2  |  When it is dark enough, you can see the stars |
|  3  |  Never let your memories be greater than your dreams  |
|  4  |  Victory is sweetest when you’ve known defeat  |


## Quick start - Python API
```python
import nlplot
import pandas as pd # Required for DataFrame creation

# Sample data (ensure this is defined before use)
target_col = "text"
texts = [
    "Think rich look poor",
    "When you come to a roadblock, take a detour",
    "When it is dark enough, you can see the stars",
    "Never let your memories be greater than your dreams",
    "Victory is sweetest when you’ve known defeat"
]
df = pd.DataFrame({target_col: texts})

# Initialize NLPlot
# You can specify a custom font path for word clouds.
# If font_path is not provided or the specified font is not found,
# a default bundled font will be used.
# npt = nlplot.NLPlot(df, target_col='text', font_path='/path/to/your/font.ttf')
npt = nlplot.NLPlot(df, target_col='text')

# Stopword calculations can be performed.
# These stopwords will be automatically used by plotting methods unless overridden.
stopwords = npt.get_stopword(top_n=1, min_freq=0) # Adjusted for small sample

# 1. N-gram bar chart
npt.bar_ngram(title='uni-gram', ngram=1, top_n=50, stopwords=stopwords)
npt.bar_ngram(title='bi-gram', ngram=2, top_n=50, stopwords=stopwords)

# 2. N-gram tree Map
npt.treemap(title='Tree of Most Common Words', ngram=1, top_n=30, stopwords=stopwords)

# 3. Histogram of the word count
npt.word_distribution(title='Word Count Distribution')

# 4. wordcloud
# You can also specify custom stopwords or a different font path here.
npt.wordcloud(
    stopwords=stopwords,
    colormap='tab20_r',
    # font_path='/path/to/another/font.ttf' # Optional: override instance font
)

# 5. co-occurrence networks
# Adjust min_edge_frequency based on your dataset size.
# For the small sample, min_edge_frequency=0 or 1 might be needed to see a graph.
npt.build_graph(stopwords=stopwords, min_edge_frequency=1)
# Expected output like: node_size:X, edge_size:Y
npt.co_network(title='Co-occurrence Network')

# 6. sunburst chart
# Ensure build_graph has been called and resulted in some nodes/edges.
if not npt.node_df.empty:
    npt.sunburst(title='Sunburst Chart', colorscale=True)
else:
    print("Node data is empty, skipping sunburst chart.")

```

## Document
TBD

## Test
```sh
cd tests
pytest
```

## Other

- Plotly is used for most interactive figures.
  - https://plotly.com/python/
- NetworkX is used for co-occurrence network calculations.
  - https://networkx.github.io/documentation/stable/tutorial.html
- WordCloud library is used for generating word clouds.
  - By default, `nlplot` uses a bundled version of the MPLUS font (mplus-1c-regular) for word clouds to support Japanese and English.
  - You can specify a custom font using the `font_path` parameter in `NLPlot()` constructor or `npt.wordcloud()`.
  - `nlplot` includes enhanced handling for cases where specified fonts are not found or are invalid, attempting to fall back to a default font and providing warnings to the user.
  - MPLUS Font: https://mplus-fonts.osdn.jp/about.html
