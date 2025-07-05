# 📝 nlplot_llm
nlplot_llm: Enhanced Natural Language Processing analysis and visualization with diverse LLM integration via LiteLLM 📈

## Key Features
- **Core Visualizations:** (Retained from original nlplot)
    - N-gram bar charts and tree maps
    - Word count histograms and word clouds
    - Co-occurrence networks and sunburst charts
- **Japanese Text Analysis:** (Retained)
    - Morphological analysis (token counts, POS ratios) using Janome.
- **LLM-Powered Text Analysis (via LiteLLM):**
    - **Sentiment Analysis:** Analyze sentiment of texts.
    - **Text Categorization:** Classify texts into predefined categories (single and multi-label).
    - **Text Summarization:** Generate summaries of texts, with support for chunking long documents.
    - Access to 100+ LLMs (OpenAI, Azure, Ollama, Cohere, Anthropic, etc.) through a unified interface.
- **Text Chunking:** Helper methods for splitting long texts, useful for LLM preprocessing (uses Langchain TextSplitters).

## Description
`nlplot_llm` extends the original `nlplot` library by integrating robust LLM capabilities through LiteLLM. This allows for advanced text analysis tasks like sentiment analysis, categorization, and summarization across a wide range of language models, while retaining useful NLP visualizations.

You can draw the following graph

1. [N-gram bar chart](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_uni-gram.html)
2. [N-gram tree Map](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_Tree%20of%20Most%20Common%20Words.html)
3. [Histogram of the word count](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_number%20of%20words%20distribution.html)
4. [wordcloud](https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/wordcloud.png)
5. [co-occurrence networks](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_Co-occurrence%20network.html)
6. [sunburst chart](https://htmlpreview.github.io/?https://github.com/takapy0210/takapy_blog/blob/master/nlp/twitter_analytics_using_nlplot/2020-05-17_sunburst%20chart.html)

（Tested in English and Japanese）

## Requirement
- Python 3.7+
- Core dependencies: `pandas`, `numpy`, `plotly>=4.12.0`, `matplotlib`, `wordcloud`, `pillow`, `networkx`, `seaborn`, `tqdm`.
- For Japanese text features: `janome`.
- For LLM-based features: `litellm`.
- For text chunking (used by LLM features): `langchain-text-splitters` (or `langchain_text_splitters` if you prefer to list the specific sub-package).
See `requirements.txt` for full details.

## Installation
```sh
pip install nlplot_llm
```
This will install `nlplot_llm` along with its core dependencies.
To use LLM features, ensure `litellm` is installed (it's listed in `requirements.txt`).
For Japanese text analysis, `janome` is required.
For text chunking (often used with LLMs), `langchain-text-splitters` is needed.

I've posted on [this blog](https://www.takapy.work/entry/2020/05/17/192947) about the specific use of the original nlplot. (Japanese)

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
from nlplot_llm import NLPlotLLM, get_colorpalette, generate_freq_df # Updated import
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

# Initialize NLPlotLLM
# Font handling has changed: nlplot_llm no longer bundles a default font.
# Provide a valid font_path or ensure system fonts are available for wordcloud.
# npt = NLPlotLLM(df, target_col='text', font_path='/path/to/your/font.ttf')
npt = NLPlotLLM(df, target_col='text')

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

## Quick Start - Japanese Text Features (New)

`nlplot` can perform basic morphological analysis on Japanese text using [Janome](https://mocobeta.github.io/janome/)
to calculate various text features.

```python
from nlplot_llm import NLPlotLLM # Updated import
import pandas as pd

# Sample Japanese texts
jp_texts = [
    "猫が窓から顔を出した。",
    "非常に美しい花が咲いている。",
    "今日は良い天気ですね。散歩に行きましょうか。",
    "素晴らしい一日でした！"
]
jp_df = pd.DataFrame({'text': jp_texts})

# Initialize NLPlotLLM (ensure target_col contains Japanese text)
npt_jp = NLPlotLLM(jp_df, target_col='text') # Updated class name

# Calculate Japanese text features
# This requires Janome to be installed.
jp_features_df = npt_jp.get_japanese_text_features(jp_df['text'])
print("Japanese Text Features:")
print(jp_features_df.head())
# Expected output (example structure):
#                        text  total_tokens  avg_token_length  noun_ratio  verb_ratio  adj_ratio  punctuation_count
# 0          猫が窓から顔を出した。             9          1.250000    0.333333    0.111111   0.000000                  1
# 1  非常に美しい花が咲いている。             9          1.625000    0.222222    0.111111   0.111111                  1
# ...

# Plot a specific feature (e.g., total_tokens)
if hasattr(npt_jp, 'plot_japanese_text_features') and not jp_features_df.empty: # Check if method exists
    fig_jp_tokens = npt_jp.plot_japanese_text_features(
        features_df=jp_features_df,
        target_feature='total_tokens',
        title='Distribution of Total Tokens (Japanese)'
    )
    # fig_jp_tokens.show() # In a Jupyter environment to display
    # To save:
    # npt_jp.plot_japanese_text_features(features_df=jp_features_df, target_feature='total_tokens', save=True)
else:
    print("Japanese text features plotting method not available or no features calculated.")

```

## ✨ Quick Start - LLM-Powered Text Analysis (with LiteLLM)
Leverage a wide range of Large Language Models (LLMs) for sentiment analysis, text categorization, and summarization. `nlplot_llm` uses [LiteLLM](https://litellm.ai/) to provide a unified interface to over 100 LLM providers.

**Important Notes:**
- Ensure `litellm` is installed (it's a core dependency of `nlplot_llm`).
- Depending on the LLM provider you choose (e.g., OpenAI, Azure, Cohere, Anthropic), you'll need to set up appropriate API keys or environment variables as per LiteLLM's documentation. For example, for OpenAI models, set `OPENAI_API_KEY`. For Ollama, ensure your Ollama server is running.
- Using cloud LLMs may incur costs.

### LLM Sentiment Analysis
```python
from nlplot_llm import NLPlotLLM # Ensure you have this import
import pandas as pd
# from nlplot_llm.core import LITELLM_AVAILABLE # Optional: to check if LiteLLM was imported by the library

# Initialize NLPlotLLM (replace with your actual DataFrame and target column)
sample_df_for_api = pd.DataFrame({'text_col': ["Initial text for NLPlotLLM."]})
npt = NLPlotLLM(sample_df_for_api, target_col='text_col')

sentiment_texts = pd.Series([
    "I love this product, it's absolutely fantastic!",
    "This is the worst experience I have ever had.",
    "The weather today is just okay, nothing special."
])

# Example with an OpenAI model via LiteLLM
# Ensure OPENAI_API_KEY environment variable is set, or pass api_key in litellm_kwargs parameter.
# if LITELLM_AVAILABLE: # You can use this check if LITELLM_AVAILABLE is exposed from nlplot_llm.core
try:
    sentiment_df = npt.analyze_sentiment_llm(
        text_series=sentiment_texts,
        model="openai/gpt-3.5-turbo", # LiteLLM model string
        # You can pass additional litellm_kwargs if needed, e.g.:
        # temperature=0.7, api_key="your_openai_key"
    )
    print("\\nSentiment Analysis Results (OpenAI via LiteLLM):")
    print(sentiment_df)
except Exception as e:
    print(f"Sentiment analysis example failed: {e}")

# Example with an Ollama model via LiteLLM (ensure Ollama server is running)
try:
    sentiment_df_ollama = npt.analyze_sentiment_llm(
        text_series=sentiment_texts,
        model="ollama/llama2" # Or your preferred Ollama model (e.g., "ollama/mistral")
        # litellm_kwargs={"api_base": "http://localhost:11434"} # Default, but can be specified
    )
    print("\\nSentiment Analysis Results (Ollama via LiteLLM):")
    print(sentiment_df_ollama)
except Exception as e:
    print(f"Ollama sentiment analysis example failed: {e}")
# else:
#     print("LiteLLM support is not available. Skipping LLM sentiment analysis examples.")
```

### LLM Text Categorization
```python
# Assuming npt is an instance of NLPlotLLM
texts_for_categorization = pd.Series([
    "The stock market hit a new record high today.",
    "The local football team secured a dramatic win in the finals.",
    "Scientists announced a breakthrough in renewable energy research.",
    "This new AI gadget is great for gaming and office work."
])
defined_categories = ["finance", "sports", "science", "technology", "gaming", "office work"]

# if LITELLM_AVAILABLE:
try:
    # Single-label categorization with an OpenAI model via LiteLLM
    cat_single_df = npt.categorize_text_llm(
        text_series=texts_for_categorization,
        categories=defined_categories,
        model="openai/gpt-3.5-turbo",
        multi_label=False
    )
    print("\\nSingle-label Categorization Results (OpenAI via LiteLLM):")
    print(cat_single_df)

    # Multi-label categorization with an Ollama model via LiteLLM
    cat_multi_df = npt.categorize_text_llm(
        text_series=texts_for_categorization,
        categories=defined_categories,
        model="ollama/llama2",
        multi_label=True
    )
    print("\\nMulti-label Categorization Results (Ollama via LiteLLM):")
    print(cat_multi_df)
except Exception as e:
    print(f"LLM categorization example failed: {e}")
# else:
#     print("LiteLLM support is not available. Skipping LLM categorization examples.")
```

### LLM Text Summarization
Generate concise summaries of your texts. Supports chunking for long documents.
```python
# Assuming npt is an instance of NLPlotLLM
long_text_series = pd.Series([
    "This is the first long document. It contains multiple sentences and discusses various topics that need to be condensed into a short summary. The goal is to capture the main essence of this text efficiently. It talks about artificial intelligence, machine learning, and natural language processing.",
    "Another document here, also quite lengthy. It explores different ideas and presents several arguments. Summarizing this will help in quick understanding. It talks about AI, programming, and the future of technology, including ethical considerations and societal impact."
])

# if LITELLM_AVAILABLE:
try:
    # Summarization with chunking (default) using an OpenAI model
    summaries_df = npt.summarize_text_llm(
        text_series=long_text_series,
        model="openai/gpt-3.5-turbo",
        # litellm_kwargs={"api_key": "your_openai_key"}
        # chunk_size=1000, # Default
        # chunk_overlap=100, # Default
        # chunk_prompt_template_str="Summarize this: {text}", # Optional
        # combine_prompt_template_str="Combine these summaries: {text}" # Optional
    )
    print("\\nLLM Text Summarization Results (OpenAI via LiteLLM, Chunked):")
    print(summaries_df)

    # Example of direct summarization without chunking for shorter texts
    short_text_series = pd.Series(["A very short text to summarize directly."])
    short_summary_df = npt.summarize_text_llm(
        text_series=short_text_series,
        model="openai/gpt-3.5-turbo",
        use_chunking=False,
        prompt_template_str="Provide a very brief summary of: {text}" # Custom prompt
    )
    print("\\nLLM Short Text Summarization (No Chunking):")
    print(short_summary_df)

except Exception as e:
    print(f"LLM summarization example failed: {e}")
# else:
#     print("LiteLLM support is not available. Skipping LLM summarization examples.")
```

### ⚙️ Underlying Helper Methods (Internal)
- `_chunk_text(text_to_chunk, strategy, chunk_size, chunk_overlap, **splitter_kwargs)`: Splits long texts using Langchain TextSplitters. This is used internally by `summarize_text_llm` when `use_chunking=True`. (Note: `_get_llm_client` is no longer used as LiteLLM is called directly).

## 🚀 Streamlit Demo Application
A Streamlit demo application, `streamlit_app.py`, is included in the repository to showcase the LLM functionalities with `nlplot_llm`.

**To run the demo:**
1. Ensure all dependencies are installed:
   ```sh
   pip install nlplot_llm streamlit pandas litellm langchain-text-splitters
   # If you cloned the repo and want to use the local nlplot_llm version:
   # pip install -e .
   ```
2. Configure your LLM provider credentials (e.g., set `OPENAI_API_KEY` environment variable for OpenAI models, ensure Ollama server is running for `ollama/...` models). Refer to [LiteLLM's documentation](https://docs.litellm.ai/docs/providers) for provider-specific setup.
3. Run the Streamlit app from the repository root:
   ```sh
   streamlit run streamlit_app.py
   ```
This will open a web interface where you can input text, specify the LiteLLM model string, choose an analysis type, and view the results.

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
  - `nlplot_llm` no longer bundles a default font. Users should specify a `font_path` in `NLPlotLLM()` or `wordcloud()` method, or ensure system fonts are available for WordCloud to use its default.
- For Japanese text analysis features (e.g., `get_japanese_text_features`), `nlplot_llm` uses Janome.
- **LLM Features (via LiteLLM):**
    - All LLM-based features (`analyze_sentiment_llm`, `categorize_text_llm`, `summarize_text_llm`) now use `litellm` for broader LLM provider support.
    - **API Keys & Costs:** Using cloud-based LLMs typically requires API keys (set as environment variables like `OPENAI_API_KEY`, `AZURE_API_KEY`, etc., as per LiteLLM docs) and may incur costs. Specific keys or other parameters can also be passed via `litellm_kwargs` to the methods.
    - **Local LLMs (Ollama, etc.):** Ensure your local LLM server (e.g., Ollama) is running. You can specify the model using the provider prefix (e.g., `ollama/llama2`). The `api_base` can be set via `litellm_kwargs` if not the default.
    - **Output Variability:** LLM outputs can vary. Review and potentially customize prompts for critical applications.
