# 📝 nlplot
nlplot: Analysis and visualization module for Natural Language Processing 📈

## Key Features
- N-gram bar charts and tree maps
- Word count histograms and word clouds
- Co-occurrence networks and sunburst charts
- **New:** Japanese text analysis (token counts, POS ratios) using Janome.
- **New (Experimental):** LLM-powered text analysis using [Langchain](https://python.langchain.com/), supporting providers like OpenAI and Ollama for:
    - Sentiment Analysis
    - Text Categorization (single and multi-label)
    - Text Chunking (via internal helper methods, useful for preprocessing long texts for LLMs)

## Description
Facilitates the visualization of natural language processing and provides quicker analysis. Now with experimental LLM capabilities for advanced text understanding.

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
- Core dependencies: `pandas`, `numpy`, `plotly>=4.12.0`, `matplotlib`, `wordcloud`, `pillow`, `networkx`, `seaborn`, `tqdm`
- For Japanese text features: `janome`
- For LLM-based features (experimental): `langchain`, `langchain-openai`, `langchain-community`, `openai`
- See `requirements.txt` for details.

## Installation
```sh
pip install nlplot
```
This will install `nlplot` along with all its dependencies, including those for Japanese text analysis (Janome) and LLM features (Langchain and related packages).

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

## Quick Start - Japanese Text Features (New)

`nlplot` can perform basic morphological analysis on Japanese text using [Janome](https://mocobeta.github.io/janome/)
to calculate various text features.

```python
import nlplot
import pandas as pd

# Sample Japanese texts
jp_texts = [
    "猫が窓から顔を出した。",
    "非常に美しい花が咲いている。",
    "今日は良い天気ですね。散歩に行きましょうか。",
    "素晴らしい一日でした！"
]
jp_df = pd.DataFrame({'text': jp_texts})

# Initialize NLPlot (ensure target_col contains Japanese text)
npt_jp = nlplot.NLPlot(jp_df, target_col='text')

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

## ✨ Quick Start - LLM-Powered Text Analysis (Experimental)
Leverage Large Language Models (LLMs) for sentiment analysis and text categorization. `nlplot` uses [Langchain](https://python.langchain.com/) for flexible LLM integration.

**Important Notes:**
- Ensure `langchain` and provider-specific packages (e.g., `openai`, `langchain-openai`, `langchain-community`) are installed (they are dependencies of `nlplot`).
- **API Keys & Costs:** Using cloud LLMs (e.g., OpenAI) requires API keys and may incur costs. Set your API key as an environment variable (e.g., `OPENAI_API_KEY`) or pass it as an argument.
- **Local LLMs (Ollama):** Requires a running Ollama server with models downloaded (e.g., `ollama pull llama2`). The default URL is `http://localhost:11434`.

### LLM Sentiment Analysis
```python
# Assuming npt_llm (NLPlot instance) and sentiment_df are already created as in previous examples.
# For OpenAI (ensure OPENAI_API_KEY is set or passed via llm_config)
if nlplot.nlplot.LANGCHAIN_AVAILABLE: # Check if Langchain components were imported successfully by nlplot
    try:
        sentiment_openai_df = npt_llm.analyze_sentiment_llm(
            text_series=sentiment_df['text'],
            llm_provider="openai",
            model_name="gpt-3.5-turbo", # Or your preferred model
            # openai_api_key="your_key_here" # Alternative if not using env var
        )
        print("\\nOpenAI Sentiment Analysis Results:")
        print(sentiment_openai_df)
    except Exception as e:
        print(f"OpenAI sentiment analysis example failed: {e}")

    # For Ollama (ensure Ollama is running and model is pulled)
    try:
        sentiment_ollama_df = npt_llm.analyze_sentiment_llm(
            text_series=sentiment_df['text'],
            llm_provider="ollama",
            model_name="llama2" # Replace with your available Ollama model
            # base_url="http://custom_ollama_host:11434" # If not default
        )
        print("\\nOllama Sentiment Analysis Results:")
        print(sentiment_ollama_df)
    except Exception as e:
        print(f"Ollama sentiment analysis example failed: {e}")
else:
    print("Langchain support is not available in this nlplot build/environment.")
```

### LLM Text Categorization
```python
# Assuming npt_llm is an NLPlot instance
texts_for_categorization = [
    "The stock market hit a new record high today.",
    "The local football team secured a dramatic win in the finals.",
    "Scientists announced a breakthrough in renewable energy research.",
    "This new AI gadget is great for gaming and office work."
]
categorization_df = pd.DataFrame({'text': texts_for_categorization})
defined_categories = ["finance", "sports", "science", "technology", "gaming", "office work"]

if nlplot.nlplot.LANGCHAIN_AVAILABLE:
    try:
        # Single-label categorization with OpenAI
        cat_openai_single_df = npt_llm.categorize_text_llm(
            text_series=categorization_df['text'],
            categories=defined_categories,
            llm_provider="openai",
            model_name="gpt-3.5-turbo",
            multi_label=False
        )
        print("\\nOpenAI Single-label Categorization Results:")
        print(cat_openai_single_df)

        # Multi-label categorization with Ollama
        cat_ollama_multi_df = npt_llm.categorize_text_llm(
            text_series=categorization_df['text'], # Can use the same series
            categories=defined_categories,
            llm_provider="ollama",
            model_name="llama2", # Replace with your model
            multi_label=True
        )
        print("\\nOllama Multi-label Categorization Results:")
        print(cat_ollama_multi_df)
    except Exception as e:
        print(f"LLM categorization example failed: {e}")
else:
    print("Langchain support is not available. Skipping LLM categorization examples.")
```

### LLM Text Summarization
Generate concise summaries of your texts. Supports chunking for long documents.
```python
# Assuming npt_llm is an NLPlot instance
long_text_series = pd.Series([
    "This is the first long document. It contains multiple sentences and discusses various topics that need to be condensed into a short summary. The goal is to capture the main essence of this text efficiently.",
    "Another document here, also quite lengthy. It explores different ideas and presents several arguments. Summarizing this will help in quick understanding. It talks about AI, programming, and the future of technology."
])

if nlplot.nlplot.LANGCHAIN_AVAILABLE:
    try:
        # Summarization with chunking (default)
        summaries_df = npt_llm.summarize_text_llm(
            text_series=long_text_series,
            llm_provider="openai", # or "ollama"
            model_name="gpt-3.5-turbo", # replace with your model
            # openai_api_key="your_key" # if not in env
            # chunk_size=1000, # Default
            # chunk_overlap=100, # Default
            # chunk_prompt_template_str="Summarize this: {text}", # Optional
            # combine_prompt_template_str="Combine these summaries: {text}" # Optional
        )
        print("\\nLLM Text Summarization Results:")
        print(summaries_df)

        # Example of direct summarization without chunking for shorter texts
        short_text_series = pd.Series(["A very short text to summarize directly."])
        short_summary_df = npt_llm.summarize_text_llm(
            text_series=short_text_series,
            llm_provider="openai",
            model_name="gpt-3.5-turbo",
            use_chunking=False
        )
        print("\\nLLM Short Text Summarization (No Chunking):")
        print(short_summary_df)

    except Exception as e:
        print(f"LLM summarization example failed: {e}")
else:
    print("Langchain support is not available. Skipping LLM summarization examples.")

```

### ⚙️ Underlying LLM Helper Methods
The LLM functionalities are built upon a few core methods:
- `_get_llm_client(llm_provider, model_name, **kwargs)`: Initializes and returns a Langchain chat model client for the specified provider (e.g., "openai", "ollama") and model. Handles API key loading (e.g., `OPENAI_API_KEY` from env or kwargs) and other configurations.
- `_chunk_text(text_to_chunk, strategy, chunk_size, chunk_overlap, **splitter_kwargs)`: Splits long texts into manageable chunks using different strategies (e.g., "recursive_char", "character"). This is useful for processing texts that exceed LLM context window limits, although `analyze_sentiment_llm` and `categorize_text_llm` currently process each text in a series individually.

These methods can be used for more custom LLM workflows if needed, though they are primarily for internal use by the higher-level analysis functions.

## 🚀 Streamlit Demo Application
A Streamlit demo application, `streamlit_app.py`, is included in the repository to showcase the LLM functionalities.

**To run the demo:**
1. Ensure all dependencies are installed:
   ```sh
   pip install nlplot streamlit pandas langchain openai langchain-community
   # If you cloned the repo and want to use the local nlplot version:
   # pip install -e .
   ```
2. Set your `OPENAI_API_KEY` environment variable if using OpenAI.
3. If using Ollama, ensure your Ollama server is running and models are downloaded (e.g., `ollama pull llama2`).
4. Run the Streamlit app from the repository root:
   ```sh
   streamlit run streamlit_app.py
   ```
This will open a web interface where you can input text, configure LLM settings, choose an analysis type (sentiment or categorization), and view the results.

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
- For Japanese text analysis features (e.g., `get_japanese_text_features`), `nlplot` uses Janome for morphological analysis.
- **LLM Features (Experimental):**
    - LLM-based features like `analyze_sentiment_llm` and `categorize_text_llm` depend on `langchain` and associated provider packages (e.g., `openai`, `langchain-openai`, `langchain-community`). These are included as dependencies.
    - **API Keys & Costs:** Using cloud-based LLMs (e.g., OpenAI) typically requires API keys and may incur costs. Set API keys via environment variables (e.g., `OPENAI_API_KEY`) or pass them as arguments to the respective methods.
    - **Local LLMs (Ollama):** To use Ollama, ensure your Ollama server is running and the desired models are downloaded (e.g., `ollama pull llama2`). The default connection URL is `http://localhost:11434` but can be configured.
    - **Output Variability:** LLM outputs can vary. The parsing logic for sentiment and categories is designed to be somewhat flexible, but for critical applications, review and potentially customize prompts or parsing.
