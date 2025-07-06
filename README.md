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
    - Access to 100+ LLMs (OpenAI, Azure, Ollama, Cohere, Anthropic, etc.) through a unified interface via LiteLLM.
- **Customizable Prompts:** Easily customize prompts for all LLM-powered analyses.
- **LLM Response Caching:** Built-in support for caching LLM API responses to save costs and speed up repeated analyses (uses `diskcache`).
- **Asynchronous Operations:** Provides asynchronous versions of LLM analysis methods (`*_async`) for improved performance with multiple texts.
- **Text Chunking:** Helper methods for splitting long texts, useful for LLM preprocessing (uses Langchain TextSplitters).

## Acknowledgements

This library is built upon the excellent `nlplot` library, originally developed by Takanobu Nozawa (takapy0210). We extend our sincere gratitude for the original author's great work and elegant design.

The core visualization functionalities are inherited from the original work. `nlplot_llm` focuses on extending it with LLM-based analysis features.

- **Original Repository:** https://github.com/takapy0210/nlplot
- **Author's Blog Post (Japanese):** https://www.takapy.work/entry/2020/05/17/192947


## Description
`nlplot_llm` extends the original `nlplot` library by integrating robust LLM capabilities through LiteLLM. This allows for advanced text analysis tasks like sentiment analysis, categorization, and summarization across a wide range of language models—with customizable prompts, response caching, and asynchronous operations—while retaining useful NLP visualizations.

## Requirement
- Python 3.7+
- Core dependencies: `pandas`, `numpy`, `plotly>=4.12.0`, `matplotlib`, `wordcloud`, `pillow`, `networkx`, `seaborn`, `tqdm`.
- For Japanese text features: `janome`.
- For LLM-based features: `litellm>=1.0` (core dependency).
- For LLM response caching: `diskcache>=5.0.0` (core dependency).
- For text chunking (used by LLM features): `langchain-text-splitters` (core dependency).
See `requirements.txt` for full details.

## Installation
```sh
pip install nlplot_llm
```
This will install `nlplot_llm` along with its core dependencies, including `litellm`, `diskcache`, and `langchain-text-splitters`.
For Japanese text analysis, `janome` is required (install separately if needed: `pip install janome`).

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
# Font handling: Provide a valid font_path or ensure system fonts are available for wordcloud.
# Caching: LLM responses are cached by default. Customize via use_cache, cache_dir, etc.
# npt = NLPlotLLM(df, target_col='text', font_path='/path/to/your/font.ttf', use_cache=False)
npt = NLPlotLLM(df, target_col='text') # Uses default font logic and default caching (enabled)

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
- **Dependencies:** Ensure `litellm`, `diskcache`, and `langchain-text-splitters` are installed (they are core dependencies).
- **LLM Configuration:** Depending on the LLM provider (e.g., OpenAI, Azure, Ollama), set up API keys or environment variables as per LiteLLM's documentation. For example, set `OPENAI_API_KEY` for OpenAI models. For local models like Ollama, ensure the server is running.
- **Costs:** Using cloud-based LLMs may incur costs.
- **Caching:** LLM responses are cached by default to `./.cache/nlplot_llm`. You can customize this via `NLPlotLLM` constructor arguments (`use_cache`, `cache_dir`, `cache_expire`, `cache_size_limit`) or disable it per-method call with `use_cache=False`.

### LLM Sentiment Analysis
```python
from nlplot_llm import NLPlotLLM
import pandas as pd
import asyncio # For async examples

# Initialize NLPlotLLM (uses default caching: enabled)
# npt = NLPlotLLM(df, target_col='text', use_cache=True, cache_dir='./my_custom_cache', cache_expire=3600)
# For this example, assume 'npt' is already initialized as in the main Quick Start.

sentiment_texts = pd.Series([
    "I love this product, it's absolutely fantastic!",
    "This is the worst experience I have ever had.",
    "The weather today is just okay, nothing special."
])

# --- Synchronous Examples ---
# Example with OpenAI model (ensure OPENAI_API_KEY is set or pass via litellm_kwargs)
try:
    sentiment_df_openai = npt.analyze_sentiment_llm(
        text_series=sentiment_texts,
        model="openai/gpt-4o",
        # litellm_kwargs={"api_key": "YOUR_KEY"} # Example of passing API key directly
    )
    print("\\nSentiment Analysis Results (OpenAI gpt-4o):")
    print(sentiment_df_openai)
except Exception as e:
    print(f"OpenAI sentiment analysis failed: {e}")

# Example with Ollama model, custom prompt, and caching disabled for this call
# Ensure Ollama server is running and 'gemma' model is pulled.
custom_prompt = "Is this text positive, negative, or neutral? Answer with one word. Text: {text}"
try:
    sentiment_df_ollama = npt.analyze_sentiment_llm(
        text_series=sentiment_texts,
        model="ollama/gemma",
        prompt_template_str=custom_prompt,
        use_cache=False, # Disable cache for this specific call
        litellm_kwargs={"temperature": 0.1}
    )
    print("\\nSentiment Analysis Results (Ollama gemma, custom prompt, no cache):")
    print(sentiment_df_ollama)
except Exception as e:
    print(f"Ollama sentiment analysis failed: {e}")

# --- Asynchronous Example ---
# Requires an async environment (e.g., running within an async function)
async def main_sentiment_async():
    try:
        sentiment_df_async = await npt.analyze_sentiment_llm_async(
            text_series=sentiment_texts,
            model="openai/gpt-4o",
            concurrency_limit=5 # Limit concurrent requests
        )
        print("\\nAsync Sentiment Analysis Results (OpenAI gpt-4o):")
        print(sentiment_df_async)
    except Exception as e:
        print(f"Async sentiment analysis failed: {e}")

# if __name__ == "__main__":
# asyncio.run(main_sentiment_async())
```

### LLM Text Categorization
```python
# Assuming 'npt' is an initialized NLPlotLLM instance.
texts_for_categorization = pd.Series([
    "The stock market hit a new record high today.",
    "The local football team secured a dramatic win in the finals.",
    "Scientists announced a breakthrough in renewable energy research."
])
defined_categories = ["finance", "sports", "science", "technology"]

# --- Synchronous Examples ---
try:
    cat_single_df = npt.categorize_text_llm(
        text_series=texts_for_categorization,
        categories=defined_categories,
        model="openai/gpt-4o",
        multi_label=False
    )
    print("\\nSingle-label Categorization Results (OpenAI gpt-4o):")
    print(cat_single_df)

    cat_multi_df_ollama = npt.categorize_text_llm(
        text_series=texts_for_categorization,
        categories=defined_categories,
        model="ollama/gemma",
        multi_label=True,
        prompt_template_str="Categories: {categories}. Text: {text}. Relevant categories (comma-separated)?",
        use_cache=True # Explicitly enable cache (though it's default)
    )
    print("\\nMulti-label Categorization Results (Ollama gemma, custom prompt):")
    print(cat_multi_df_ollama)
except Exception as e:
    print(f"LLM categorization failed: {e}")

# --- Asynchronous Example ---
async def main_categorize_async():
    try:
        cat_df_async = await npt.categorize_text_llm_async(
            text_series=texts_for_categorization,
            categories=defined_categories,
            model="ollama/gemma",
            multi_label=True,
            concurrency_limit=2
        )
        print("\\nAsync Multi-label Categorization Results (Ollama gemma):")
        print(cat_df_async)
    except Exception as e:
        print(f"Async categorization failed: {e}")

# if __name__ == "__main__":
# asyncio.run(main_categorize_async())
```

### LLM Text Summarization
```python
# Assuming 'npt' is an initialized NLPlotLLM instance.
long_text_series = pd.Series([
    "Artificial intelligence (AI) is rapidly transforming various industries. Its applications range from natural language processing and computer vision to robotics and healthcare. The potential of AI to solve complex problems and drive innovation is immense, but it also raises ethical considerations that need careful attention.",
    "The global economy is facing a period of uncertainty due to geopolitical tensions and inflationary pressures. Central banks are tightening monetary policies, leading to concerns about a potential recession. Businesses are adapting by focusing on efficiency and resilience."
])

# --- Synchronous Example (Chunking with custom prompts) ---
custom_chunk_prompt = "Briefly summarize this part: {text}"
custom_combine_prompt = "Combine these summaries into a coherent overview: {text}"
try:
    summaries_df_openai = npt.summarize_text_llm(
        text_series=long_text_series,
        model="openai/gpt-4o",
        use_chunking=True,
        chunk_prompt_template_str=custom_chunk_prompt,
        combine_prompt_template_str=custom_combine_prompt,
        chunk_size=500, # Smaller chunk for example
        litellm_kwargs={"max_tokens": 100} # Limit summary length
    )
    print("\\nLLM Text Summarization (OpenAI gpt-4o, Chunked, Custom Prompts):")
    print(summaries_df_openai)
except Exception as e:
    print(f"OpenAI summarization failed: {e}")

# --- Asynchronous Example (Direct summarization) ---
async def main_summarize_async():
    short_texts = pd.Series(["This is a short text. It should be summarized directly."])
    try:
        summaries_df_async = await npt.summarize_text_llm_async(
            text_series=short_texts,
            model="ollama/gemma",
            use_chunking=False, # Direct summarization
            prompt_template_str="One sentence summary: {text}",
            concurrency_limit=3
        )
        print("\\nAsync Direct Summarization (Ollama gemma):")
        print(summaries_df_async)
    except Exception as e:
        print(f"Async summarization failed: {e}")

# if __name__ == "__main__":
# asyncio.run(main_summarize_async())
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
   # Ensure Janome is installed if you use Japanese text features with the core library:
   # pip install janome
   ```
2. **Configure LLM Provider:**
   - Set necessary environment variables for your chosen LLM provider (e.g., `OPENAI_API_KEY`, `AZURE_API_KEY`, `COHERE_API_KEY`).
   - For local models like Ollama, ensure your Ollama server is running (`ollama serve`) and the desired models are pulled (e.g., `ollama pull mistral`).
   - Refer to the [LiteLLM Documentation](https://docs.litellm.ai/docs/providers) for detailed provider-specific setup.
   - You can also provide API keys and other parameters directly in the Streamlit app's sidebar if preferred over environment variables.
3. **Run the Streamlit App:**
   From the repository root, execute:
   ```sh
   streamlit run streamlit_app.py
   ```
This will open a web interface where you can:
- Input text for analysis.
- Specify the LiteLLM model string (e.g., `openai/gpt-4o`, `ollama/gemma`, `azure/your-deployment`).
- Adjust common LLM parameters like Temperature and Max Tokens.
- Enable or disable LLM response caching.
- Optionally override API Key and API Base URL.
- Choose an analysis type (Sentiment, Categorization, Summarization).
- For each analysis type, you can further customize options, including the **prompt templates** used by the LLM.
- View the results directly in the app.

## Document
API documentation can be generated using Sphinx. Navigate to the `docs/` directory and run:
```sh
make html
```
The generated documentation will be available in `docs/_build/html/index.html`.
(More detailed user guides and explanations are TBD).

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
