# nlplot_llm
現在、構築中です。このプロジェクトは [nlplot](https://github.com/takapy0210/nlplot) をベースに拡張しています。takapy0210 氏に感謝します。ありがとうございます。

**Enhanced NLP analysis & visualization with LLM integration via LiteLLM**

## Table of Contents

- [Features](#features)  
- [Installation](#installation)  
- [Quick Start](#quick-start)  
  - [Python API](#python-api)  
  - [Streamlit Demo](#streamlit-demo)  
- [Documentation](#documentation)  
- [Testing & CI](#testing--ci)  
- [Contributing](#contributing)  
- [License](#license)  

## Features

### Core Visualizations (from original nlplot)
- N-gram bar charts & tree maps  
- Word count histograms & word clouds  
- Co-occurrence networks & sunburst charts  

### Japanese Text Analysis
- Morphological analysis (token counts, POS ratios) using [Janome](https://mocobeta.github.io/janome/)  

### LLM-Powered Text Analysis (via [LiteLLM](https://litellm.ai/))
- **Sentiment Analysis**  
- **Text Categorization** (single- & multi-label)  
- **Text Summarization** (supports chunking long documents)  
- Unified interface to 100+ providers (OpenAI, Azure, Ollama, Cohere, Anthropic, etc.)  
- Customizable prompt templates  
- Response caching (via [diskcache](https://pypi.org/project/diskcache/))  
- Asynchronous operations (`*_async` methods)  
- Helpers for text chunking (via Langchain TextSplitters)  

## Installation

### From GitHub (latest)

```bash
pip install git+https://github.com/chottokun/nlplot_llm.git
```

### Editable local install (development)

```bash
pip install -e .
```

### Core Dependencies

Installs core dependencies (`litellm`, `diskcache`, `langchain-text-splitters`, etc.).

### Additional Dependencies

- Japanese analysis:

```bash
pip install janome
```

- Development requirements:

```bash
pip install -r requirements-dev.txt
```

## Quick Start

### Python API

```python
import pandas as pd
from nlplot_llm import NLPlotLLM, get_colorpalette, generate_freq_df

# Prepare data
texts = [
    "Think rich look poor",
    "When you come to a roadblock, take a detour",
    "When it is dark enough, you can see the stars",
    "Never let your memories be greater than your dreams",
    "Victory is sweetest when you’ve known defeat"
]
df = pd.DataFrame({"text": texts})

# Initialize
npt = NLPlotLLM(df, target_col="text")

# Stopwords (sample)
stopwords = npt.get_stopword(top_n=10, min_freq=1)

# 1. N-gram bar chart
npt.bar_ngram("Uni-gram", ngram=1, top_n=20, stopwords=stopwords)
npt.bar_ngram("Bi-gram",  ngram=2, top_n=20, stopwords=stopwords)

# 2. Word cloud
npt.wordcloud(stopwords=stopwords, colormap="tab20_r")

# 3. Co-occurrence network + sunburst
npt.build_graph(stopwords=stopwords, min_edge_frequency=1)
npt.co_network("Co-occurrence Network")
npt.sunburst("Sunburst Chart", colorscale=True)
```

### Streamlit Demo

A demo app (`streamlit_app.py`) showcases LLM integration.

```bash
# From repo root
pip install streamlit nlplot_llm pandas litellm langchain-text-splitters
streamlit run streamlit_app.py
```

Use the sidebar to choose model, prompts, caching, and analysis type.

## Streamlit Demo Notes

### Japanese Word Cloud Font
To correctly display Japanese characters in word clouds within the Streamlit demo, a Japanese-compatible font file is required.

1.  **Recommended Font**: We recommend using [IPAexGothic (ipaexg.ttf)](https://moji.or.jp/ipafont/ipaex00401/). Download `ipaexg00401.zip` from the link and extract `ipaexg.ttf`.
2.  **Placement**:
    *   Create a directory named `fonts` in the root of this repository if it doesn't already exist.
    *   Place the `ipaexg.ttf` file into this `fonts` directory (i.e., the path should be `fonts/ipaexg.ttf` from the repository root).
3.  **Usage**: The Streamlit application (`streamlit_app.py`) is configured to automatically look for this font at `fonts/ipaexg.ttf` when "Japanese" is selected as the language for traditional NLP tasks like Word Cloud.

If this font file is not found at the specified location, Japanese characters in word clouds may not render correctly, and you might see garbled text or empty squares.

## Documentation

Generated with Sphinx in the `docs/` folder. To rebuild:

```bash
cd docs
make html
```

Browse at `docs/_build/html/index.html`.

## Testing & CI

Run the test suite with:

```bash
pytest
```

Continuous integration is configured via GitHub Actions (`.github/workflows/ci.yml`).


## License

This project is licensed under the [MIT License](LICENSE).  
Acknowledges original [nlplot](https://github.com/takapy0210/nlplot) by takapy0210.
