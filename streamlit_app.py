import streamlit as st
import pandas as pd
import os
import re
from nlplot_llm import NLPlotLLM
from nlplot_llm.core import JANOME_AVAILABLE

DEFAULT_JP_FONT_PATH = "fonts/ipaexg.ttf"

st.set_page_config(page_title="NLPlotLLM Demo", layout="wide")


def get_nlplot_llm_instance():
    dummy_df = pd.DataFrame(
        {"text_llm": ["dummy text for nlplot_llm init"]}
    )
    use_llm_cache = st.session_state.get("use_llm_cache", True)
    return NLPlotLLM(
        dummy_df,
        target_col="text_llm",
        use_cache=use_llm_cache,
        font_path=None,
    )


def get_nlplot_instance_for_traditional_nlp(
    input_text_lines: list[str],
    language: str,
    target_column_name: str = "processed_text",
):
    font_to_use = None
    tokenized_lines = []

    if language == language_options[1] and JANOME_AVAILABLE:
        if os.path.exists(DEFAULT_JP_FONT_PATH):
            font_to_use = DEFAULT_JP_FONT_PATH
        else:
            st.warning(
                "Recommended Japanese font not found: "
                f"{DEFAULT_JP_FONT_PATH}. Word Cloud display for Japanese "
                "may not work correctly. Please follow the instructions in "
                "README.md to place `fonts/ipaexg.ttf`."
            )
        if JANOME_AVAILABLE:
            try:
                from janome.tokenizer import Tokenizer as JanomeTokenizer

                t_janome = JanomeTokenizer(wakati=True)
                for line in input_text_lines:
                    tokenized_lines.append(list(t_janome.tokenize(line)))
            except Exception as e:
                st.error(f"Failed to initialize Janome Tokenizer: {e}")
                for line in input_text_lines:
                    tokenized_lines.append(list(line))
        else:
            for line in input_text_lines:
                tokenized_lines.append(list(line))
    else:
        for line in input_text_lines:
            line_cleaned = re.sub(r"[^\w\s]", "", line).lower()
            tokenized_lines.append(line_cleaned.split())

    if not input_text_lines:
        df = pd.DataFrame(
            {target_column_name: pd.Series([], dtype="object")}
        )
    else:
        df = pd.DataFrame({target_column_name: tokenized_lines})

    return NLPlotLLM(
        df,
        target_col=target_column_name,
        font_path=font_to_use,
        use_cache=False,
    )


st.sidebar.header("LLM Configuration (LiteLLM)")

st.sidebar.checkbox(
    "Use LLM Response Cache",
    value=st.session_state.get("use_llm_cache", True),
    key="use_llm_cache",
    help=(
        "Enable caching of LLM responses to speed up repeated analyses "
        "with the same inputs and parameters. Cache is stored locally."
    ),
)
st.sidebar.markdown("---")

model_string = st.sidebar.text_input(
    "LiteLLM Model String",
    value="openai/gpt-3.5-turbo",
    help="e.g., `openai/gpt-3.5-turbo`, `ollama/llama2`, "
    "`azure/your-deployment-name`",
)

temperature = st.sidebar.slider(
    "Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1
)
max_tokens = st.sidebar.number_input(
    "Max Tokens (Output)",
    min_value=10,
    max_value=8192,
    value=256,
    step=10,
    help=(
        "Maximum number of tokens the model should generate. Adjust based "
        "on model and task."
    ),
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Optional Overrides (if not using environment variables):**"
)
api_key_input = st.sidebar.text_input(
    "API Key (e.g., OpenAI, Azure)",
    type="password",
    help="Sets `api_key` for LiteLLM.",
)
api_base_input = st.sidebar.text_input(
    "API Base URL (e.g., Ollama, custom OpenAI-compatible)",
    help="Sets `api_base` for LiteLLM.",
)

litellm_kwargs = {}
litellm_kwargs["temperature"] = temperature
if max_tokens > 0:
    litellm_kwargs["max_tokens"] = max_tokens
if api_key_input:
    litellm_kwargs["api_key"] = api_key_input
if api_base_input:
    litellm_kwargs["api_base"] = api_base_input

openai_key_needed_for_model = False
if "openai/" in model_string or (
    model_string.startswith("gpt-") and "/" not in model_string
):
    openai_key_needed_for_model = True
    if not api_key_input and not os.getenv("OPENAI_API_KEY"):
        st.sidebar.warning(
            "OpenAI model selected. Ensure OPENAI_API_KEY is set as an "
            "environment variable or provided above."
        )

st.title("NLPlotLLM Functions Demo")

st.header("Input Text")
input_text = st.text_area(
    "Enter text to analyze (one document per line for multiple inputs):",
    height=200,
    value=(
        "This is a wonderfully positive statement!\n"
        "This movie was terrible and a waste of time.\n"
        "Global warming is a serious issue that needs addressing.\n"
        "これは素晴らしい肯定的な声明です。\n"
        "この映画はひどく、時間の無駄でした。\n"
        "地球温暖化は深刻な問題であり、対処が必要です。"
    ),
)

st.header("Language Setting (for Traditional NLP)")
language_options = [
    "English (Space-separated)",
    "Japanese (Janome for tokenization)",
]
selected_language = st.radio(
    "Select language for tokenization in N-gram/Word Cloud:",
    language_options,
    index=0,
    key="language_select_key",
)
if selected_language == language_options[1] and not JANOME_AVAILABLE:
    st.warning(
        "Japanese is selected, but Janome is not available. Tokenization "
        "will fall back to space separation. Please install Janome for "
        "proper Japanese processing: `pip install janome`"
    )

st.header("Analysis Type")


def clear_state_on_analysis_type_change():
    st.session_state.show_jp_plot_options = False
    st.session_state.jp_features_df = None
    st.session_state.analysis_type_at_run = ""


analysis_options = [
    "Sentiment Analysis",
    "Text Categorization",
    "Text Summarization",
    "N-gram Analysis (Traditional)",
    "Word Cloud (Traditional)",
    "Japanese Text Analysis (Traditional)",
    "Word Count Distribution (Traditional)",
    "Co-occurrence Analysis (Traditional)",
    "TF-IDF Top Features (Traditional)",
    "KWIC (Keyword in Context) (Traditional)",
]
analysis_type = st.selectbox(
    "Select Analysis",
    analysis_options,
    key="analysis_type_selectbox",
    on_change=clear_state_on_analysis_type_change,
)

if "jp_features_df" not in st.session_state:
    st.session_state.jp_features_df = None
if "analysis_type_at_run" not in st.session_state:
    st.session_state.analysis_type_at_run = ""
if "show_jp_plot_options" not in st.session_state:
    st.session_state.show_jp_plot_options = False
if "jp_feature_selectbox_key" not in st.session_state:
    st.session_state.jp_feature_selectbox_key = 0

ngram_type_selected = 1
ngram_top_n_selected = 20
ngram_stopwords_str = ""

wc_max_words = 100
wc_stopwords_str = ""

wcd_bins = 20

co_stopwords_str = ""
co_min_edge_freq = 1
if "npt_graph_instance" not in st.session_state:
    st.session_state.npt_graph_instance = None
if "graph_built_success" not in st.session_state:
    st.session_state.graph_built_success = False
if "show_sunburst_chart" not in st.session_state:
    st.session_state.show_sunburst_chart = False
if "co_occurrence_network_fig_cache" not in st.session_state:
    st.session_state.co_occurrence_network_fig_cache = None

tfidf_n_features = 10
tfidf_custom_stopwords_str = ""
tfidf_ngram_min = 1
tfidf_ngram_max = 1
tfidf_max_df = 1.0
tfidf_min_df = 1
tfidf_return_type = "Overall Top Features"

kwic_keyword = ""
kwic_window_size = 5
kwic_ignore_case = True

prompt_sentiment = ""

categories_input_str = ""
multi_label_categories = False
prompt_categorize = ""

use_chunking_summarize = True
chunk_size_summarize = 1000
chunk_overlap_summarize = 100
chunk_prompt_template_summarize = ""
combine_prompt_template_summarize = ""


if analysis_type == "Sentiment Analysis":
    st.subheader("Sentiment Analysis Options")
    prompt_sentiment = st.text_area(
        "Sentiment Analysis Prompt (optional, use {text})",
        value=(
            "Analyze the sentiment of the following text and classify it as "
            "'positive', 'negative', or 'neutral'. Return only the single "
            "word classification for the sentiment. Text: {text}"
        ),
        height=150,
        help=(
            "Define the prompt for sentiment analysis. Available "
            "placeholder: {text}. If empty, library default is used."
        ),
    )
elif analysis_type == "Text Categorization":
    st.subheader("Categorization Options")
    categories_input_str = st.text_input(
        "Categories (comma-separated)",
        value="news,sports,technology,finance,health",
    )
    multi_label_categories = st.checkbox(
        "Allow multiple labels per text", value=False
    )
    prompt_categorize = st.text_area(
        "Categorization Prompt (optional, use {text} and {categories})",
        value="",
        height=150,
        help=(
            "Define the prompt for text categorization. Placeholders: "
            "{text}, {categories}. If empty, library default is used."
        ),
    )
elif analysis_type == "Text Summarization":
    st.subheader("Summarization Options")
    use_chunking_summarize = st.checkbox(
        "Use Chunking for Long Texts", value=True
    )
    if use_chunking_summarize:
        chunk_size_summarize = st.number_input(
            "Chunk Size",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
        )
        chunk_overlap_summarize = st.number_input(
            "Chunk Overlap",
            min_value=0,
            max_value=chunk_size_summarize - 50
            if chunk_size_summarize > 50
            else 0,
            value=100,
            step=50,
        )
        chunk_prompt_template_summarize = st.text_area(
            "Chunk Summarization Prompt (optional)",
            value="Summarize this text concisely: {text}",
            height=120,
            help=(
                "Prompt for summarizing individual text chunks. Use "
                "placeholder: {text}. If empty, library default is used."
            ),
        )
        combine_prompt_template_summarize = st.text_area(
            "Combine Summaries Prompt (optional)",
            value="Combine the following summaries into a coherent final "
            "summary: {text}",
            height=120,
            help=(
                "Prompt for combining summaries of multiple chunks. Use "
                "placeholder: {text} (which will contain concatenated "
                "chunk summaries). If empty, library default is used or "
                "summaries are just joined."
            ),
        )
    else:
        chunk_prompt_template_summarize = st.text_area(
            "Summarization Prompt (optional, use {text})",
            value="Please summarize the following text: {text}",
            height=120,
            help=(
                "Prompt for summarizing the text directly (when chunking is "
                "disabled). Use placeholder: {text}. If empty, library "
                "default is used."
            ),
        )
elif analysis_type == "N-gram Analysis (Traditional)":
    st.subheader("N-gram Analysis Options")
    ngram_type_selected = st.number_input(
        "N-gram (e.g., 1 for unigram, 2 for bigram)",
        min_value=1,
        max_value=5,
        value=ngram_type_selected,
        step=1,
    )
    ngram_top_n_selected = st.number_input(
        "Top N results to display",
        min_value=5,
        max_value=100,
        value=ngram_top_n_selected,
        step=5,
    )
    ngram_stopwords_str = st.text_input(
        "N-gram Stopwords (comma-separated)",
        value=ngram_stopwords_str
        if ngram_stopwords_str
        else "is,a,the,an,and,or,but",
    )
elif analysis_type == "Word Cloud (Traditional)":
    st.subheader("Word Cloud Options")
    wc_max_words = st.number_input(
        "Max Words in Cloud",
        min_value=10,
        max_value=500,
        value=wc_max_words,
        step=10,
    )
    wc_stopwords_str = st.text_input(
        "Word Cloud Stopwords (comma-separated)",
        value=wc_stopwords_str
        if wc_stopwords_str
        else "is,a,the,an,and,or,but",
    )
elif analysis_type == "Japanese Text Analysis (Traditional)":
    st.subheader("Japanese Text Analysis Options")
    if not JANOME_AVAILABLE:
        st.warning(
            "Janome (Japanese morphological analyzer) is not installed. "
            "This feature is unavailable. Please install it: "
            "`pip install janome`"
        )
    else:
        st.info(
            "This feature calculates various linguistic features for "
            "Japanese text. Ensure your input text is in Japanese."
        )
elif analysis_type == "Word Count Distribution (Traditional)":
    st.subheader("Word Count Distribution Options")
    wcd_bins = st.number_input(
        "Number of Bins for Histogram",
        min_value=5,
        max_value=100,
        value=wcd_bins,
        step=5,
    )
elif analysis_type == "Co-occurrence Analysis (Traditional)":
    st.subheader("Co-occurrence Analysis Options")
    co_stopwords_str = st.text_input(
        "Stopwords (comma-separated)",
        value=co_stopwords_str
        if co_stopwords_str
        else "is,a,the,an,and,or,but",
        key="co_stopwords",
    )
    co_min_edge_freq = st.number_input(
        "Minimum Edge Frequency",
        min_value=1,
        max_value=100,
        value=co_min_edge_freq,
        step=1,
        key="co_min_freq",
    )
elif analysis_type == "TF-IDF Top Features (Traditional)":
    st.subheader("TF-IDF Top Features Options")
    tfidf_n_features = st.number_input(
        "Top N Features to Display",
        min_value=1,
        max_value=100,
        value=tfidf_n_features,
        step=1,
        key="tfidf_n",
    )
    tfidf_custom_stopwords_str = st.text_area(
        "Custom Stopwords (comma-separated)",
        value=tfidf_custom_stopwords_str,
        key="tfidf_stopwords",
    )

    col1, col2 = st.columns(2)
    with col1:
        tfidf_ngram_min = st.number_input(
            "Min N-gram",
            min_value=1,
            max_value=5,
            value=tfidf_ngram_min,
            step=1,
            key="tfidf_ngram_min",
        )
    with col2:
        tfidf_ngram_max = st.number_input(
            "Max N-gram",
            min_value=tfidf_ngram_min,
            max_value=5,
            value=max(tfidf_ngram_min, tfidf_ngram_max),
            step=1,
            key="tfidf_ngram_max",
        )

    tfidf_max_df = st.slider(
        "Max Document Frequency (max_df)",
        0.5,
        1.0,
        tfidf_max_df,
        step=0.01,
        key="tfidf_max_df",
        help="Ignore terms that appear in more than this fraction of "
        "documents.",
    )
    tfidf_min_df = st.number_input(
        "Min Document Frequency (min_df)",
        min_value=1,
        value=tfidf_min_df,
        step=1,
        key="tfidf_min_df",
        help="Ignore terms that appear in less than this number of "
        "documents.",
    )

    tfidf_return_type_options = [
        "Overall Top Features",
        "Top Features per Document",
    ]
    tfidf_return_type = st.radio(
        "Result Type",
        options=tfidf_return_type_options,
        index=tfidf_return_type_options.index(tfidf_return_type),
        key="tfidf_return_type_radio",
    )
elif analysis_type == "KWIC (Keyword in Context) (Traditional)":
    st.subheader("KWIC Options")
    kwic_keyword = st.text_input(
        "Keyword to search for", value=kwic_keyword, key="kwic_keyword_input"
    )
    kwic_window_size = st.number_input(
        "Window Size (words on each side)",
        min_value=1,
        max_value=50,
        value=kwic_window_size,
        step=1,
        key="kwic_window",
    )
    kwic_ignore_case = st.checkbox(
        "Ignore Case", value=kwic_ignore_case, key="kwic_case"
    )

if st.button(f"Run {analysis_type}"):
    if not input_text.strip():
        st.error("Please enter some text to analyze.")
    elif (
        analysis_type
        in [
            "Sentiment Analysis",
            "Text Categorization",
            "Text Summarization",
        ]
        and not model_string.strip()
    ):
        st.error(
            "Please enter a LiteLLM Model String for LLM-based analysis."
        )
    else:
        lines = [
            line.strip() for line in input_text.split("\n") if line.strip()
        ]
        if not lines:
            st.error("No valid text lines found after stripping.")
        else:
            if analysis_type in [
                "Sentiment Analysis",
                "Text Categorization",
                "Text Summarization",
            ]:
                text_series = pd.Series(lines)
                npt_llm = get_nlplot_llm_instance()
                st.info(
                    f"Processing {len(text_series)} text document(s) with "
                    f"LLM: {model_string}..."
                )
                try:
                    with st.spinner("Analyzing with LLM..."):
                        if analysis_type == "Sentiment Analysis":
                            st.subheader("Sentiment Analysis Results")
                            analyze_kwargs = litellm_kwargs.copy()
                            if prompt_sentiment.strip():
                                analyze_kwargs[
                                    "prompt_template_str"
                                ] = prompt_sentiment
                            result_df = npt_llm.analyze_sentiment_llm(
                                text_series=text_series,
                                model=model_string,
                                **analyze_kwargs,
                            )
                            st.dataframe(result_df)

                        elif analysis_type == "Text Categorization":
                            st.subheader("Text Categorization Results")
                            if not categories_input_str.strip():
                                st.error(
                                    "Please enter categories for text "
                                    "categorization."
                                )
                            else:
                                categories_list = [
                                    cat.strip()
                                    for cat in categories_input_str.split(
                                        ","
                                    )
                                    if cat.strip()
                                ]
                                if not categories_list:
                                    st.error("No valid categories provided.")
                                else:
                                    categorize_kwargs = litellm_kwargs.copy()
                                    if prompt_categorize.strip():
                                        categorize_kwargs[
                                            "prompt_template_str"
                                        ] = (prompt_categorize)
                                    result_df = npt_llm.categorize_text_llm(
                                        text_series=text_series,
                                        categories=categories_list,
                                        model=model_string,
                                        multi_label=multi_label_categories,
                                        **categorize_kwargs,
                                    )
                                    st.dataframe(result_df)

                        elif analysis_type == "Text Summarization":
                            st.subheader("Text Summarization Results")
                            summarize_final_kwargs = litellm_kwargs.copy()
                            summarize_final_kwargs[
                                "use_chunking"
                            ] = use_chunking_summarize
                            if use_chunking_summarize:
                                summarize_final_kwargs[
                                    "chunk_size"
                                ] = chunk_size_summarize
                                summarize_final_kwargs[
                                    "chunk_overlap"
                                ] = chunk_overlap_summarize
                                if chunk_prompt_template_summarize.strip():
                                    summarize_final_kwargs[
                                        "chunk_prompt_template_str"
                                    ] = chunk_prompt_template_summarize
                                if (
                                    combine_prompt_template_summarize.strip()
                                ):
                                    summarize_final_kwargs[
                                        "combine_prompt_template_str"
                                    ] = combine_prompt_template_summarize
                            else:
                                if chunk_prompt_template_summarize.strip():
                                    summarize_final_kwargs[
                                        "prompt_template_str"
                                    ] = chunk_prompt_template_summarize
                            result_df = npt_llm.summarize_text_llm(
                                text_series=text_series,
                                model=model_string,
                                **summarize_final_kwargs,
                            )
                            st.dataframe(result_df)
                except ImportError as ie:
                    st.error(
                        f"ImportError: {ie}. Make sure Langchain and "
                        "necessary LLM provider libraries are installed."
                    )
                except ValueError as ve:
                    st.error(f"ValueError: {ve}")
                except Exception as e:
                    st.error(
                        "An unexpected error occurred during LLM analysis: "
                        f"{e}"
                    )

            elif analysis_type == "N-gram Analysis (Traditional)":
                st.info(
                    "Processing {len(lines)} text document(s) for "
                    "N-gram Analysis..."
                )
                npt_traditional = get_nlplot_instance_for_traditional_nlp(
                    lines,
                    selected_language,
                    target_column_name="input_tokens",
                )

                stopwords_list = [
                    sw.strip()
                    for sw in ngram_stopwords_str.split(",")
                    if sw.strip()
                ]

                try:
                    with st.spinner("Generating N-gram Bar Chart..."):
                        st.subheader(
                            f"{ngram_type_selected}-gram Bar Chart "
                            f"(Top {ngram_top_n_selected})"
                        )
                        fig_bar = npt_traditional.bar_ngram(
                            title=f"{ngram_type_selected}-gram Frequency",
                            ngram=ngram_type_selected,
                            top_n=ngram_top_n_selected,
                            stopwords=stopwords_list,
                            
                        )
                        if fig_bar and fig_bar.data:
                            st.plotly_chart(
                                fig_bar, use_container_width=True
                            )
                        else:
                            st.warning(
                                "No data to display for N-gram Bar Chart. "
                                "Adjust parameters or input text."
                            )

                    with st.spinner("Generating N-gram Treemap..."):
                        st.subheader(
                            f"{ngram_type_selected}-gram Treemap "
                            f"(Top {ngram_top_n_selected})"
                        )
                        fig_treemap = npt_traditional.treemap(
                            title=f"{ngram_type_selected}-gram Treemap",
                            ngram=ngram_type_selected,
                            top_n=ngram_top_n_selected,
                            stopwords=stopwords_list,
                            
                        )
                        if fig_treemap and fig_treemap.data:
                            st.plotly_chart(
                                fig_treemap, use_container_width=True
                            )
                        else:
                            st.warning(
                                "No data to display for N-gram Treemap. "
                                "Adjust parameters or input text."
                            )

                except Exception as e:
                    st.error(
                        f"An error occurred during N-gram Analysis: {e}"
                    )

            elif analysis_type == "Word Cloud (Traditional)":
                st.info(
                    "Processing {len(lines)} text document(s) for Word "
                    "Cloud..."
                )
                npt_traditional = get_nlplot_instance_for_traditional_nlp(
                    lines,
                    selected_language,
                    target_column_name="input_tokens_wc",
                )

                wc_stopwords_list = [
                    sw.strip()
                    for sw in wc_stopwords_str.split(",")
                    if sw.strip()
                ]

                try:
                    with st.spinner("Generating Word Cloud..."):
                        st.subheader(
                            f"Word Cloud (Max Words: {wc_max_words})"
                        )
                        pil_image = npt_traditional.wordcloud(
                            max_words=wc_max_words,
                            stopwords=wc_stopwords_list,
                        )
                        if pil_image:
                            st.image(pil_image, use_column_width=True)
                        else:
                            st.warning(
                                "Could not generate Word Cloud. Input text "
                                "might be empty or all words filtered out."
                            )

                except Exception as e:
                    st.error(
                        "An error occurred during Word Cloud generation: "
                        f"{e}"
                    )

            elif analysis_type == "Japanese Text Analysis (Traditional)":
                if not JANOME_AVAILABLE:
                    st.error(
                        "Janome is not installed, cannot perform Japanese "
                        "Text Analysis."
                    )
                else:
                    st.info(
                        "Processing {len(lines)} text document(s) for "
                        "Japanese Text Analysis..."
                    )
                    npt_jp_analyzer = get_nlplot_llm_instance()

                    text_series_jp = pd.Series(lines)

                    try:
                        with st.spinner(
                            "Calculating Japanese text features..."
                        ):
                            st.session_state.show_jp_plot_options = False
                            st.session_state.jp_features_df = (
                                npt_jp_analyzer.get_japanese_text_features(
                                    text_series_jp
                                )
                            )
                            st.session_state.analysis_type_at_run = (
                                analysis_type
                            )

                        if (
                            st.session_state.jp_features_df is not None
                            and not st.session_state.jp_features_df.empty
                        ):
                            st.subheader("Japanese Text Features")
                            st.dataframe(st.session_state.jp_features_df)
                            st.session_state.show_jp_plot_options = True
                        else:
                            st.warning(
                                "No features were calculated. Input text "
                                "might be unsuitable or empty."
                            )
                            st.session_state.show_jp_plot_options = False

                    except Exception as e:
                        st.error(
                            "An error occurred during Japanese Text "
                            f"Analysis feature calculation: {e}"
                        )
                        st.session_state.jp_features_df = None
                        st.session_state.show_jp_plot_options = False

            elif analysis_type == "Word Count Distribution (Traditional)":
                st.info(
                    "Processing {len(lines)} text document(s) for Word "
                    "Count Distribution..."
                )
                npt_traditional = get_nlplot_instance_for_traditional_nlp(
                    lines,
                    selected_language,
                    target_column_name="input_tokens_wcd",
                )
                try:
                    with st.spinner(
                        "Generating Word Count Distribution Plot..."
                    ):
                        st.subheader("Word Count Distribution")
                        fig_wcd = npt_traditional.word_distribution(
                            title="Distribution of Word Counts per Document",
                            bins=wcd_bins,
                        )
                        if fig_wcd and fig_wcd.data:
                            st.plotly_chart(
                                fig_wcd, use_container_width=True
                            )
                        else:
                            st.warning(
                                "No data to display for Word Count "
                                "Distribution. Input text might be empty."
                            )
                except Exception as e:
                    st.error(
                        "An error occurred during Word Count Distribution "
                        f"generation: {e}"
                    )

            elif analysis_type == "Co-occurrence Analysis (Traditional)":
                st.info(
                    "Processing {len(lines)} text document(s) for "
                    "Co-occurrence Analysis..."
                )
                npt_co = get_nlplot_instance_for_traditional_nlp(
                    lines,
                    selected_language,
                    target_column_name="input_tokens_co",
                )

                co_stopwords_list = [
                    sw.strip()
                    for sw in co_stopwords_str.split(",")
                    if sw.strip()
                ]

                st.session_state.npt_graph_instance = None
                st.session_state.graph_built_success = False

                try:
                    with st.spinner(
                        "Building graph for Co-occurrence Network..."
                    ):
                        npt_co.build_graph(
                            stopwords=co_stopwords_list,
                            min_edge_frequency=co_min_edge_freq,
                        )
                        st.session_state.npt_graph_instance = npt_co
                        st.session_state.graph_built_success = True
                        st.success(
                            "Graph for co-occurrence analysis built "
                            "successfully."
                        )

                    if st.session_state.graph_built_success:
                        with st.spinner(
                            "Generating Co-occurrence Network plot..."
                        ):
                            fig_co_network = st.session_state.npt_graph_instance.co_network(
                                title="Co-occurrence Network"
                            )
                            st.session_state.co_occurrence_network_fig_cache = (
                                fig_co_network
                            )
                        st.session_state.analysis_type_at_run = (
                            analysis_type
                        )
                        st.session_state.show_sunburst_chart = False
                    else:
                        st.session_state.co_occurrence_network_fig_cache = (
                            None
                        )
                        st.session_state.show_sunburst_chart = False
                        if (
                            hasattr(npt_co, "node_df")
                            and npt_co.node_df.empty
                        ):
                            st.warning(
                                "No nodes found for co-occurrence network. "
                                "Try adjusting stopwords or minimum edge "
                                "frequency."
                            )

                except Exception as e:
                    st.error(
                        "An error occurred during Co-occurrence Analysis: "
                        f"{e}"
                    )
                    st.session_state.npt_graph_instance = None
                    st.session_state.graph_built_success = False
                    st.session_state.co_occurrence_network_fig_cache = None
                    st.session_state.show_sunburst_chart = False

            elif analysis_type == "TF-IDF Top Features (Traditional)":
                st.info(
                    "Processing {len(lines)} text document(s) for TF-IDF "
                    "Analysis..."
                )
                text_series_for_tfidf = pd.Series(lines)

                npt_tfidf = get_nlplot_llm_instance()

                custom_stopwords_list = [
                    sw.strip()
                    for sw in tfidf_custom_stopwords_str.split(",")
                    if sw.strip()
                ]

                actual_tfidf_ngram_max = max(
                    tfidf_ngram_min, tfidf_ngram_max
                )
                if actual_tfidf_ngram_max < tfidf_ngram_min:
                    st.warning(
                        "Max N-gram was less than Min N-gram, adjusting "
                        "Max N-gram to be equal to Min N-gram."
                    )
                    actual_tfidf_ngram_max = tfidf_ngram_min

                return_type_param = (
                    "overall"
                    if tfidf_return_type == "Overall Top Features"
                    else "per_document"
                )

                try:
                    with st.spinner(
                        "Calculating TF-IDF and extracting top features..."
                    ):
                        df_tfidf_results = (
                            npt_tfidf.get_tfidf_top_features(
                                text_series=text_series_for_tfidf,
                                language=selected_language.split(" ")[
                                    0
                                ].lower(),
                                n_features=tfidf_n_features,
                                custom_stopwords=custom_stopwords_list,
                                use_janome_tokenizer_for_japanese=True,
                                tfidf_ngram_range=(
                                    tfidf_ngram_min,
                                    actual_tfidf_ngram_max,
                                ),
                                tfidf_max_df=tfidf_max_df,
                                tfidf_min_df=tfidf_min_df,
                                return_type=return_type_param,
                            )
                        )

                        st.subheader("TF-IDF Top Features Results")
                        if (
                            df_tfidf_results is not None
                            and not df_tfidf_results.empty
                        ):
                            st.dataframe(df_tfidf_results)

                            if (
                                return_type_param == "overall"
                                and "word" in df_tfidf_results.columns
                                and "tfidf_score"
                                in df_tfidf_results.columns
                            ):
                                try:
                                    chart_data = (
                                        df_tfidf_results.set_index("word")[
                                            "tfidf_score"
                                        ].sort_values(ascending=False)
                                    )
                                    if not chart_data.empty:
                                        st.bar_chart(chart_data)
                                except Exception as e_chart:
                                    st.warning(
                                        "Could not generate bar chart for "
                                        f"TF-IDF results: {e_chart}"
                                    )

                        else:
                            st.warning(
                                "No TF-IDF features found. Try adjusting "
                                "parameters (e.g., stopwords, min_df) or "
                                "input text."
                            )

                except ImportError:
                    st.error(
                        "scikit-learn is required for TF-IDF analysis. "
                        "Please install it: `pip install scikit-learn`"
                    )
                except Exception as e:
                    st.error(
                        f"An error occurred during TF-IDF Analysis: {e}"
                    )

            elif analysis_type == "KWIC (Keyword in Context) (Traditional)":
                if not kwic_keyword.strip():
                    st.error("Please enter a keyword for KWIC analysis.")
                else:
                    st.info(
                        f"Searching for keyword '{kwic_keyword}' in "
                        f"{len(lines)} document(s)..."
                    )
                    text_series_for_kwic = pd.Series(lines)
                    npt_kwic = get_nlplot_llm_instance()

                    try:
                        with st.spinner(
                            "Performing KWIC analysis for "
                            f"'{kwic_keyword}'..."
                        ):
                            kwic_results_list = npt_kwic.get_kwic_results(
                                text_series=text_series_for_kwic,
                                keyword=kwic_keyword,
                                language=selected_language.split(" ")[
                                    0
                                ].lower(),
                                window_size=kwic_window_size,
                                use_janome_tokenizer_for_japanese=True,
                                ignore_case=kwic_ignore_case,
                            )

                            st.subheader(
                                f"KWIC Results for '{kwic_keyword}'"
                            )
                            if kwic_results_list:
                                html_lines = []
                                for item in kwic_results_list:
                                    left = item["left_context"]
                                    keyword_match = item["keyword_match"]
                                    right = item["right_context"]
                                    html_lines.append(
                                        "<tr><td style='text-align:right; "
                                        "padding-right:5px;'>"
                                        f"{left}</td><td style='"
                                        "font-weight:bold; padding:0 5px;'>"
                                        f"{keyword_match}</td><td "
                                        "style='text-align:left; "
                                        "padding-left:5px;'>"
                                        f"{right}</td><td style='"
                                        "font-size:smaller; "
                                        "padding-left:15px; "
                                        "color:gray;'>"
                                        f"DocID: {item['document_id']}"
                                        "</td></tr>"
                                    )
                                table_html = (
                                    "<table>"
                                    + "".join(html_lines)
                                    + "</table>"
                                )
                                st.markdown(
                                    table_html, unsafe_allow_html=True
                                )
                            else:
                                st.info(
                                    f"Keyword '{kwic_keyword}' not found "
                                    "in the provided text with the "
                                    "current settings."
                                )
                    except Exception as e:
                        st.error(
                            "An error occurred during KWIC Analysis: " f"{e}"
                        )


if (
    analysis_type == "Co-occurrence Analysis (Traditional)"
    and st.session_state.get("graph_built_success", False)
    and st.session_state.co_occurrence_network_fig_cache is not None
):
    st.subheader("Co-occurrence Network")
    st.plotly_chart(
        st.session_state.co_occurrence_network_fig_cache,
        use_container_width=True,
    )

    if st.button("Show/Hide Sunburst Chart", key="toggle_sunburst_co"):
        st.session_state.show_sunburst_chart = not st.session_state.get(
            "show_sunburst_chart", False
        )

if (
    analysis_type == "Co-occurrence Analysis (Traditional)"
    and st.session_state.get("graph_built_success", False)
    and st.session_state.get("show_sunburst_chart", False)
    and st.session_state.npt_graph_instance is not None
):
    with st.spinner("Generating Sunburst Chart..."):
        st.subheader("Sunburst Chart (from Co-occurrence Data)")
        fig_sunburst = st.session_state.npt_graph_instance.sunburst(
            title="Co-occurrence Sunburst"
        )
        if fig_sunburst and fig_sunburst.data:
            st.plotly_chart(fig_sunburst, use_container_width=True)
        else:
            st.warning(
                "Could not generate Sunburst Chart. Graph data might be "
                "unsuitable."
            )

if (
    analysis_type == "Japanese Text Analysis (Traditional)"
    and st.session_state.get("show_jp_plot_options", False)
    and st.session_state.jp_features_df is not None
    and not st.session_state.jp_features_df.empty
):
    st.subheader("Plot Japanese Text Feature Distribution")
    numeric_cols = st.session_state.jp_features_df.select_dtypes(
        include="number"
    ).columns.tolist()

    default_idx = 0
    if numeric_cols:
        if "total_tokens" in numeric_cols:
            default_idx = numeric_cols.index("total_tokens")
    else:
        st.info("No numeric features available to plot.")

    if numeric_cols:
        selected_feature_for_plot = st.selectbox(
            "Select feature to plot:",
            options=numeric_cols,
            index=default_idx,
            key="jp_feature_selector_stable_key",
        )

        if selected_feature_for_plot:
            npt_jp_plotter = get_nlplot_llm_instance()
            try:
                with st.spinner(
                    f"Generating plot for {selected_feature_for_plot}..."
                ):
                    fig_jp_plot = (
                        npt_jp_plotter.plot_japanese_text_features(
                            st.session_state.jp_features_df,
                            target_feature=selected_feature_for_plot,
                            title=(
                                "Distribution of "
                                f"{selected_feature_for_plot}"
                            ),
                        )
                    )
                    if fig_jp_plot:
                        st.plotly_chart(
                            fig_jp_plot, use_container_width=True
                        )
                    else:
                        st.warning(
                            f"Could not generate plot for "
                            f"{selected_feature_for_plot}."
                        )
            except Exception as e:
                st.error(
                    "An error occurred during plot generation for "
                    f"'{selected_feature_for_plot}': {e}"
                )

if not st.session_state.get("run_button_clicked", False):
    st.caption(f"Click the 'Run {analysis_type}' button to start.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Refer to [LiteLLM Documentation]"
    "(https://docs.litellm.ai/docs/providers) for provider-specific API "
    "keys (e.g., `OPENAI_API_KEY`, `AZURE_API_KEY`, `COHERE_API_KEY`, "
    "`ANTHROPIC_API_KEY`) and other parameters. For local models like "
    "Ollama, ensure the server is running."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by NLPlotLLM and Streamlit.")
st.sidebar.markdown(
    "Ensure your LLM is configured and accessible as per LiteLLM "
    "requirements."
)

print("Streamlit App Script Created/Updated.")
