import streamlit as st
import pandas as pd
import os # Added for os.getenv
from nlplot_llm import NLPlotLLM # Updated import
from nlplot_llm.core import JANOME_AVAILABLE # Import JANOME_AVAILABLE flag

# --- App Configuration ---
st.set_page_config(page_title="NLPlotLLM Demo", layout="wide") # Updated title

# --- Helper Functions (placeholder for now) ---
def get_nlplot_llm_instance(): # Updated function name
    """Creates and returns an NLPlotLLM instance.""" # Updated class name in docstring
    # For the demo, we might not need a DataFrame for NLPlotLLM initialization
    # if we are only using LLM methods that take Series directly.
    # However, NLPlotLLM requires a df and target_col for its constructor.
    # We can use a dummy one for LLM tasks that take Series directly.
    dummy_df = pd.DataFrame({'text_llm': ["dummy text for nlplot_llm init"]}) # Changed column name for clarity
    # Get cache setting from sidebar
    use_llm_cache = st.session_state.get("use_llm_cache", True) # Default to True if not set
    # Note: Font path is not critical for LLM-only tasks but set for completeness.
    return NLPlotLLM(dummy_df, target_col='text_llm', use_cache=use_llm_cache, font_path=None)

def get_nlplot_instance_for_traditional_nlp(input_text_lines: list[str], target_column_name: str = "processed_text"):
    """
    Creates an NLPlotLLM instance suitable for traditional NLP tasks.
    The input text lines are tokenized (by simple space splitting) and put into a DataFrame.
    """
    if not input_text_lines:
        st.warning("No input text provided for traditional NLP analysis. Using empty DataFrame.")
        # NLPlotLLM expects a DataFrame with a specific column, even if empty.
        df = pd.DataFrame({target_column_name: pd.Series([], dtype='object')})
    else:
        # Simple tokenization (splitting by space) for traditional NLP functions
        # NLPlotLLM's traditional methods expect a list of tokens.
        tokenized_lines = [line.split() for line in input_text_lines]
        df = pd.DataFrame({target_column_name: tokenized_lines})

    # For traditional NLP, caching at the NLPlotLLM level is not currently implemented for its plotting methods.
    # Font path might be relevant for word cloud.
    # TODO: Consider making font_path configurable via Streamlit UI if WordCloud is added.
    font_path_ui = None # Placeholder for potential future UI input for font
    return NLPlotLLM(df, target_col=target_column_name, font_path=font_path_ui, use_cache=False)

# --- Sidebar for LLM Configuration ---
st.sidebar.header("LLM Configuration (LiteLLM)")

# Cache setting
st.sidebar.checkbox(
    "Use LLM Response Cache",
    value=st.session_state.get("use_llm_cache", True),
    key="use_llm_cache",
    help="Enable caching of LLM responses to speed up repeated analyses with the same inputs and parameters. Cache is stored locally."
)
st.sidebar.markdown("---") # Visual separator

# Model string input
model_string = st.sidebar.text_input(
    "LiteLLM Model String",
    value="openai/gpt-3.5-turbo",
    help="e.g., `openai/gpt-3.5-turbo`, `ollama/llama2`, `azure/your-deployment-name`"
)

# Common LiteLLM parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
max_tokens = st.sidebar.number_input("Max Tokens (Output)", min_value=10, max_value=8192, value=256, step=10, help="Maximum number of tokens the model should generate. Adjust based on model and task.")

# Optional API Key / Base URL (LiteLLM often reads from env vars)
st.sidebar.markdown("---")
st.sidebar.markdown("**Optional Overrides (if not using environment variables):**")
api_key_input = st.sidebar.text_input("API Key (e.g., OpenAI, Azure)", type="password", help="Sets `api_key` for LiteLLM.")
api_base_input = st.sidebar.text_input("API Base URL (e.g., Ollama, custom OpenAI-compatible)", help="Sets `api_base` for LiteLLM.")

# Prepare litellm_kwargs
litellm_kwargs = {}
litellm_kwargs["temperature"] = temperature
if max_tokens > 0: # Ensure max_tokens is positive, though number_input should enforce min_value
    litellm_kwargs["max_tokens"] = max_tokens
if api_key_input:
    litellm_kwargs["api_key"] = api_key_input
if api_base_input:
    litellm_kwargs["api_base"] = api_base_input

# Check for OpenAI key specifically if OpenAI model is chosen, for user guidance
# This is a bit of a heuristic as LiteLLM handles it, but good for UX.
openai_key_needed_for_model = False
if "openai/" in model_string or (model_string.startswith("gpt-") and "/" not in model_string): # Simple check
    openai_key_needed_for_model = True
    if not api_key_input and not os.getenv("OPENAI_API_KEY"):
        st.sidebar.warning("OpenAI model selected. Ensure OPENAI_API_KEY is set as an environment variable or provided above.")



# --- Main App Area ---
st.title("NLPlotLLM Functions Demo") # Updated title

# Input Text Area
st.header("Input Text")
input_text = st.text_area("Enter text to analyze (one document per line for multiple inputs):", height=200,
                          value="This is a wonderfully positive statement!\nThis movie was terrible and a waste of time.\nGlobal warming is a serious issue that needs addressing.")

# Analysis Type Selection
st.header("Analysis Type")
analysis_options = [
    "Sentiment Analysis",
    "Text Categorization",
    "Text Summarization",
    "N-gram Analysis (Traditional)",
    "Word Cloud (Traditional)",
    "Japanese Text Analysis (Traditional)",
    # Add other traditional NLP functions here later e.g. "Co-occurrence Network"
]
analysis_type = st.selectbox("Select Analysis", analysis_options)

# --- Analysis Specific Options ---
# These will store UI selections for various parameters

# Japanese Text Analysis Options
# jp_feature_to_plot = None # This will be handled by the selectbox directly or session_state if needed for persistence across runs unrelated to its own change
# Session state to store generated features dataframe to avoid recomputing
if 'jp_features_df' not in st.session_state:
    st.session_state.jp_features_df = None
if 'analysis_type_at_run' not in st.session_state: # To store analysis type when button is clicked
    st.session_state.analysis_type_at_run = ""
if 'show_jp_plot_options' not in st.session_state: # Flag to control display of plot options
    st.session_state.show_jp_plot_options = False
if 'jp_feature_selectbox_key' not in st.session_state: # Key for selectbox, might not be strictly needed if value is read directly
    st.session_state.jp_feature_selectbox_key = 0 # Or some other initial value if you want to control index by state

# N-gram Analysis Options
ngram_type_selected = 1
ngram_top_n_selected = 20
ngram_stopwords_str = "" # Comma-separated string for stopwords

# Word Cloud Options
wc_max_words = 100
wc_stopwords_str = "" # Comma-separated string for stopwords for wordcloud
# wc_font_path_str = "" # Optional: UI for font path

# Sentiment Analysis Options
prompt_sentiment = ""

# Text Categorization Options
categories_input_str = ""
multi_label_categories = False
prompt_categorize = ""

# Text Summarization Options
use_chunking_summarize = True
chunk_size_summarize = 1000
chunk_overlap_summarize = 100
chunk_prompt_template_summarize = "" # This will be used for direct prompt if not chunking
combine_prompt_template_summarize = ""


if analysis_type == "Sentiment Analysis":
    st.subheader("Sentiment Analysis Options")
    prompt_sentiment = st.text_area(
        "Sentiment Analysis Prompt (optional, use {text})",
        value="Analyze the sentiment of the following text and classify it as 'positive', 'negative', or 'neutral'. Return only the single word classification for the sentiment. Text: {text}",
        height=150,
        help="Define the prompt for sentiment analysis. Available placeholder: {text}. If empty, library default is used."
    )
elif analysis_type == "Text Categorization":
    st.subheader("Categorization Options")
    categories_input_str = st.text_input("Categories (comma-separated)", value="news,sports,technology,finance,health")
    multi_label_categories = st.checkbox("Allow multiple labels per text", value=False)
    prompt_categorize = st.text_area(
        "Categorization Prompt (optional, use {text} and {categories})",
        value="", # Default will be set in core.py or based on multi_label
        height=150,
        help="Define the prompt for text categorization. Placeholders: {text}, {categories}. If empty, library default is used."
    )
elif analysis_type == "Text Summarization":
    st.subheader("Summarization Options")
    use_chunking_summarize = st.checkbox("Use Chunking for Long Texts", value=True)
    if use_chunking_summarize:
        chunk_size_summarize = st.number_input("Chunk Size", min_value=100, max_value=10000, value=1000, step=100)
        chunk_overlap_summarize = st.number_input("Chunk Overlap", min_value=0, max_value=chunk_size_summarize-50 if chunk_size_summarize > 50 else 0, value=100, step=50)
        chunk_prompt_template_summarize = st.text_area(
            "Chunk Summarization Prompt (optional)",
            value="Summarize this text concisely: {text}",
            height=120,
            help="Prompt for summarizing individual text chunks. Use placeholder: {text}. If empty, library default is used."
        )
        combine_prompt_template_summarize = st.text_area(
            "Combine Summaries Prompt (optional)",
            value="Combine the following summaries into a coherent final summary: {text}",
            height=120,
            help="Prompt for combining summaries of multiple chunks. Use placeholder: {text} (which will contain concatenated chunk summaries). If empty, library default is used or summaries are just joined."
        )
    else: # Direct summarization prompt (if not chunking)
         chunk_prompt_template_summarize = st.text_area( # Re-use this variable for the direct prompt
            "Summarization Prompt (optional, use {text})",
            value="Please summarize the following text: {text}", # Default direct prompt
            height=120,
            help="Prompt for summarizing the text directly (when chunking is disabled). Use placeholder: {text}. If empty, library default is used."
        )
elif analysis_type == "N-gram Analysis (Traditional)":
    st.subheader("N-gram Analysis Options")
    ngram_type_selected = st.number_input("N-gram (e.g., 1 for unigram, 2 for bigram)", min_value=1, max_value=5, value=ngram_type_selected, step=1)
    ngram_top_n_selected = st.number_input("Top N results to display", min_value=5, max_value=100, value=ngram_top_n_selected, step=5)
    ngram_stopwords_str = st.text_input("N-gram Stopwords (comma-separated)", value=ngram_stopwords_str if ngram_stopwords_str else "is,a,the,an,and,or,but")
elif analysis_type == "Word Cloud (Traditional)":
    st.subheader("Word Cloud Options")
    wc_max_words = st.number_input("Max Words in Cloud", min_value=10, max_value=500, value=wc_max_words, step=10)
    wc_stopwords_str = st.text_input("Word Cloud Stopwords (comma-separated)", value=wc_stopwords_str if wc_stopwords_str else "is,a,the,an,and,or,but")
    # Potentially add font path input:
    # wc_font_path_str = st.text_input("Font Path for Word Cloud (optional, .ttf file)", value=wc_font_path_str)
elif analysis_type == "Japanese Text Analysis (Traditional)":
    st.subheader("Japanese Text Analysis Options")
    if not JANOME_AVAILABLE:
        st.warning("Janome (Japanese morphological analyzer) is not installed. This feature is unavailable. Please install it: `pip install janome`")
    else:
        st.info("This feature calculates various linguistic features for Japanese text. Ensure your input text is in Japanese.")
        # Options for this analysis will be primarily for plotting after feature generation.
        # Feature selection for plotting will appear after features are generated.
        pass


# Execute Button
if st.button(f"Run {analysis_type}"):
    if not input_text.strip():
        st.error("Please enter some text to analyze.")
    # LLM model string is only required for LLM-based analyses
    elif analysis_type in ["Sentiment Analysis", "Text Categorization", "Text Summarization"] and not model_string.strip():
        st.error("Please enter a LiteLLM Model String for LLM-based analysis.")
    else:
        lines = [line.strip() for line in input_text.split('\n') if line.strip()]
        if not lines:
            st.error("No valid text lines found after stripping.")
        else:
            # LLM specific processing
            if analysis_type in ["Sentiment Analysis", "Text Categorization", "Text Summarization"]:
                text_series = pd.Series(lines)
                npt_llm = get_nlplot_llm_instance() # Instance for LLM tasks
                st.info(f"Processing {len(text_series)} text document(s) with LLM: {model_string}...")
                try:
                    with st.spinner("Analyzing with LLM..."):
                        if analysis_type == "Sentiment Analysis":
                            st.subheader("Sentiment Analysis Results")
                            analyze_kwargs = litellm_kwargs.copy()
                            if prompt_sentiment.strip():
                                analyze_kwargs["prompt_template_str"] = prompt_sentiment
                            result_df = npt_llm.analyze_sentiment_llm(
                                text_series=text_series,
                                model=model_string,
                                **analyze_kwargs
                            )
                            st.dataframe(result_df)

                        elif analysis_type == "Text Categorization":
                            st.subheader("Text Categorization Results")
                            if not categories_input_str.strip():
                                st.error("Please enter categories for text categorization.")
                            else:
                                categories_list = [cat.strip() for cat in categories_input_str.split(',') if cat.strip()]
                                if not categories_list:
                                    st.error("No valid categories provided.")
                                else:
                                    categorize_kwargs = litellm_kwargs.copy()
                                    if prompt_categorize.strip():
                                        categorize_kwargs["prompt_template_str"] = prompt_categorize
                                    result_df = npt_llm.categorize_text_llm(
                                        text_series=text_series,
                                        categories=categories_list,
                                        model=model_string,
                                        multi_label=multi_label_categories,
                                        **categorize_kwargs
                                    )
                                    st.dataframe(result_df)

                        elif analysis_type == "Text Summarization":
                            st.subheader("Text Summarization Results")
                            summarize_final_kwargs = litellm_kwargs.copy()
                            summarize_final_kwargs["use_chunking"] = use_chunking_summarize
                            if use_chunking_summarize:
                                summarize_final_kwargs["chunk_size"] = chunk_size_summarize
                                summarize_final_kwargs["chunk_overlap"] = chunk_overlap_summarize
                                if chunk_prompt_template_summarize.strip():
                                    summarize_final_kwargs["chunk_prompt_template_str"] = chunk_prompt_template_summarize
                                if combine_prompt_template_summarize.strip():
                                    summarize_final_kwargs["combine_prompt_template_str"] = combine_prompt_template_summarize
                            else:
                                if chunk_prompt_template_summarize.strip():
                                    summarize_final_kwargs["prompt_template_str"] = chunk_prompt_template_summarize
                            result_df = npt_llm.summarize_text_llm(
                                text_series=text_series,
                                model=model_string,
                                **summarize_final_kwargs
                            )
                            st.dataframe(result_df)
                except ImportError as ie:
                    st.error(f"ImportError: {ie}. Make sure Langchain and necessary LLM provider libraries are installed.")
                except ValueError as ve:
                    st.error(f"ValueError: {ve}")
                except Exception as e:
                    st.error(f"An unexpected error occurred during LLM analysis: {e}")

            # Traditional NLP processing
            elif analysis_type == "N-gram Analysis (Traditional)":
                st.info(f"Processing {len(lines)} text document(s) for N-gram Analysis...")
                # For N-gram, we need an NLPlotLLM instance with the actual text data
                # The get_nlplot_instance_for_traditional_nlp helper function prepares this.
                # The target column name used in the helper is "processed_text" by default.
                npt_traditional = get_nlplot_instance_for_traditional_nlp(lines, target_column_name="input_tokens")

                # Prepare stopwords list
                stopwords_list = [sw.strip() for sw in ngram_stopwords_str.split(',') if sw.strip()]

                try:
                    with st.spinner("Generating N-gram Bar Chart..."):
                        st.subheader(f"{ngram_type_selected}-gram Bar Chart (Top {ngram_top_n_selected})")
                        fig_bar = npt_traditional.bar_ngram(
                            title=f"{ngram_type_selected}-gram Frequency",
                            ngram=ngram_type_selected,
                            top_n=ngram_top_n_selected,
                            stopwords=stopwords_list,
                            verbose=False # Reduce console output in Streamlit app
                        )
                        if fig_bar and fig_bar.data:
                            st.plotly_chart(fig_bar, use_container_width=True)
                        else:
                            st.warning("No data to display for N-gram Bar Chart. Adjust parameters or input text.")

                    with st.spinner("Generating N-gram Treemap..."):
                        st.subheader(f"{ngram_type_selected}-gram Treemap (Top {ngram_top_n_selected})")
                        fig_treemap = npt_traditional.treemap(
                            title=f"{ngram_type_selected}-gram Treemap",
                            ngram=ngram_type_selected,
                            top_n=ngram_top_n_selected,
                            stopwords=stopwords_list,
                            verbose=False # Reduce console output
                        )
                        if fig_treemap and fig_treemap.data:
                            st.plotly_chart(fig_treemap, use_container_width=True)
                        else:
                            st.warning("No data to display for N-gram Treemap. Adjust parameters or input text.")

                except Exception as e:
                    st.error(f"An error occurred during N-gram Analysis: {e}")

            elif analysis_type == "Word Cloud (Traditional)":
                st.info(f"Processing {len(lines)} text document(s) for Word Cloud...")
                npt_traditional = get_nlplot_instance_for_traditional_nlp(lines, target_column_name="input_tokens_wc")

                # Prepare stopwords list for word cloud
                wc_stopwords_list = [sw.strip() for sw in wc_stopwords_str.split(',') if sw.strip()]

                # Font path (optional) - for now, we'll let NLPlotLLM use its default
                # font_path_to_use = wc_font_path_str if wc_font_path_str.strip() else None

                try:
                    with st.spinner("Generating Word Cloud..."):
                        st.subheader(f"Word Cloud (Max Words: {wc_max_words})")
                        # The wordcloud method now returns a PIL Image object
                        pil_image = npt_traditional.wordcloud(
                            max_words=wc_max_words,
                            stopwords=wc_stopwords_list,
                            # font_path=font_path_to_use, # Pass if UI for font_path is enabled
                        )
                        if pil_image:
                            st.image(pil_image, use_column_width=True)
                        else:
                            st.warning("Could not generate Word Cloud. Input text might be empty or all words filtered out.")

                except Exception as e:
                    st.error(f"An error occurred during Word Cloud generation: {e}")

            elif analysis_type == "Japanese Text Analysis (Traditional)":
                if not JANOME_AVAILABLE:
                    st.error("Janome is not installed, cannot perform Japanese Text Analysis.")
                else:
                    st.info(f"Processing {len(lines)} text document(s) for Japanese Text Analysis...")
                    # For Japanese text features, we pass the raw text lines as a Series.
                    # NLPlotLLM instance can be the basic one used for LLM tasks, as get_japanese_text_features
                    # doesn't rely on the instance's df or target_col in the same way traditional plots do.
                    # However, it might be cleaner to have a dedicated instance or ensure the dummy one is fine.
                    # For now, using the LLM instance.
                    npt_jp_analyzer = get_nlplot_llm_instance() # Or a specific one if needed

                    text_series_jp = pd.Series(lines)

                    try:
                        with st.spinner("Calculating Japanese text features..."):
                            # Reset plot options flag before new calculation
                            st.session_state.show_jp_plot_options = False
                            st.session_state.jp_features_df = npt_jp_analyzer.get_japanese_text_features(text_series_jp)
                            st.session_state.analysis_type_at_run = analysis_type # Persist analysis type

                        # Display features and set flag to show plot options outside button block
                        if st.session_state.jp_features_df is not None and not st.session_state.jp_features_df.empty:
                            st.subheader("Japanese Text Features")
                            st.dataframe(st.session_state.jp_features_df)
                            st.session_state.show_jp_plot_options = True # Enable plot options display area
                        else:
                            st.warning("No features were calculated. Input text might be unsuitable or empty.")
                            st.session_state.show_jp_plot_options = False

                    except Exception as e:
                        st.error(f"An error occurred during Japanese Text Analysis feature calculation: {e}")
                        st.session_state.jp_features_df = None
                        st.session_state.show_jp_plot_options = False


# This block will now handle the display of plot options and the plot itself,
# based on session_state, outside the main "Run Analysis" button's conditional block.
if st.session_state.get('show_jp_plot_options', False) and \
   st.session_state.get('analysis_type_at_run') == "Japanese Text Analysis (Traditional)" and \
   st.session_state.jp_features_df is not None and \
   not st.session_state.jp_features_df.empty:

    st.subheader("Plot Japanese Text Feature Distribution")
    numeric_cols = st.session_state.jp_features_df.select_dtypes(include='number').columns.tolist()

    default_idx = 0
    if numeric_cols:
        if 'total_tokens' in numeric_cols:
            default_idx = numeric_cols.index('total_tokens')
    else: # Should not happen if df has data and numeric columns
        st.info("No numeric features available to plot.")

    if numeric_cols:
        # Using a consistent key for the selectbox
        selected_feature_for_plot = st.selectbox(
            "Select feature to plot:",
            options=numeric_cols,
            index=default_idx,
            key="jp_feature_selector_stable_key"
        )

        if selected_feature_for_plot:
            # Re-instantiate the analyzer for plotting (or ensure it's available)
            # For simplicity, re-instantiating here. Could be optimized if complex.
            npt_jp_plotter = get_nlplot_llm_instance()
            try:
                with st.spinner(f"Generating plot for {selected_feature_for_plot}..."):
                    fig_jp_plot = npt_jp_plotter.plot_japanese_text_features(
                        st.session_state.jp_features_df, # Use the stored DataFrame
                        target_feature=selected_feature_for_plot,
                        title=f"Distribution of {selected_feature_for_plot}"
                    )
                    if fig_jp_plot:
                        st.plotly_chart(fig_jp_plot, use_container_width=True)
                    else:
                        st.warning(f"Could not generate plot for {selected_feature_for_plot}.")
            except Exception as e:
                st.error(f"An error occurred during plot generation for '{selected_feature_for_plot}': {e}")

if not st.session_state.get('run_button_clicked', False): # Show initial caption if button not clicked yet
    st.caption(f"Click the 'Run {analysis_type}' button to start.")

# At the end of the script, or after the button logic, reset the button click state if needed,
# or manage it to allow re-runs. For now, assume selectbox change is the primary interaction after first run.
# If a clear "New Analysis" button or similar is added, that would be the place to reset states like 'jp_features_df'.

st.sidebar.markdown("---")
st.sidebar.markdown("Refer to [LiteLLM Documentation](https://docs.litellm.ai/docs/providers) for provider-specific API keys (e.g., `OPENAI_API_KEY`, `AZURE_API_KEY`, `COHERE_API_KEY`, `ANTHROPIC_API_KEY`) and other parameters. For local models like Ollama, ensure the server is running.")
st.sidebar.markdown("---")
st.sidebar.markdown("Powered by NLPlotLLM and Streamlit.")
st.sidebar.markdown("Ensure your LLM is configured and accessible as per LiteLLM requirements.")

# To run this app:
# 1. Ensure nlplot_llm, streamlit, pandas, litellm are installed.
#    pip install streamlit pandas litellm # nlplot_llm from local setup
# 2. Save this code as streamlit_app.py
# 3. Run `streamlit run streamlit_app.py` from your terminal.
#
# Note: The NLPlotLLM library itself would need to be installed (e.g., editable install `pip install -e .`)
# or its path added to PYTHONPATH for the import to work correctly if running from a different directory.
# For simplicity, this assumes `nlplot_llm` is importable.
#
# Example texts:
# Sentiment:
# This is a great day!
# I am very unhappy with the service.
# The weather is just okay.
#
# Categorization (Categories: news, sports, food):
# The local team won their championship game last night.
# A new recipe for apple pie has been released.
# The government announced new economic policies.
# This dish is delicious and easy to make. (multi-label possible with food)
# The election results are in. (news)
# He scored a goal in the final minute. (sports)

print("Streamlit App Script Created/Updated.")
