import streamlit as st
import pandas as pd
import os # Added for os.getenv
from nlplot_llm import NLPlotLLM # Updated import

# --- App Configuration ---
st.set_page_config(page_title="NLPlotLLM Demo", layout="wide") # Updated title

# --- Helper Functions (placeholder for now) ---
def get_nlplot_llm_instance(): # Updated function name
    """Creates and returns an NLPlotLLM instance.""" # Updated class name in docstring
    # For the demo, we might not need a DataFrame for NLPlotLLM initialization
    # if we are only using LLM methods that take Series directly.
    # However, NLPlotLLM requires a df and target_col for its constructor.
    # We can use a dummy one.
    dummy_df = pd.DataFrame({'text': ["dummy text for nlplot_llm init"]})
    # Get cache setting from sidebar
    use_llm_cache = st.session_state.get("use_llm_cache", True) # Default to True if not set
    return NLPlotLLM(dummy_df, target_col='text', use_cache=use_llm_cache)

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
analysis_options = ["Sentiment Analysis", "Text Categorization", "Text Summarization"]
analysis_type = st.selectbox("Select Analysis", analysis_options)

# --- Analysis Specific Options ---

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


# Execute Button
if st.button(f"Run {analysis_type}"):
    if not input_text.strip():
        st.error("Please enter some text to analyze.")
    elif not model_string.strip():
        st.error("Please enter a LiteLLM Model String.")
    # Removed specific OpenAI key check here as LiteLLM handles various auth methods.
    # The warning in the sidebar should guide the user.
    else:
        lines = [line.strip() for line in input_text.split('\n') if line.strip()]
        if not lines:
            st.error("No valid text lines found after stripping.")
        else:
            text_series = pd.Series(lines)
            npt = get_nlplot_llm_instance()

            st.info(f"Processing {len(text_series)} text document(s) using model: {model_string}...")

            try:
                with st.spinner("Analyzing..."):
                    if analysis_type == "Sentiment Analysis":
                        st.subheader("Sentiment Analysis Results")
                        analyze_kwargs = litellm_kwargs.copy()
                        if prompt_sentiment.strip():
                            analyze_kwargs["prompt_template_str"] = prompt_sentiment
                        result_df = npt.analyze_sentiment_llm(
                            text_series=text_series,
                            model=model_string, # Use the LiteLLM model string
                            **analyze_kwargs # Pass all collected LiteLLM kwargs
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
                                result_df = npt.categorize_text_llm(
                                    text_series=text_series,
                                    categories=categories_list,
                                    model=model_string, # Use the LiteLLM model string
                                    multi_label=multi_label_categories,
                                    **categorize_kwargs # Pass all collected LiteLLM kwargs
                                )
                                st.dataframe(result_df)

                    elif analysis_type == "Text Summarization":
                        st.subheader("Text Summarization Results")

                        # Start with base litellm_kwargs
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
                            if chunk_prompt_template_summarize.strip(): # This UI field is reused for direct prompt
                                summarize_final_kwargs["prompt_template_str"] = chunk_prompt_template_summarize

                        result_df = npt.summarize_text_llm(
                            text_series=text_series,
                            model=model_string, # Use the LiteLLM model string
                            **summarize_final_kwargs
                        )
                        st.dataframe(result_df)

            except ImportError as ie:
                st.error(f"ImportError: {ie}. Make sure Langchain and necessary LLM provider libraries are installed.")
            except ValueError as ve:
                st.error(f"ValueError: {ve}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
else:
    st.caption(f"Click the 'Run {analysis_type}' button to start.") # Dynamic button text in caption

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
