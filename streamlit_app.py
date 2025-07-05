import streamlit as st
import pandas as pd
from nlplot import NLPlot # Assuming nlplot is installed or in PYTHONPATH

# --- App Configuration ---
st.set_page_config(page_title="NLPlot LLM Demo", layout="wide")

# --- Helper Functions (placeholder for now) ---
def get_nlplot_instance():
    """Creates and returns an NLPlot instance."""
    # For the demo, we might not need a DataFrame for NLPlot initialization
    # if we are only using LLM methods that take Series directly.
    # However, NLPlot requires a df and target_col for its constructor.
    # We can use a dummy one.
    dummy_df = pd.DataFrame({'text': ["dummy text for nlplot init"]})
    return NLPlot(dummy_df, target_col='text')

# --- Sidebar for LLM Configuration ---
st.sidebar.header("LLM Configuration")
llm_provider = st.sidebar.selectbox("LLM Provider", ["OpenAI", "Ollama"], index=0)
model_name = st.sidebar.text_input("Model Name", value="gpt-3.5-turbo" if llm_provider == "OpenAI" else "llama2")

llm_config = {}
if llm_provider == "OpenAI":
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if openai_api_key:
        llm_config["openai_api_key"] = openai_api_key
    else:
        st.sidebar.warning("OpenAI API Key is required.")
elif llm_provider == "Ollama":
    base_url = st.sidebar.text_input("Ollama Base URL", value="http://localhost:11434")
    llm_config["base_url"] = base_url

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
llm_config["temperature"] = temperature


# --- Main App Area ---
st.title("NLPlot LLM Functions Demo")

# Input Text Area
st.header("Input Text")
input_text = st.text_area("Enter text to analyze (one document per line for multiple inputs):", height=200,
                          value="This is a wonderfully positive statement!\nThis movie was terrible and a waste of time.\nGlobal warming is a serious issue that needs addressing.")

# Analysis Type Selection
st.header("Analysis Type")
analysis_options = ["Sentiment Analysis", "Text Categorization", "Text Summarization"]
analysis_type = st.selectbox("Select Analysis", analysis_options)

# Analysis Specific Options
categories_input_str = ""
multi_label_categories = False
use_chunking_summarize = True
chunk_size_summarize = 1000
chunk_overlap_summarize = 100
chunk_prompt_template_summarize = ""
combine_prompt_template_summarize = ""

if analysis_type == "Text Categorization":
    st.subheader("Categorization Options")
    categories_input_str = st.text_input("Categories (comma-separated)", value="news,sports,technology,finance,health")
    multi_label_categories = st.checkbox("Allow multiple labels per text", value=False)
elif analysis_type == "Text Summarization":
    st.subheader("Summarization Options")
    use_chunking_summarize = st.checkbox("Use Chunking for Long Texts", value=True)
    if use_chunking_summarize:
        chunk_size_summarize = st.number_input("Chunk Size", min_value=100, max_value=10000, value=1000, step=100)
        chunk_overlap_summarize = st.number_input("Chunk Overlap", min_value=0, max_value=chunk_size_summarize-50 if chunk_size_summarize > 50 else 0, value=100, step=50)
        chunk_prompt_template_summarize = st.text_area(
            "Chunk Summarization Prompt (optional, use {text})",
            value="Summarize this text concisely: {text}",
            height=100
        )
        combine_prompt_template_summarize = st.text_area(
            "Combine Summaries Prompt (optional, use {text} for combined chunks)",
            value="Combine the following summaries into a coherent final summary: {text}",
            height=100
        )
    else: # Direct summarization prompt (if not chunking)
        # This could also be the 'prompt_template_str' for summarize_text_llm
        # For simplicity, let's use a general prompt if not chunking, or allow user to specify one.
         chunk_prompt_template_summarize = st.text_area( # Re-use for direct prompt if not chunking
            "Summarization Prompt (use {text})",
            value="Please summarize the following text: {text}",
            height=100
        )


# Execute Button
if st.button(f"Run {analysis_type}"):
    if not input_text.strip():
        st.error("Please enter some text to analyze.")
    elif llm_provider == "OpenAI" and not openai_api_key:
        st.error("OpenAI API Key is required to run the analysis.")
    else:
        lines = [line.strip() for line in input_text.split('\n') if line.strip()]
        if not lines:
            st.error("No valid text lines found after stripping.")
        else:
            text_series = pd.Series(lines)
            npt = get_nlplot_instance()

            st.info(f"Processing {len(text_series)} text document(s) using {llm_provider} ({model_name})...")

            try:
                with st.spinner("Analyzing..."):
                    if analysis_type == "Sentiment Analysis":
                        st.subheader("Sentiment Analysis Results")
                        result_df = npt.analyze_sentiment_llm(
                            text_series=text_series,
                            llm_provider=llm_provider.lower(),
                            model_name=model_name,
                            **llm_config
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
                                result_df = npt.categorize_text_llm(
                                    text_series=text_series,
                                    categories=categories_list,
                                    llm_provider=llm_provider.lower(),
                                    model_name=model_name,
                                    multi_label=multi_label_categories,
                                    **llm_config
                                )
                                st.dataframe(result_df)

                    elif analysis_type == "Text Summarization":
                        st.subheader("Text Summarization Results")

                        summarize_kwargs = {
                            "llm_provider": llm_provider.lower(),
                            "model_name": model_name,
                            "use_chunking": use_chunking_summarize,
                            **llm_config
                        }
                        if use_chunking_summarize:
                            summarize_kwargs["chunk_size"] = chunk_size_summarize
                            summarize_kwargs["chunk_overlap"] = chunk_overlap_summarize
                            if chunk_prompt_template_summarize.strip():
                                summarize_kwargs["chunk_prompt_template_str"] = chunk_prompt_template_summarize
                            if combine_prompt_template_summarize.strip():
                                summarize_kwargs["combine_prompt_template_str"] = combine_prompt_template_summarize
                        else: # Direct summarization (not using chunking specific prompts)
                            if chunk_prompt_template_summarize.strip(): # Re-using this field for direct prompt
                                summarize_kwargs["prompt_template_str"] = chunk_prompt_template_summarize

                        result_df = npt.summarize_text_llm(
                            text_series=text_series,
                            **summarize_kwargs
                        )
                        st.dataframe(result_df)

            except ImportError as ie:
                st.error(f"ImportError: {ie}. Make sure Langchain and necessary LLM provider libraries are installed.")
            except ValueError as ve:
                st.error(f"ValueError: {ve}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
else:
    st.caption("Click the 'Analyze Text' button to start.")

st.sidebar.markdown("---")
st.sidebar.markdown("Powered by [NLPlot](https://github.com/your-repo/nlplot) and Streamlit.") # Replace with actual repo link if available
st.sidebar.markdown("Ensure your LLM (OpenAI API or Ollama server) is configured and accessible.")

# To run this app:
# 1. Ensure nlplot, streamlit, langchain, openai, langchain-community are installed.
#    pip install streamlit pandas nlplot langchain openai langchain-community
# 2. Save this code as streamlit_app.py
# 3. Run `streamlit run streamlit_app.py` from your terminal.
#
# Note: The NLPlot library itself would need to be installed (e.g., editable install `pip install -e .`)
# or its path added to PYTHONPATH for the import to work correctly if running from a different directory.
# For simplicity, this assumes `nlplot` is importable.
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
