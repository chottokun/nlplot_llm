import asyncio
import pandas as pd
from typing import Optional, List

try:
    import litellm
    from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
    LITELLM_AVAILABLE = True
    LANGCHAIN_SPLITTERS_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    LANGCHAIN_SPLITTERS_AVAILABLE = False
    class litellm_dummy: # type: ignore
        def completion(self, *args, **kwargs): raise ImportError("litellm is not installed.")
        class exceptions: # type: ignore
            class APIConnectionError(Exception): pass
            class AuthenticationError(Exception): pass
            class RateLimitError(Exception): pass
    litellm = litellm_dummy() # type: ignore
    class RecursiveCharacterTextSplitter: # type: ignore
        def __init__(self, chunk_size=None, chunk_overlap=None, length_function=None, **kwargs): pass
        def split_text(self, text: str) -> List[str]: return [text] if text else []
    class CharacterTextSplitter: # type: ignore
        def __init__(self, separator=None, chunk_size=None, chunk_overlap=None, length_function=None, **kwargs): pass
        def split_text(self, text: str) -> List[str]: return [text] if text else []

def _chunk_text(
    text_to_chunk: str,
    strategy: str = "recursive_char",
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    **splitter_kwargs
) -> List[str]:
    if not LANGCHAIN_SPLITTERS_AVAILABLE:
        print("Warning: Langchain text splitter components are not installed. Chunking will not be performed; returning original text as a single chunk.")
        return [text_to_chunk] if text_to_chunk else []

    if not text_to_chunk:
        return []

    if strategy == "recursive_char":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            **splitter_kwargs
        )
    elif strategy == "character":
        separator = splitter_kwargs.pop("separator", "\n\n")
        splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            **splitter_kwargs
        )
    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}. Supported strategies: 'recursive_char', 'character'.")

    try:
        return splitter.split_text(text_to_chunk)
    except Exception as e:
        print(f"Error during text chunking with strategy '{strategy}': {e}")
        return [text_to_chunk]

async def _summarize_text_single_async(
    nlplot_instance,
    text_to_summarize: str,
    model: str,
    direct_prompt_template_str: str,
    chunk_prompt_template_str_in: Optional[str],
    combine_prompt_template_str_in: Optional[str],
    temperature: float,
    use_chunking: bool,
    chunk_size: int,
    chunk_overlap: int,
    current_use_cache: bool,
    litellm_kwargs: dict
) -> dict:
    original_text_for_df = str(text_to_summarize) if pd.notna(text_to_summarize) else ""
    summary_text = ""
    raw_llm_response_agg = ""

    if not original_text_for_df.strip():
        summary_text = ""
        raw_llm_response_agg = "Input text was empty or whitespace."
    else:
        cacheable_kwargs_items = sorted(
            (k, v) for k, v in litellm_kwargs.items() if k in ("temperature", "max_tokens", "top_p")
        )
        _actual_chunk_prompt = chunk_prompt_template_str_in if chunk_prompt_template_str_in else "Concisely summarize this piece of text: {text}"
        _actual_combine_prompt = combine_prompt_template_str_in

        num_chunks_for_key = 0
        if use_chunking and LANGCHAIN_SPLITTERS_AVAILABLE:
            temp_chunks_for_key = _chunk_text(original_text_for_df, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            num_chunks_for_key = len(temp_chunks_for_key)

        cache_key_tuple = (
            "summarize_text_llm", model,
            direct_prompt_template_str,
            _actual_chunk_prompt if use_chunking else None,
            _actual_combine_prompt if use_chunking and num_chunks_for_key > 1 else None,
            original_text_for_df,
            use_chunking, chunk_size if use_chunking else None, chunk_overlap if use_chunking else None,
            tuple(cacheable_kwargs_items)
        )

        cached_result = None
        if current_use_cache and nlplot_instance.cache:
            try: cached_result = nlplot_instance.cache.get(cache_key_tuple)
            except Exception as e: print(f"Warning: Async cache get failed for summarize. Error: {e}")

        if cached_result is not None:
            summary_text = cached_result["summary"]
            raw_llm_response_agg = cached_result["raw_llm_output"]
        else:
            _final_completion_kwargs = {**litellm_kwargs}
            if 'temperature' not in _final_completion_kwargs: _final_completion_kwargs['temperature'] = temperature

            if not use_chunking or not LANGCHAIN_SPLITTERS_AVAILABLE:
                try:
                    messages = [{"role": "user", "content": direct_prompt_template_str.format(text=original_text_for_df)}]
                    response = await litellm.acompletion(model=model, messages=messages, **_final_completion_kwargs)
                    summary_text = response.choices[0].message.content.strip() if response.choices and response.choices[0].message else ""
                    raw_llm_response_agg = str(response)
                    if use_chunking and not LANGCHAIN_SPLITTERS_AVAILABLE : raw_llm_response_agg = f"Chunking skipped (splitters unavailable). {raw_llm_response_agg}"
                except Exception as e:
                    summary_text, raw_llm_response_agg = "error", f"Async direct/fallback summary error: {e}"
            else:
                chunks = _chunk_text(original_text_for_df, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if not chunks:
                    summary_text, raw_llm_response_agg = "", "Text was empty or too short after attempting to chunk."
                else:
                    chunk_summaries_list, raw_chunk_outputs_list = [], []
                    current_chunk_prompt_str = _actual_chunk_prompt
                    if "{text}" not in current_chunk_prompt_str:
                        summary_text, raw_llm_response_agg = "error_prompt_chunk", "Chunk prompt template error: missing {text}"
                    else:
                        chunk_tasks = []
                        for chunk_item_text in chunks:
                            messages_chunk = [{"role": "user", "content": current_chunk_prompt_str.format(text=chunk_item_text)}]
                            chunk_tasks.append(litellm.acompletion(model=model, messages=messages_chunk, **_final_completion_kwargs))

                        try:
                            chunk_responses = await asyncio.gather(*chunk_tasks, return_exceptions=True)
                            for i_chunk, response_chunk_or_exc in enumerate(chunk_responses):
                                if isinstance(response_chunk_or_exc, Exception):
                                    chunk_summaries_list.append(f"[Error in chunk {i_chunk+1}]")
                                    raw_chunk_outputs_list.append(f"Chunk {i_chunk+1} Error: {str(response_chunk_or_exc)}")
                                elif response_chunk_or_exc.choices and response_chunk_or_exc.choices[0].message:
                                    chunk_summaries_list.append(response_chunk_or_exc.choices[0].message.content.strip())
                                    raw_chunk_outputs_list.append(f"Chunk {i_chunk+1} Raw: {str(response_chunk_or_exc)}")
                                else:
                                    chunk_summaries_list.append("")
                                    raw_chunk_outputs_list.append(f"Chunk {i_chunk+1} Raw (empty): {str(response_chunk_or_exc)}")
                        except Exception as e_gather_chunks:
                             summary_text, raw_llm_response_agg = "error", f"Error gathering chunk summaries: {e_gather_chunks}"

                    if not summary_text.startswith("error"):
                        intermediate_summary = "\n\n".join(chunk_summaries_list)
                        raw_llm_response_agg = "\n---\n".join(raw_chunk_outputs_list)

                        if _actual_combine_prompt and len(chunks) > 1:
                            if "{text}" not in _actual_combine_prompt:
                                summary_text = intermediate_summary
                                raw_llm_response_agg += "\n---\nWarning: Combine prompt invalid, using intermediate."
                            else:
                                try:
                                    messages_combine = [{"role": "user", "content": _actual_combine_prompt.format(text=intermediate_summary)}]
                                    response_combine = await litellm.acompletion(model=model, messages=messages_combine, **_final_completion_kwargs)
                                    summary_text = response_combine.choices[0].message.content.strip() if response_combine.choices and response_combine.choices[0].message else ""
                                    raw_llm_response_agg += f"\n---\nFinal Combined Summary Raw: {str(response_combine)}"
                                except Exception as e_combine:
                                    summary_text = f"[Error combining: {intermediate_summary[:100]}...]"
                                    raw_llm_response_agg += f"\n---\nError combining summaries: {e_combine}"
                        else:
                            summary_text = intermediate_summary

            if current_use_cache and nlplot_instance.cache and not summary_text.startswith("error_") and not summary_text.startswith("[Error"):
                try: nlplot_instance.cache.set(cache_key_tuple, {"summary": summary_text, "raw_llm_output": raw_llm_response_agg})
                except Exception as e: print(f"Warning: Async cache set failed for summarize. Error: {e}")

    return {"original_text": original_text_for_df, "summary": summary_text, "raw_llm_output": raw_llm_response_agg}

async def summarize_text_llm_async(
    nlplot_instance,
    text_series: pd.Series, model: str, prompt_template_str: Optional[str]=None,
    chunk_prompt_template_str: Optional[str]=None, combine_prompt_template_str: Optional[str]=None,
    temperature: float=0.0, use_chunking: bool=True, chunk_size: int=1000, chunk_overlap: int=100,
    use_cache: Optional[bool]=None, concurrency_limit: Optional[int]=None, return_exceptions: bool=True,
    max_length: Optional[int]=None, min_length: Optional[int]=None, **litellm_kwargs
) -> pd.DataFrame:
    cols = ["original_text", "summary", "raw_llm_output"]
    if not LITELLM_AVAILABLE: return pd.DataFrame(columns=cols)
    if not isinstance(text_series, pd.Series): return pd.DataFrame(columns=cols)
    if text_series.empty: return pd.DataFrame(columns=cols)

    use_cache_flag = use_cache if use_cache is not None else nlplot_instance.use_cache_default
    kw_main = {**litellm_kwargs}
    if 'temperature' not in kw_main: kw_main['temperature'] = temperature
    if max_length and 'max_tokens' not in kw_main: kw_main['max_tokens'] = max_length

    direct_prompt = prompt_template_str if prompt_template_str else "Please summarize the following text concisely: {text}"
    if "{text}" not in direct_prompt:
        return pd.DataFrame([{"original_text":str(t),"summary":"error_prompt_direct","raw_llm_output":"Direct prompt error"} for t in text_series], columns=cols)
    if use_chunking and chunk_prompt_template_str and "{text}" not in chunk_prompt_template_str:
        return pd.DataFrame([{"original_text":str(t),"summary":"error_prompt_chunk","raw_llm_output":"Chunk prompt error"} for t in text_series], columns=cols)

    tasks = []
    sem = asyncio.Semaphore(concurrency_limit) if concurrency_limit and concurrency_limit > 0 else None
    async def wrapper(text):
        if sem:
            async with sem:
                return await _summarize_text_single_async(nlplot_instance, text, model, direct_prompt, chunk_prompt_template_str, combine_prompt_template_str, temperature, use_chunking, chunk_size, chunk_overlap, use_cache_flag, kw_main)
        else:
            return await _summarize_text_single_async(nlplot_instance, text, model, direct_prompt, chunk_prompt_template_str, combine_prompt_template_str, temperature, use_chunking, chunk_size, chunk_overlap, use_cache_flag, kw_main)
    for txt_in in text_series: tasks.append(wrapper(txt_in))
    raw_res = await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    proc_res = []
    for i, r_exc in enumerate(raw_res):
        orig_txt = str(text_series.iloc[i]) if pd.notna(text_series.iloc[i]) else ""
        if isinstance(r_exc, Exception): proc_res.append({"original_text":orig_txt, "summary":"error", "raw_llm_output":str(r_exc)})
        else: proc_res.append(r_exc)
    if nlplot_instance.cache:
        try: nlplot_instance.cache.close()
        except Exception as e: print(f"Warning: Error closing cache: {e}")
    return pd.DataFrame(proc_res, columns=cols)


def summarize_text_llm(
    nlplot_instance,
    text_series: pd.Series,
    model: str,
    prompt_template_str: Optional[str] = None,
    chunk_prompt_template_str: Optional[str] = None,
    combine_prompt_template_str: Optional[str] = None,
    max_length: Optional[int] = None,
    min_length: Optional[int] = None,
    temperature: float = 0.0,
    use_chunking: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    use_cache: Optional[bool] = None,
    **litellm_kwargs
) -> pd.DataFrame:
    default_columns = ["original_text", "summary", "raw_llm_output"]
    if not LITELLM_AVAILABLE:
        print("Warning: LiteLLM is not installed. LLM-based summarization is not available.")
        return pd.DataFrame(columns=default_columns)

    if not isinstance(text_series, pd.Series):
        print("Warning: Input 'text_series' must be a pandas Series.")
        return pd.DataFrame(columns=default_columns)
    if text_series.empty:
        return pd.DataFrame(columns=default_columns)

    current_use_cache_for_call = use_cache if use_cache is not None else nlplot_instance.use_cache_default

    _litellm_kwargs = {**litellm_kwargs}
    if 'temperature' not in _litellm_kwargs:
        _litellm_kwargs['temperature'] = temperature
    if max_length is not None and 'max_tokens' not in _litellm_kwargs:
         _litellm_kwargs['max_tokens'] = max_length

    final_direct_prompt_template_str = prompt_template_str if prompt_template_str else "Please summarize the following text concisely: {text}"
    if "{text}" not in final_direct_prompt_template_str:
        print("Error: Direct summarization prompt template must include '{text}' placeholder.")
        return pd.DataFrame(
            [{'original_text': str(txt) if pd.notna(txt) else "", 'summary': 'error_prompt_direct', 'raw_llm_output': "Direct prompt error"} for txt in text_series],
            columns=default_columns
        )

    results = []
    for text_input in text_series:
        original_text_for_df = str(text_input) if pd.notna(text_input) else ""
        summary_text = ""
        raw_llm_response_agg = ""

        if not original_text_for_df.strip():
            summary_text = ""
            raw_llm_response_agg = "Input text was empty or whitespace."
        else:
            cacheable_kwargs_items = sorted(
                (k, v) for k, v in _litellm_kwargs.items()
                if k in ("temperature", "max_tokens", "top_p")
            )
            _effective_chunk_prompt = chunk_prompt_template_str if chunk_prompt_template_str else "Concisely summarize this piece of text: {text}"
            _effective_combine_prompt = combine_prompt_template_str

            num_chunks_for_key = 0
            if use_chunking and LANGCHAIN_SPLITTERS_AVAILABLE:
                temp_chunks_for_key = _chunk_text(original_text_for_df, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                num_chunks_for_key = len(temp_chunks_for_key)

            cache_key_tuple = (
                "summarize_text_llm", model,
                final_direct_prompt_template_str,
                _effective_chunk_prompt if use_chunking else None,
                _effective_combine_prompt if use_chunking and num_chunks_for_key > 1 else None,
                original_text_for_df,
                use_chunking, chunk_size if use_chunking else None, chunk_overlap if use_chunking else None,
                tuple(cacheable_kwargs_items)
            )

            cached_result = None
            if current_use_cache_for_call and nlplot_instance.cache:
                try:
                    cached_result = nlplot_instance.cache.get(cache_key_tuple)
                except Exception as e:
                    print(f"Warning: Cache get failed for key {cache_key_tuple}. Error: {e}")

            if cached_result is not None:
                summary_text = cached_result["summary"]
                raw_llm_response_agg = cached_result["raw_llm_output"]
            else:
                if not use_chunking:
                    try:
                        messages = [{"role": "user", "content": final_direct_prompt_template_str.format(text=original_text_for_df)}]
                        response = litellm.completion(model=model, messages=messages, **_litellm_kwargs)
                        summary_text = response.choices[0].message.content.strip() if response.choices and response.choices[0].message else ""
                        raw_llm_response_agg = str(response)
                    except Exception as e:
                        print(f"Error during direct summarization for text '{original_text_for_df[:50]}...': {e}")
                        summary_text, raw_llm_response_agg = "error", f"Direct summarization error: {e}"
                else:
                    if not LANGCHAIN_SPLITTERS_AVAILABLE:
                        print("Warning: Langchain text splitters not available. Attempting direct summarization.")
                        try:
                            messages = [{"role": "user", "content": final_direct_prompt_template_str.format(text=original_text_for_df)}]
                            response = litellm.completion(model=model, messages=messages, **_litellm_kwargs)
                            summary_text = response.choices[0].message.content.strip() if response.choices and response.choices[0].message else ""
                            raw_llm_response_agg = f"Chunking skipped. Direct summary raw: {str(response)}"
                        except Exception as e:
                            print(f"Error during fallback direct summarization: {e}"); summary_text, raw_llm_response_agg = "error", f"Fallback direct summarization error: {e}"
                    else:
                        chunks = _chunk_text(original_text_for_df, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        if not chunks:
                            summary_text, raw_llm_response_agg = "", "Text too short after chunking."
                        else:
                            chunk_summaries_list, raw_chunk_outputs_list = [], []
                            current_chunk_prompt = _effective_chunk_prompt
                            if "{text}" not in current_chunk_prompt:
                                summary_text, raw_llm_response_agg = "error_prompt_chunk", "Chunk prompt error"
                                results.append({"original_text": original_text_for_df, "summary": summary_text, "raw_llm_output": raw_llm_response_agg}); continue

                            for i, chunk_item_text in enumerate(chunks):
                                try:
                                    messages_chunk = [{"role": "user", "content": current_chunk_prompt.format(text=chunk_item_text)}]
                                    response_chunk = litellm.completion(model=model, messages=messages_chunk, **_litellm_kwargs)
                                    chunk_summary = response_chunk.choices[0].message.content.strip() if response_chunk.choices and response_chunk.choices[0].message else ""
                                    chunk_summaries_list.append(chunk_summary)
                                    raw_chunk_outputs_list.append(f"Chunk {i+1}/{len(chunks)} Raw: {str(response_chunk)}")
                                except Exception as e_chunk:
                                    chunk_summaries_list.append(f"[Err chunk {i+1}]"); raw_chunk_outputs_list.append(f"Chunk {i+1} Err: {e_chunk}")

                            intermediate_summary = "\n\n".join(chunk_summaries_list)
                            raw_llm_response_agg = "\n---\n".join(raw_chunk_outputs_list)

                            if _effective_combine_prompt and len(chunks) > 1:
                                if "{text}" not in _effective_combine_prompt:
                                    summary_text = intermediate_summary
                                    raw_llm_response_agg += "\n---\nWarn: Combine prompt invalid."
                                else:
                                    try:
                                        messages_combine = [{"role": "user", "content": _effective_combine_prompt.format(text=intermediate_summary)}]
                                        response_combine = litellm.completion(model=model, messages=messages_combine, **_litellm_kwargs)
                                        summary_text = response_combine.choices[0].message.content.strip() if response_combine.choices and response_combine.choices[0].message else ""
                                        raw_llm_response_agg += f"\n---\nCombined Raw: {str(response_combine)}"
                                    except Exception as e_combine:
                                        summary_text = f"[Err combine: {intermediate_summary[:100]}...]"; raw_llm_response_agg += f"\n---\nErr combine: {e_combine}"
                            else:
                                summary_text = intermediate_summary

                if current_use_cache_for_call and nlplot_instance.cache and not summary_text.startswith("error_") and not summary_text.startswith("[Err"):
                    try:
                        nlplot_instance.cache.set(cache_key_tuple, {"summary": summary_text, "raw_llm_output": raw_llm_response_agg})
                    except Exception as e:
                        print(f"Warning: Cache set failed for key {cache_key_tuple}. Error: {e}")

        results.append({
            "original_text": original_text_for_df,
            "summary": summary_text,
            "raw_llm_output": raw_llm_response_agg
        })

    if nlplot_instance.cache:
        try: nlplot_instance.cache.close()
        except Exception as e: print(f"Warning: Error closing cache: {e}")

    return pd.DataFrame(results, columns=default_columns)
