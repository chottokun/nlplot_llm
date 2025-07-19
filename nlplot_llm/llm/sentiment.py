import asyncio
import pandas as pd
from typing import Optional

try:
    import litellm

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    # Dummy class for litellm if not installed

    class litellm_dummy:
        def completion(self, *args, **kwargs):
            raise ImportError("litellm is not installed.")

        class exceptions:
            class APIConnectionError(Exception):
                pass

            class AuthenticationError(Exception):
                pass

            class RateLimitError(Exception):
                pass

    litellm = litellm_dummy()


def analyze_sentiment_llm(
    nlplot_instance,
    text_series: pd.Series,
    model: str,
    prompt_template_str: Optional[str] = None,
    temperature: float = 0.0,
    use_cache: Optional[bool] = None,
    **litellm_kwargs,
) -> pd.DataFrame:
    if not LITELLM_AVAILABLE:
        print(
            "Warning: LiteLLM is not installed. LLM-based sentiment "
            "analysis is not available."
        )
        return pd.DataFrame(columns=["text", "sentiment", "raw_llm_output"])

    if not isinstance(text_series, pd.Series):
        print(
            "Warning: Input 'text_series' must be a pandas Series. "
            "Returning empty DataFrame with expected columns."
        )
        return pd.DataFrame(columns=["text", "sentiment", "raw_llm_output"])
    if text_series.empty:
        return pd.DataFrame(columns=["text", "sentiment", "raw_llm_output"])

    current_use_cache = (
        use_cache if use_cache is not None else nlplot_instance.use_cache_default
    )

    final_prompt_template_str = prompt_template_str
    if final_prompt_template_str is None:
        final_prompt_template_str = (
            "Analyze the sentiment of the following text and classify it as "
            "'positive', 'negative', or 'neutral'. Return only the single "
            "word classification for the sentiment. Text: {text}"
        )

    if "{text}" not in final_prompt_template_str:
        print("Error: Prompt template must include '{text}' placeholder.")
        return pd.DataFrame(
            [
                {
                    "text": str(txt) if pd.notna(txt) else "",
                    "sentiment": "error",
                    "raw_llm_output": "Prompt template error: missing {text}",
                }
                for txt in text_series
            ],
            columns=["text", "sentiment", "raw_llm_output"],
        )

    results = []
    for text_input in text_series:
        original_text_for_df = str(text_input) if pd.notna(text_input) else ""
        sentiment = "unknown"
        raw_llm_response_content = ""

        if not original_text_for_df.strip():
            sentiment = "neutral"
            raw_llm_response_content = "Input text was empty or whitespace."
        else:
            cacheable_kwargs_items = sorted(
                (k, v)
                for k, v in litellm_kwargs.items()
                if k in ("temperature", "max_tokens", "top_p")
            )
            cache_key_tuple = (
                "analyze_sentiment_llm",
                model,
                final_prompt_template_str,
                original_text_for_df,
                tuple(cacheable_kwargs_items),
            )

            cached_result = None
            if current_use_cache and nlplot_instance.cache:
                try:
                    cached_result = nlplot_instance.cache.get(cache_key_tuple)
                except Exception as e:
                    print(
                        "Warning: Cache get failed for key "
                        f"{cache_key_tuple}. Error: {e}"
                    )

            if cached_result is not None:
                sentiment = cached_result["sentiment"]
                raw_llm_response_content = cached_result["raw_llm_output"]
            else:
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": final_prompt_template_str.format(
                                text=original_text_for_df
                            ),
                        }
                    ]
                    completion_kwargs = {k: v for k, v in litellm_kwargs.items()}
                    if "temperature" not in completion_kwargs:
                        completion_kwargs["temperature"] = temperature

                    response = litellm.completion(
                        model=model, messages=messages, **completion_kwargs
                    )
                    if response.choices and response.choices[0].message:
                        raw_llm_response_content = (
                            response.choices[0].message.content or ""
                        )
                    else:
                        raw_llm_response_content = str(response)

                    processed_output = raw_llm_response_content.strip().lower()
                    if "positive" in processed_output:
                        sentiment = "positive"
                    elif "negative" in processed_output:
                        sentiment = "negative"
                    elif "neutral" in processed_output:
                        sentiment = "neutral"

                    if current_use_cache and nlplot_instance.cache:
                        try:
                            nlplot_instance.cache.set(
                                cache_key_tuple,
                                {
                                    "sentiment": sentiment,
                                    "raw_llm_output": raw_llm_response_content,
                                },
                            )
                        except Exception as e:
                            print(
                                "Warning: Cache set failed for key "
                                f"{cache_key_tuple}. Error: {e}"
                            )

                except ImportError:
                    print(
                        "LiteLLM not installed. Cannot perform sentiment " "analysis."
                    )
                    sentiment = "error"
                    raw_llm_response_content = "LiteLLM not installed."
                except litellm.exceptions.AuthenticationError as e:
                    print(f"LiteLLM Authentication Error: {e}")
                    sentiment = "error"
                    raw_llm_response_content = f"AuthenticationError: {e}"
                except litellm.exceptions.APIConnectionError as e:
                    print(f"LiteLLM API Connection Error: {e}")
                    sentiment = "error"
                    raw_llm_response_content = f"APIConnectionError: {e}"
                except litellm.exceptions.RateLimitError as e:
                    print(f"LiteLLM Rate Limit Error: {e}")
                    sentiment = "error"
                    raw_llm_response_content = f"RateLimitError: {e}"
                except Exception as e:
                    print(
                        "Error analyzing sentiment for text "
                        f"'{original_text_for_df[:50]}...': {e}"
                    )
                    sentiment = "error"
                    raw_llm_response_content = str(e)

        results.append(
            {
                "text": original_text_for_df,
                "sentiment": sentiment,
                "raw_llm_output": raw_llm_response_content,
            }
        )

    if nlplot_instance.cache:
        try:
            nlplot_instance.cache.close()
        except Exception as e:
            print(f"Warning: Error closing cache: {e}")

    return pd.DataFrame(results, columns=["text", "sentiment", "raw_llm_output"])


async def _analyze_sentiment_single_async(
    nlplot_instance,
    text_to_analyze: str,
    model: str,
    prompt_template_str: str,
    temperature: float,
    current_use_cache: bool,
    litellm_kwargs: dict,
) -> dict:
    original_text_for_df = str(text_to_analyze) if pd.notna(text_to_analyze) else ""
    sentiment = "unknown"
    raw_llm_response_content = ""

    if not original_text_for_df.strip():
        sentiment = "neutral"
        raw_llm_response_content = "Input text was empty or whitespace."
    else:
        cacheable_kwargs_items = sorted(
            (k, v)
            for k, v in litellm_kwargs.items()
            if k in ("temperature", "max_tokens", "top_p")
        )
        cache_key_tuple = (
            "analyze_sentiment_llm",
            model,
            prompt_template_str,
            original_text_for_df,
            tuple(cacheable_kwargs_items),
        )

        cached_result = None
        if current_use_cache and nlplot_instance.cache:
            try:
                cached_result = nlplot_instance.cache.get(cache_key_tuple)
            except Exception as e:
                print(f"Warning: Async cache get failed. Error: {e}")

        if cached_result is not None:
            sentiment = cached_result["sentiment"]
            raw_llm_response_content = cached_result["raw_llm_output"]
        else:
            try:
                messages = [
                    {
                        "role": "user",
                        "content": prompt_template_str.format(
                            text=original_text_for_df
                        ),
                    }
                ]
                completion_kwargs = {**litellm_kwargs}
                if "temperature" not in completion_kwargs:
                    completion_kwargs["temperature"] = temperature

                response = await litellm.acompletion(
                    model=model, messages=messages, **completion_kwargs
                )

                if response.choices and response.choices[0].message:
                    raw_llm_response_content = response.choices[0].message.content or ""
                else:
                    raw_llm_response_content = str(response)

                processed_output = raw_llm_response_content.strip().lower()
                if "positive" in processed_output:
                    sentiment = "positive"
                elif "negative" in processed_output:
                    sentiment = "negative"
                elif "neutral" in processed_output:
                    sentiment = "neutral"

                if current_use_cache and nlplot_instance.cache:
                    try:
                        nlplot_instance.cache.set(
                            cache_key_tuple,
                            {
                                "sentiment": sentiment,
                                "raw_llm_output": raw_llm_response_content,
                            },
                        )
                    except Exception as e:
                        print(f"Warning: Async cache set failed. Error: {e}")
            except Exception as e:
                print(
                    "Error in _analyze_sentiment_single_async for "
                    f"'{original_text_for_df[:30]}...': {e}"
                )
                sentiment = "error"
                raw_llm_response_content = str(e)

    return {
        "text": original_text_for_df,
        "sentiment": sentiment,
        "raw_llm_output": raw_llm_response_content,
    }


async def analyze_sentiment_llm_async(
    nlplot_instance,
    text_series: pd.Series,
    model: str,
    prompt_template_str: Optional[str] = None,
    temperature: float = 0.0,
    use_cache: Optional[bool] = None,
    concurrency_limit: Optional[int] = None,
    return_exceptions: bool = True,
    **litellm_kwargs,
) -> pd.DataFrame:
    if not LITELLM_AVAILABLE:
        print(
            "Warning: LiteLLM is not installed. Async LLM-based sentiment "
            "analysis is not available."
        )
        return pd.DataFrame(columns=["text", "sentiment", "raw_llm_output"])
    if not isinstance(text_series, pd.Series):
        print("Warning: Input 'text_series' must be a pandas Series.")
        return pd.DataFrame(columns=["text", "sentiment", "raw_llm_output"])
    if text_series.empty:
        return pd.DataFrame(columns=["text", "sentiment", "raw_llm_output"])

    current_use_cache = (
        use_cache if use_cache is not None else nlplot_instance.use_cache_default
    )
    final_prompt_template_str = prompt_template_str
    if final_prompt_template_str is None:
        final_prompt_template_str = (
            "Analyze the sentiment of the following text and classify it as "
            "'positive', 'negative', or 'neutral'. Return only the single "
            "word classification for the sentiment. Text: {text}"
        )
    if "{text}" not in final_prompt_template_str:
        print("Error: Prompt template must include '{text}' placeholder.")
        results = [
            {
                "text": str(txt),
                "sentiment": "error",
                "raw_llm_output": "Prompt template error: missing {text}",
            }
            for txt in text_series
        ]
        return pd.DataFrame(results, columns=["text", "sentiment", "raw_llm_output"])

    tasks = []
    semaphore = (
        asyncio.Semaphore(concurrency_limit)
        if concurrency_limit and concurrency_limit > 0
        else None
    )

    async def task_wrapper(text_input):
        if semaphore:
            async with semaphore:
                return await _analyze_sentiment_single_async(
                    nlplot_instance,
                    text_input,
                    model,
                    final_prompt_template_str,
                    temperature,
                    current_use_cache,
                    litellm_kwargs,
                )
        else:
            return await _analyze_sentiment_single_async(
                nlplot_instance,
                text_input,
                model,
                final_prompt_template_str,
                temperature,
                current_use_cache,
                litellm_kwargs,
            )

    for text_input in text_series:
        tasks.append(task_wrapper(text_input))

    all_results_raw = await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    processed_results = []
    for i, res_or_exc in enumerate(all_results_raw):
        original_text = (
            str(text_series.iloc[i]) if pd.notna(text_series.iloc[i]) else ""
        )
        if isinstance(res_or_exc, Exception):
            print(f"Exception for text '{original_text[:30]}...': {res_or_exc}")
            processed_results.append(
                {
                    "text": original_text,
                    "sentiment": "error",
                    "raw_llm_output": str(res_or_exc),
                }
            )
        else:
            processed_results.append(res_or_exc)

    if nlplot_instance.cache:
        try:
            nlplot_instance.cache.close()
        except Exception as e:
            print(f"Warning: Error closing cache: {e}")

    return pd.DataFrame(
        processed_results, columns=["text", "sentiment", "raw_llm_output"]
    )
