import asyncio
import pandas as pd
from typing import Optional, List, Any

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


async def _categorize_text_single_async(
    nlplot_instance,
    text_to_categorize: str,
    categories: List[str],
    category_list_str: str,
    model: str,
    prompt_template_str: str,
    multi_label: bool,
    temperature: float,
    current_use_cache: bool,
    litellm_kwargs: dict,
) -> dict:
    original_text_for_df = (
        str(text_to_categorize) if pd.notna(text_to_categorize) else ""
    )
    category_col_name = "categories" if multi_label else "category"
    parsed_categories_result: Any = [] if multi_label else "unknown"
    raw_llm_response_content = ""

    if not original_text_for_df.strip():
        raw_llm_response_content = "Input text was empty or whitespace."
    else:
        cacheable_kwargs_items = sorted(
            (k, v)
            for k, v in litellm_kwargs.items()
            if k in ("temperature", "max_tokens", "top_p")
        )
        cache_key_tuple = (
            "categorize_text_llm",
            model,
            prompt_template_str,
            original_text_for_df,
            tuple(sorted(categories)),
            multi_label,
            tuple(cacheable_kwargs_items),
        )

        cached_result = None
        if current_use_cache and nlplot_instance.cache:
            try:
                cached_result = nlplot_instance.cache.get(cache_key_tuple)
            except Exception as e:
                print(
                    "Warning: Async cache get failed for categorize. "
                    f"Error: {e}"
                )

        if cached_result is not None:
            parsed_categories_result = cached_result["categories_result"]
            raw_llm_response_content = cached_result["raw_llm_output"]
        else:
            try:
                format_args = {"text": original_text_for_df}
                if "{categories}" in prompt_template_str:
                    format_args["categories"] = category_list_str
                formatted_content_for_llm = prompt_template_str.format(
                    **format_args
                )
                messages = [
                    {"role": "user", "content": formatted_content_for_llm}
                ]

                completion_kwargs = {**litellm_kwargs}
                if "temperature" not in completion_kwargs:
                    completion_kwargs["temperature"] = temperature

                response = await litellm.acompletion(
                    model=model, messages=messages, **completion_kwargs
                )

                if response.choices and response.choices[0].message:
                    raw_llm_response_content = (
                        response.choices[0].message.content or ""
                    )
                else:
                    raw_llm_response_content = str(response)

                processed_output = raw_llm_response_content.strip().lower()
                if multi_label:
                    found_cats_raw = [
                        cat.strip() for cat in processed_output.split(",")
                    ]
                    parsed_categories_result = [
                        og_cat
                        for og_cat in categories
                        if og_cat.lower() in [
                            fcr.lower()
                            for fcr in found_cats_raw
                            if fcr and fcr != "none"
                        ]
                    ]
                else:
                    parsed_categories_result = "unknown"
                    exact_match_found = False
                    for cat_og in categories:
                        if cat_og.lower() == processed_output:
                            parsed_categories_result = cat_og
                            exact_match_found = True
                            break
                    if not exact_match_found:
                        for cat_og in categories:
                            if cat_og.lower() in processed_output:
                                parsed_categories_result = cat_og
                                break
                    if (
                        parsed_categories_result == "unknown" and
                        processed_output in ["unknown", "none"]
                    ):
                        pass

                if current_use_cache and nlplot_instance.cache:
                    try:
                        nlplot_instance.cache.set(
                            cache_key_tuple,
                            {
                                "categories_result": parsed_categories_result,
                                "raw_llm_output": raw_llm_response_content,
                            },
                        )
                    except Exception as e:
                        print(
                            "Warning: Async cache set failed for "
                            f"categorize. Error: {e}"
                        )
            except Exception as e:
                print(
                    "Error in _categorize_text_single_async for "
                    f"'{original_text_for_df[:30]}...': {e}"
                )
                parsed_categories_result = [] if multi_label else "error"
                raw_llm_response_content = str(e)

    return {
        "text": original_text_for_df,
        category_col_name: parsed_categories_result,
        "raw_llm_output": raw_llm_response_content,
        "_multi_label": multi_label,
    }


async def categorize_text_llm_async(
    nlplot_instance,
    text_series: pd.Series,
    categories: List[str],
    model: str,
    prompt_template_str: Optional[str] = None,
    multi_label: bool = False,
    temperature: float = 0.0,
    use_cache: Optional[bool] = None,
    concurrency_limit: Optional[int] = None,
    return_exceptions: bool = True,
    **litellm_kwargs,
) -> pd.DataFrame:
    category_col_name = "categories" if multi_label else "category"
    default_columns = ["text", category_col_name, "raw_llm_output"]

    if not LITELLM_AVAILABLE:
        print("Warning: LiteLLM is not available for async categorization.")
        return pd.DataFrame(columns=default_columns)
    if not isinstance(text_series, pd.Series):
        print("Warning: Input 'text_series' must be a pandas Series.")
        return pd.DataFrame(columns=default_columns)
    if text_series.empty:
        return pd.DataFrame(columns=default_columns)
    if not (
        categories and
        isinstance(categories, list) and
        all(isinstance(c, str) and c for c in categories)
    ):
        raise ValueError(
            "Categories must be a non-empty list of non-empty strings."
        )

    current_use_cache = (
        use_cache
        if use_cache is not None
        else nlplot_instance.use_cache_default
    )
    category_list_str_for_prompt = ", ".join(f"'{c}'" for c in categories)

    final_prompt_template_str = prompt_template_str
    if final_prompt_template_str is None:
        if multi_label:
            final_prompt_template_str = (
                "Analyze the following text and classify it into one or more "
                "of these categories: "
                f"{category_list_str_for_prompt}. Return a comma-separated "
                "list of matching category names. If no categories match, "
                "return 'none'. Text: {text}"
            )
        else:
            final_prompt_template_str = (
                "Analyze the following text and classify it into exactly "
                "one of these categories: "
                f"{category_list_str_for_prompt}. Return only the single "
                "matching category name. If no categories match, "
                "return 'unknown'. Text: {text}"
            )

    if "{text}" not in final_prompt_template_str:
        results = [
            {
                "text": str(txt),
                category_col_name: [] if multi_label else "error",
                "raw_llm_output": "Prompt template error: missing {text}",
            }
            for txt in text_series
        ]
        return pd.DataFrame(results, columns=default_columns)

    tasks = []
    semaphore = (
        asyncio.Semaphore(concurrency_limit)
        if concurrency_limit and concurrency_limit > 0
        else None
    )

    async def task_wrapper(text_input):
        if semaphore:
            async with semaphore:
                return await _categorize_text_single_async(
                    nlplot_instance,
                    text_input,
                    categories,
                    category_list_str_for_prompt,
                    model,
                    final_prompt_template_str,
                    multi_label,
                    temperature,
                    current_use_cache,
                    litellm_kwargs,
                )
        else:
            return await _categorize_text_single_async(
                nlplot_instance,
                text_input,
                categories,
                category_list_str_for_prompt,
                model,
                final_prompt_template_str,
                multi_label,
                temperature,
                current_use_cache,
                litellm_kwargs,
            )

    for text_input in text_series:
        tasks.append(task_wrapper(text_input))

    all_results_raw = await asyncio.gather(
        *tasks, return_exceptions=return_exceptions
    )

    processed_results = []
    for i, res_or_exc in enumerate(all_results_raw):
        original_text = (
            str(text_series.iloc[i]) if pd.notna(text_series.iloc[i]) else ""
        )
        if isinstance(res_or_exc, Exception):
            processed_results.append(
                {
                    "text": original_text,
                    category_col_name: [] if multi_label else "error",
                    "raw_llm_output": str(res_or_exc),
                }
            )
        else:
            res_or_exc.pop("_multi_label", None)
            processed_results.append(res_or_exc)

    if nlplot_instance.cache:
        try:
            nlplot_instance.cache.close()
        except Exception as e:
            print(f"Warning: Error closing cache: {e}")

    return pd.DataFrame(processed_results, columns=default_columns)


def categorize_text_llm(
    nlplot_instance,
    text_series: pd.Series,
    categories: List[str],
    model: str,
    prompt_template_str: Optional[str] = None,
    multi_label: bool = False,
    temperature: float = 0.0,
    use_cache: Optional[bool] = None,
    **litellm_kwargs,
) -> pd.DataFrame:
    category_col_name = "categories" if multi_label else "category"
    default_columns = ["text", category_col_name, "raw_llm_output"]

    if not LITELLM_AVAILABLE:
        print(
            "Warning: LiteLLM is not installed. LLM-based categorization "
            "is not available."
        )
        return pd.DataFrame(columns=default_columns)

    if not isinstance(text_series, pd.Series):
        print("Warning: Input 'text_series' must be a pandas Series.")
        return pd.DataFrame(columns=default_columns)
    if text_series.empty:
        return pd.DataFrame(columns=default_columns)
    if not (
        categories and
        isinstance(categories, list) and
        all(isinstance(c, str) and c for c in categories)
    ):
        raise ValueError(
            "Categories must be a non-empty list of non-empty strings."
        )

    category_list_str = ", ".join(f"'{c}'" for c in categories)

    final_prompt_template_str = prompt_template_str
    if final_prompt_template_str is None:
        if multi_label:
            final_prompt_template_str = (
                "Analyze the following text and classify it into one or more "
                f"of these categories: {category_list_str}. Return a "
                "comma-separated list of the matching category names. If no "
                "categories match, return 'none'. Text: {text}"
            )
        else:
            final_prompt_template_str = (
                "Analyze the following text and classify it into exactly one "
                f"of these categories: {category_list_str}. Return only the "
                "single matching category name. If no categories match, "
                "return 'unknown'. Text: {text}"
            )

    if "{text}" not in final_prompt_template_str:
        print("Error: Prompt template must include '{text}' placeholder.")
        return pd.DataFrame(
            [
                {
                    "text": str(txt) if pd.notna(txt) else "",
                    category_col_name: [] if multi_label else "error",
                    "raw_llm_output": "Prompt template error: missing {text}",
                }
                for txt in text_series
            ],
            columns=default_columns,
        )
    if (
        "{categories}" in final_prompt_template_str and
        not category_list_str
    ):
        print(
            "Error: Prompt template includes '{categories}' but no "
            "categories were provided effectively."
        )
        return pd.DataFrame(
            [
                {
                    "text": str(txt) if pd.notna(txt) else "",
                    category_col_name: [] if multi_label else "error",
                    "raw_llm_output": (
                        "Prompt template error: {categories} placeholder "
                        "used with empty category list."
                    ),
                }
                for txt in text_series
            ],
            columns=default_columns,
        )

    results = []
    for text_input in text_series:
        original_text_for_df = str(text_input) if pd.notna(text_input) else ""
        raw_llm_response_content = ""
        parsed_categories: Any = [] if multi_label else "unknown"

        if not original_text_for_df.strip():
            raw_llm_response_content = "Input text was empty or whitespace."
        else:
            cacheable_kwargs_items = sorted(
                (k, v)
                for k, v in litellm_kwargs.items()
                if k in ("temperature", "max_tokens", "top_p")
            )
            cache_key_tuple = (
                "categorize_text_llm",
                model,
                final_prompt_template_str,
                original_text_for_df,
                tuple(sorted(categories)),
                multi_label,
                tuple(cacheable_kwargs_items),
            )

            cached_result = None
            current_use_cache_for_call = (
                use_cache
                if use_cache is not None
                else nlplot_instance.use_cache_default
            )
            if current_use_cache_for_call and nlplot_instance.cache:
                try:
                    cached_result = nlplot_instance.cache.get(
                        cache_key_tuple
                    )
                except Exception as e:
                    print(
                        "Warning: Cache get failed for key "
                        f"{cache_key_tuple}. Error: {e}"
                    )

            if cached_result is not None:
                parsed_categories = cached_result["categories_result"]
                raw_llm_response_content = cached_result["raw_llm_output"]
            else:
                try:
                    prompt_to_format = final_prompt_template_str
                    format_args = {"text": original_text_for_df}
                    if "{categories}" in prompt_to_format:
                        format_args["categories"] = category_list_str
                    formatted_content_for_llm = prompt_to_format.format(
                        **format_args
                    )
                    messages = [
                        {"role": "user", "content": formatted_content_for_llm}
                    ]

                    completion_kwargs = {
                        k: v for k, v in litellm_kwargs.items()
                    }
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
                    if multi_label:
                        found_cats_raw = [
                            cat.strip()
                            for cat in processed_output.split(",")
                        ]
                        parsed_categories = [
                            original_cat
                            for original_cat in categories
                            if original_cat.lower() in [
                                fcr.lower()
                                for fcr in found_cats_raw
                                if fcr and fcr != "none"
                            ]
                        ]
                    else:
                        parsed_categories = "unknown"
                        exact_match_found = False
                        for cat_original_case in categories:
                            if (
                                cat_original_case.lower() ==
                                processed_output
                            ):
                                parsed_categories = cat_original_case
                                exact_match_found = True
                                break
                        if not exact_match_found:
                            for cat_original_case in categories:
                                if (
                                    cat_original_case.lower()
                                    in processed_output
                                ):
                                    parsed_categories = cat_original_case
                                    break
                        if (
                            parsed_categories == "unknown" and
                            processed_output in ["unknown", "none"]
                        ):
                            pass

                    if current_use_cache_for_call and nlplot_instance.cache:
                        try:
                            nlplot_instance.cache.set(
                                cache_key_tuple,
                                {
                                    "categories_result": parsed_categories,
                                    "raw_llm_output": raw_llm_response_content,
                                },
                            )
                        except Exception as e:
                            print(
                                "Warning: Cache set failed for key "
                                f"{cache_key_tuple}. Error: {e}"
                            )

                except ImportError:
                    parsed_categories = [] if multi_label else "error"
                    raw_llm_response_content = "LiteLLM not installed."
                    print(
                        "LiteLLM not installed. Cannot perform "
                        "categorization."
                    )
                except litellm.exceptions.AuthenticationError as e:
                    parsed_categories = [] if multi_label else "error"
                    raw_llm_response_content = f"AuthenticationError: {e}"
                    print(f"LiteLLM Authentication Error: {e}")
                except litellm.exceptions.APIConnectionError as e:
                    parsed_categories = [] if multi_label else "error"
                    raw_llm_response_content = f"APIConnectionError: {e}"
                    print(f"LiteLLM API Connection Error: {e}")
                except litellm.exceptions.RateLimitError as e:
                    parsed_categories = [] if multi_label else "error"
                    raw_llm_response_content = f"RateLimitError: {e}"
                    print(f"LiteLLM Rate Limit Error: {e}")
                except Exception as e:
                    parsed_categories = [] if multi_label else "error"
                    raw_llm_response_content = str(e)
                    print(
                        "Error categorizing text "
                        f"'{original_text_for_df[:50]}...': {e}"
                    )

        results.append(
            {
                "text": original_text_for_df,
                category_col_name: parsed_categories,
                "raw_llm_output": raw_llm_response_content,
            }
        )

    if nlplot_instance.cache:
        try:
            nlplot_instance.cache.close()
        except Exception as e:
            print(f"Warning: Error closing cache: {e}")

    return pd.DataFrame(results, columns=default_columns)
