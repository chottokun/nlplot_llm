import pandas as pd
from typing import Optional, List, Dict, Any

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from janome.tokenizer import Tokenizer as JanomeTokenizer
    JANOME_AVAILABLE = True
except ImportError:
    JANOME_AVAILABLE = False
    class JanomeTokenizer:
        def __init__(self, wakati=False): pass
        def tokenize(self, text: str): return []

def get_tfidf_top_features(
    nlplot_instance,
    text_series: pd.Series,
    language: str = "english",
    n_features: int = 10,
    custom_stopwords: Optional[List[str]] = None,
    use_janome_tokenizer_for_japanese: bool = True,
    tfidf_ngram_range: tuple = (1, 1),
    tfidf_max_df: float = 1.0,
    tfidf_min_df: int = 1,
    return_type: str = "overall"
) -> pd.DataFrame:
    if not SKLEARN_AVAILABLE:
        print("Error: scikit-learn is not installed. TF-IDF functionality requires scikit-learn.")
        if return_type == "overall":
            return pd.DataFrame(columns=['word', 'tfidf_score'])
        else:
            return pd.DataFrame(columns=['document_id', 'word', 'tfidf_score'])

    if not isinstance(text_series, pd.Series):
        print("Warning: text_series is not a pandas Series. Returning empty DataFrame.")
        return pd.DataFrame()

    if text_series.empty:
        print("Warning: text_series is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    processed_corpus: List[str] = []
    janome_tokenizer_for_tfidf = None
    if language == "japanese" and use_janome_tokenizer_for_japanese and JANOME_AVAILABLE:
        try:
            janome_tokenizer_for_tfidf = JanomeTokenizer(wakati=True)
        except Exception as e:
            print(f"Warning: Failed to initialize Janome Tokenizer for TF-IDF (wakati=True): {e}. Will fallback.")

    for i, text_content in enumerate(text_series):
        if not isinstance(text_content, str) or not text_content.strip():
            processed_corpus.append("")
            continue
        if janome_tokenizer_for_tfidf:
            try:
                tokens = list(janome_tokenizer_for_tfidf.tokenize(text_content))
                processed_corpus.append(" ".join(tokens))
            except Exception as e_tok:
                print(f"Error during Janome tokenization for document {i}: {e_tok}. Using raw text.")
                processed_corpus.append(text_content)
        else:
            processed_corpus.append(text_content)

    final_stopwords: Optional[List[str]] = None
    temp_stopwords_list: List[str] = []
    if language == "english":
        temp_stopwords_list.extend(list(ENGLISH_STOP_WORDS))
    if hasattr(nlplot_instance, 'default_stopwords') and nlplot_instance.default_stopwords:
        temp_stopwords_list.extend(nlplot_instance.default_stopwords)
    if custom_stopwords:
        if isinstance(custom_stopwords, list):
            temp_stopwords_list.extend(custom_stopwords)
        else:
            print("Warning: custom_stopwords should be a list of strings. It will be ignored.")
    if temp_stopwords_list:
        final_stopwords = sorted(list(set(temp_stopwords_list)))

    vectorizer = TfidfVectorizer(
        stop_words=final_stopwords,
        ngram_range=tfidf_ngram_range,
        max_df=tfidf_max_df,
        min_df=tfidf_min_df,
    )

    try:
        if not any(processed_corpus):
            print("Warning: Corpus for TF-IDF is empty after processing. Returning empty DataFrame.")
            if return_type == "overall": return pd.DataFrame(columns=['word', 'tfidf_score'])
            else: return pd.DataFrame(columns=['document_id', 'word', 'tfidf_score'])
        tfidf_matrix = vectorizer.fit_transform(processed_corpus)
        feature_names = vectorizer.get_feature_names_out()
    except ValueError as ve:
        print(f"ValueError during TF-IDF vectorization: {ve}.")
        if return_type == "overall": return pd.DataFrame(columns=['word', 'tfidf_score'])
        else: return pd.DataFrame(columns=['document_id', 'word', 'tfidf_score'])

    if return_type == "overall":
        if tfidf_matrix.shape[1] == 0:
            return pd.DataFrame(columns=['word', 'tfidf_score'])
        sum_tfidf = tfidf_matrix.sum(axis=0)
        sum_tfidf_array = sum_tfidf.A1 if hasattr(sum_tfidf, 'A1') else sum_tfidf.flatten()
        feature_scores = list(zip(feature_names, sum_tfidf_array))
        sorted_features = sorted(feature_scores, key=lambda x: x[1], reverse=True)
        top_n = sorted_features[:n_features]
        return pd.DataFrame(top_n, columns=['word', 'tfidf_score'])
    elif return_type == "per_document":
        if tfidf_matrix.shape[1] == 0:
            return pd.DataFrame(columns=['document_id', 'word', 'tfidf_score'])
        results_data = []
        for i in range(tfidf_matrix.shape[0]):
            doc_vector = tfidf_matrix[i, :]
            non_zero_indices = doc_vector.nonzero()[1]
            doc_feature_scores = [(feature_names[idx], doc_vector[0, idx]) for idx in non_zero_indices]
            sorted_doc_features = sorted(doc_feature_scores, key=lambda x: x[1], reverse=True)
            top_n_doc = sorted_doc_features[:n_features]
            doc_id = text_series.index[i] if text_series.index.is_unique and not isinstance(text_series.index, pd.RangeIndex) else i
            for word, score in top_n_doc:
                results_data.append({'document_id': doc_id, 'word': word, 'tfidf_score': score})
        return pd.DataFrame(results_data, columns=['document_id', 'word', 'tfidf_score'])
    else:
        print(f"Error: Invalid return_type '{return_type}' encountered.")
        return pd.DataFrame()

def get_kwic_results(
    nlplot_instance,
    text_series: pd.Series,
    keyword: str,
    language: str = "english",
    window_size: int = 5,
    use_janome_tokenizer_for_japanese: bool = True,
    ignore_case: bool = True
) -> List[Dict[str, Any]]:
    if not isinstance(text_series, pd.Series) or not isinstance(keyword, str):
        print("Warning: Invalid input types for KWIC. text_series must be a Series and keyword a string.")
        return []
    if text_series.empty or not keyword.strip():
        print("Warning: text_series is empty or keyword is blank. Returning empty list for KWIC.")
        return []

    all_tokenized_documents: List[List[str]] = []
    janome_tokenizer_for_kwic = None
    if language == "japanese" and use_janome_tokenizer_for_japanese and JANOME_AVAILABLE:
        try:
            janome_tokenizer_for_kwic = JanomeTokenizer(wakati=True)
        except Exception as e:
            print(f"Warning: Failed to initialize Janome Tokenizer for KWIC (wakati=True): {e}. Will fallback to space splitting.")

    for i, text_content in enumerate(text_series):
        if not isinstance(text_content, str) or not text_content.strip():
            all_tokenized_documents.append([])
            continue
        if janome_tokenizer_for_kwic:
            try:
                tokens = list(janome_tokenizer_for_kwic.tokenize(text_content))
                all_tokenized_documents.append(tokens)
            except Exception as e_tok:
                print(f"Error during Janome tokenization for KWIC, document {i}: {e_tok}. Using space splitting as fallback.")
                all_tokenized_documents.append(text_content.split())
        else:
            all_tokenized_documents.append(text_content.split())

    raw_kwic_tuples: List[tuple] = []
    search_keyword_processed = keyword.lower() if ignore_case else keyword
    for doc_idx, tokens in enumerate(all_tokenized_documents):
        if not tokens:
            continue
        for i, token in enumerate(tokens):
            current_token_processed = token.lower() if ignore_case else token
            if current_token_processed == search_keyword_processed:
                left_start_idx = max(0, i - window_size)
                left_tokens = tokens[left_start_idx:i]
                keyword_match_original_case = tokens[i]
                right_end_idx = min(len(tokens), i + 1 + window_size)
                right_tokens = tokens[i+1:right_end_idx]
                original_doc_id = text_series.index[doc_idx] if text_series.index.is_unique and not isinstance(text_series.index, pd.RangeIndex) else doc_idx
                raw_kwic_tuples.append(
                    (original_doc_id, left_tokens, keyword_match_original_case, right_tokens)
                )

    final_kwic_results: List[Dict[str, Any]] = []
    for doc_id_val, left_tokens_val, keyword_match_val, right_tokens_val in raw_kwic_tuples:
        final_kwic_results.append({
            "document_id": doc_id_val,
            "left_context": " ".join(left_tokens_val),
            "keyword_match": keyword_match_val,
            "right_context": " ".join(right_tokens_val)
        })
    return final_kwic_results
