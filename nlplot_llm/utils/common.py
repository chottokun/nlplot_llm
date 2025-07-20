import itertools
from collections import defaultdict
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def _ranked_topics_for_edges(batch_list: list) -> list:
    return sorted(list(map(str, batch_list)))


def _unique_combinations_for_edges(batch_list: list) -> list:
    return list(
        itertools.combinations(_ranked_topics_for_edges(batch_list), 2)
    )


def _add_unique_combinations_to_dict(
    unique_combs: list, combo_dict: dict
) -> dict:
    for combination in unique_combs:
        combo_dict[combination] = combo_dict.get(combination, 0) + 1
    return combo_dict


def get_colorpalette(colorpalette: str, n_colors: int) -> list:
    if not isinstance(n_colors, int) or n_colors <= 0:
        raise ValueError("n_colors must be a positive integer")
    palette = sns.color_palette(colorpalette, n_colors)
    return [
        f'rgb({",".join(str(int(x * 255)) for x in rgb_val)})'
        for rgb_val in palette
    ]


def generate_freq_df(
    value: pd.Series,
    n_gram: int = 1,
    top_n: int = 50,
    stopwords: list = [],
    verbose: bool = True,
) -> pd.DataFrame:
    if not isinstance(n_gram, int) or n_gram <= 0:
        raise ValueError("n_gram must be a positive integer")
    if not isinstance(top_n, int) or top_n < 0:
        raise ValueError("top_n must be a non-negative integer")

    def generate_ngrams(text: str, n_gram_val: int = 1) -> list:
        token = [
            t
            for t in str(text).lower().split(" ")
            if t and t not in stopwords
        ]
        ngrams_zip = zip(*[token[i:] for i in range(n_gram_val)])
        return [" ".join(ngram) for ngram in ngrams_zip]

    freq_dict = defaultdict(int)
    iterable_value = tqdm(value) if verbose else value
    for sent in iterable_value:
        for word in generate_ngrams(sent, n_gram):
            freq_dict[word] += 1
    output_df = pd.DataFrame(
        sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    )
    if not output_df.empty:
        output_df.columns = ["word", "word_count"]
    else:
        output_df = pd.DataFrame(columns=["word", "word_count"])
    return output_df.head(top_n)
