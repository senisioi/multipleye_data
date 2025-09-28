import os
import pandas as pd
import numpy as np
from wordfreq import available_languages
from wordfreq import zipf_frequency
from transformers import AutoTokenizer


id2name = {
    1: "PopSci_MultiplEYE",
    2: "Ins_HumanRights",
    3: "Ins_LearningMobility",
    4: "Lit_Alchemist",
    6: "Lit_MagicMountain",
    7: "Lit_NorthWind",
    8: "Lit_Solaris",
    9: "Lit_BrokenApril",
    10: "Arg_PISACowsMilk",
    11: "Arg_PISARapaNui",
    12: "PopSci_Caveman",
    13: "Enc_WikiMoon",
}
name2id = {v: k for k, v in id2name.items()}


def iter_pages_df(df):
    for (sname, page_num), page_df in df.groupby(["stimulus_name", "page"]):
        yield sname, page_num, page_df


def basic_page_features(df):
    rows = []
    for stim_name, page_num, page_df in iter_pages_df(df):
        tokens = [tok.lower() for tok in page_df[page_df.is_alpha].token]
        types = set(tokens)
        ttr = len(types) / len(tokens) if tokens else 0
        num_punct = len(page_df[page_df.is_punct])
        num_stop = len(page_df[page_df.is_stop])
        num_sentences = sum(page_df.token == '<eos>')
        word_len = [len(tok) for tok in tokens]
        rows.append(
            {
                "stimulus_name": stim_name,
                "page": page_num,
                "num. words": len(tokens),
                "num. types": len(types),
                "type token ratio": ttr,
                "num. punct.": num_punct,
                "punct. ratio": num_punct / len(tokens) if len(tokens) else 0,
                "num. function words": num_stop,
                "function words ratio": num_stop / len(tokens) if len(tokens) else 0,
                "num. sentences": num_sentences,
                "word length": sum(word_len) / len(word_len) if len(word_len) else 0,
            }
        )
    df = pd.DataFrame(rows).sort_values(by=["stimulus_name", "page"])
    df.set_index(["stimulus_name", "page"], inplace=True)
    return df


def POS_count(df, POS="PRON"):
    rows = []
    for stim_name, page_num, page_df in iter_pages_df(df):
        num_pos = sum(page_df.upos == POS)
        tokens = [tok.lower() for tok in page_df[page_df.is_alpha].token]
        rows.append(
            {
                "stimulus_name": stim_name,
                "page": page_num,
                f"num. {POS}": num_pos,
                f"{POS} ratio": num_pos / len(tokens) if len(tokens) else 0,
            }
        )
    df = pd.DataFrame(rows).sort_values(by=["stimulus_name", "page"])
    df.set_index(["stimulus_name", "page"], inplace=True)
    return df


def zipf_freq(df):
    rows = []
    lang_code = df.language_code.unique()[0]
    for stim_name, page_num, page_df in iter_pages_df(df):
        # no lowercasing
        tokens = [tok for tok in page_df[page_df.is_alpha].token]
        frequencies = [0]
        if lang_code in available_languages(wordlist="best") or lang_code in {
            "sh",
            "hr",
            "sr",
            "bs",
        }:
            frequencies = [zipf_frequency(token, lang_code) for token in tokens]
        rows.append(
            {
                "stimulus_name": stim_name,
                "page": page_num,
                "mean Zipf freq.": np.mean(frequencies),
                "median Zipf freq.": np.median(frequencies),
                "std Zipf freq.": np.std(frequencies),
            }
        )
    df = pd.DataFrame(rows).sort_values(by=["stimulus_name", "page"])
    df.set_index(["stimulus_name", "page"], inplace=True)
    return df


def fertility(df, tokenizer):
    rows = []
    for stim_name, page_num, page_df in iter_pages_df(df):
        # no lowercase
        tokens = [tok for tok in page_df[page_df.is_alpha].token]
        llm_tokens = 0
        for token in tokens:
            llm_tokens += len(tokenizer.encode(token, add_special_tokens=False))
        rows.append(
            {
                "stimulus_name": stim_name,
                "page": page_num,
                f"fertility {os.path.basename(tokenizer.name_or_path)}": llm_tokens
                / len(tokens) if len(tokens) else 0,
            }
        )
    df = pd.DataFrame(rows).sort_values(by=["stimulus_name", "page"])
    df.set_index(["stimulus_name", "page"], inplace=True)
    return df


def featurize(df):
    tok_name = "swiss-ai/Apertus-70B-2509"
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    return pd.concat(
        [
            basic_page_features(df),
            zipf_freq(df),
            fertility(df, tokenizer),
            POS_count(df, "PRON"),
            POS_count(df, "NOUN"),
            POS_count(df, "VERB"),
        ],
        axis=1,
    )
