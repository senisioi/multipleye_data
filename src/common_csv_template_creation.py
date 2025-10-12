# -*- coding: utf-8 -*-
"""MultiplEYE Common CSV template creation.ipynb
requirements:
pip install spacy pandas tqdm openpyxl

# for Turkish, we'd have to run a separate pipeline with a different spacy version
#! pip install "tr_core_news_lg @ https://huggingface.co/turkish-nlp-suite/tr_core_news_lg/resolve/main/tr_core_news_lg-1.0-py3-none-any.whl"


python -m spacy download ca_core_news_lg
python -m spacy download de_core_news_lg
python -m spacy download da_core_news_lg
python -m spacy download el_core_news_lg
python -m spacy download en_core_web_lg
python -m spacy download es_core_news_lg
python -m spacy download fr_core_news_lg
python -m spacy download hr_core_news_lg
python -m spacy download it_core_news_lg
python -m spacy download lt_core_news_lg
python -m spacy download mk_core_news_lg
python -m spacy download nl_core_news_lg
python -m spacy download pl_core_news_lg
python -m spacy download pt_core_news_lg
python -m spacy download ro_core_news_lg
python -m spacy download ru_core_news_lg
python -m spacy download sl_core_news_lg
python -m spacy download sv_core_news_lg
python -m spacy download uk_core_news_lg
python -m spacy download zh_core_web_lg

python -m spacy download xx_sent_ud_sm


"""

from language_constants import CODE2LANG, SPACY_LANGUAGES, LANGUAGES

import json
import spacy
import os
import re
import pandas as pd
from spacy.util import get_lang_class
from spacy.symbols import ORTH
from tqdm import tqdm
import pandas as pd
from collections import defaultdict


id2name = {1: "PopSci_MultiplEYE", 2: "Ins_HumanRights", 3: "Ins_LearningMobility", 4: "Lit_Alchemist", 6: "Lit_MagicMountain", 7: "Lit_NorthWind", 8: "Lit_Solaris", 9: "Lit_BrokenApril", 10: "Arg_PISACowsMilk", 11: "Arg_PISARapaNui", 12: "PopSci_Caveman", 13: "Enc_WikiMoon"}
name2id = {v:k for k,v in id2name.items()}


def xls_to_json(in_file, out_file):
    print('-'*60)
    print(in_file)
    out_dir = os.path.dirname(out_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df = pd.read_excel(in_file)
    stimuli = []
    for ridx, row in df.iterrows():
        pages = []
        for col in df.columns:
            if col.startswith("page_") and pd.notna(row[col]):
                pages.append(str(row[col]).strip())
        try:
            stimuli.append({
                "stimulus_id": name2id[row["stimulus_name"]],
                "stimulus_name": row["stimulus_name"],
                "stimulus_type": row["stimulus_type"],
                "pages": pages
            })
        except Exception as e:
            print(f"Error with row {ridx}: {e}")
    assert(len(stimuli) == len(id2name))
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stimuli, f, indent=2, ensure_ascii=False)
    print(f"Written {out_file}")


def load_all_json(lang_folder):
    all_data = {}
    for file in os.listdir(lang_folder):
        if file.endswith('.json'):
            lang_code = file.replace('.json', '').replace('multipleye_stimuli_experiment_', '')
            if lang_code == 'zd':
                lang_code = 'gsw'
            if lang_code not in LANGUAGES:
                continue
            with open(os.path.join(lang_folder, file), 'r', encoding='utf-8') as f:
                all_data[lang_code] = json.load(f)
    return all_data


def exists_spacy_blank(lang_code):
    try:
        get_lang_class(lang_code)
        return True
    except:
        return False


def load_spacy_model(lang_code, small=True):
    model = None
    if lang_code in SPACY_LANGUAGES:
        genre = 'news'
        if lang_code in {'zh', 'en'}:
            genre = 'web'
        if lang_code == 'rm':
            return ''
        model_name = f'{lang_code}_core_{genre}_{"sm" if small else "lg"}'
        print(f"Loading model {model_name} for {lang_code}")
        model = spacy.load(model_name)
        model.add_pipe("sentencizer")
    elif lang_code == "rm":
        model = spacy.load("it_core_news_lg")
        # keep 'morphologizer' ?
        model.disable_pipes('tok2vec', 'tagger', 'parser', 'lemmatizer', 'attribute_ruler', 'ner')
        model.add_pipe("sentencizer")
    elif lang_code == 'gsw':
        model = spacy.load('de_core_news_lg')
    elif exists_spacy_blank(lang_code):
        print(f"Loading model blank model for {lang_code}")
        model = spacy.blank(lang_code)
        model.add_pipe("sentencizer")
    else:
        model_name = f'xx_sent_ud_sm'
        print(f"Loading model {model_name} for {lang_code}")
        model = spacy.load(model_name)
        model.add_pipe("sentencizer")
    special_cases = {"eye-tracking": [{ORTH: "eye-tracking"},
                                      {ORTH: "Eye-tracking"},
                                      {ORTH: "Eye-Tracking"}]}
    for token, special_case in special_cases.items():
        model.tokenizer.add_special_case(token, special_case)
    return model


NLP_MODEL = None
CURRENT_LANG = ''


def get_nlp(lang_code, small=False):
    """To avoid loading all models at the same time
    """
    global NLP_MODEL, CURRENT_LANG
    if lang_code != CURRENT_LANG:
        try:
            print(f"Deleting model for {CURRENT_LANG}")
            del NLP_MODEL
        except:
            print("No model to delete")
        print(f"Loading model for {lang_code}")
        NLP_MODEL = load_spacy_model(lang_code, small=small)
        CURRENT_LANG = lang_code
    return NLP_MODEL


def feats_str(token):
    if not token.morph:
        return "_"
    md = token.morph.to_dict()
    if not md:
        return "_"
    bits = []
    for k in sorted(md):
        v = md[k]
        if isinstance(v, (list, tuple)):
            bits.append(f"{k}={','.join(v)}")
        else:
            bits.append(f"{k}={v}")
    return "|".join(bits) if bits else "_"


def get_head(token, sent):
    if token.head == token or token.dep_ == "ROOT":
        head = 0
        deprel = "root"
    else:
        head = (token.head.i - sent.start) + 1  # 1-based in sentence
        deprel = token.dep_.lower() if token.dep_ else "_"
    return head, deprel


def get_misc(token, include_ner=True):
    misc_parts = []
    if not token.whitespace_:
        misc_parts.append("SpaceAfter=No")
    if include_ner and token.ent_iob_ != "O":
        misc_parts.append(f"NER={token.ent_iob_}-{token.ent_type_}")
    misc = "|".join(misc_parts) if misc_parts else "_"
    return misc


def iter_pages(stimuli, nlp):
    for stim in stimuli:
        sid, sname = stim["stimulus_id"], stim["stimulus_name"]
        for pnum, page_text in enumerate(stim["pages"], start=1):
            yield sid, sname, pnum, nlp(page_text)


def stimuli2csv(stimuli, lang_code, level="page", small=False):
    rows = []
    nlp = get_nlp(lang_code, small=small)
    for sid, sname, page, doc in iter_pages(stimuli, nlp):
        ptext = doc.text
        document = nlp(ptext)
        for sent_idx, sentence in enumerate(document.sents):
            eos = {
              "language": CODE2LANG[lang_code],
              "language_code": lang_code,
              "stimulus_name": sname,
              "page": page,
              #"sent_idx": sent_idx+1,
              "token": "<eos>",
              "is_alpha": False,
              "is_stop": False,
              "is_punct": False,
              "lemma": "",
              "upos": "",
              "xpos": "",
              "feats": "",
              "head": "",
              "deprel": "",
              "deps": "",
              "misc": ""
              }
            for token in sentence:
                head, deprel = get_head(token, sentence)
                # for Romansh we use the Italian model and
                # don't set all the values
                if lang_code == 'rm':
                    rows.append(
                          {
                          "language": CODE2LANG[lang_code],
                          "language_code": lang_code,
                          "stimulus_name": sname,
                          "page": page,
                          "token": token.text,
                          "is_alpha": token.is_alpha,
                          "is_stop": False,
                          "is_punct": token.is_punct,
                          "lemma": "",
                          "upos": "",
                          "xpos": "",
                          "feats": "_",
                          "head": "0",
                          "deprel": "root",
                          "deps": "_",
                          "misc": get_misc(token, include_ner=False)
                      }
                    )
                else:
                    rows.append(
                        {
                            "language": CODE2LANG[lang_code],
                            "language_code": lang_code,
                            "stimulus_name": sname,
                            "page": page,
                            "token": token.text,
                            "is_alpha": token.is_alpha,
                            "is_stop": token.is_stop,
                            "is_punct": token.is_punct,
                            "lemma": token.lemma_,
                            "upos": token.pos_,
                            "xpos": token.tag_,
                            "feats": feats_str(token),
                            "head": head,
                            "deprel": deprel,
                            "deps": "_",
                            "misc": get_misc(token, include_ner=True)
                        }
                    )
            rows.append(eos)
    df = pd.DataFrame(rows).sort_values(by=["stimulus_name", "page"])
    df = pd.DataFrame(rows)
    return df


#! rm -rf languages_x*
#! wget https://github.com/senisioi/repository/releases/download/eyelanguages0/languages_xlsx.zip
#! unzip languages_xlsx.zip


IN_DIR = "languages_xlsx/"
LANG_FOLDER = "languages_json/"
os.makedirs(LANG_FOLDER, exist_ok=True)
OUT_DIR = 'csv'
os.makedirs(OUT_DIR, exist_ok=True)

for lang in os.listdir(IN_DIR):
    lang_file = os.path.join(IN_DIR, lang)
    out_file = os.path.join(LANG_FOLDER, f"{lang.replace('.xlsx', '')}.json")
    xls_to_json(lang_file, out_file)

all_data = load_all_json(LANG_FOLDER)
for k,v in all_data.items():
    if v:
        print(k, v[0][:10])


preproc = defaultdict(dict)
for lang_code, data in tqdm(all_data.items()):
    preproc[lang_code] = stimuli2csv(data, lang_code, small=False)


for lang_code, df in tqdm(preproc.items()):
    if lang_code == 'zd':
        lang_code = 'gsw'
    out_dir = os.path.join(OUT_DIR, lang_code)
    os.makedirs(out_dir, exist_ok=True)
    for stim_name, group in df.groupby('stimulus_name'):
        out_fis = os.path.join(out_dir, f'{stim_name}.csv')
        group['language_code'] = lang_code
        group.to_csv(out_fis, index=False)
        print(out_fis)
