import os
import sys
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

os.makedirs("html/img", exist_ok=True)

from language_constants import (LANG_ORDER,
                                LANGUAGES,
                                CODE2LANG,
                                LANG_COLORS
                                )
from features import featurize


IN_DIR='data/processed/'


#columns = ['language', 'language_code', 'stimulus_name', 'page', 'sent_idx', 'token', 'is_alpha', 'is_stop', 'is_punct', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc']
#dtypes = [str, str, str, int, int, str, bool, bool, bool, str, str, str, str, str, str, str, str]
#na_values = {'language': '', 'language_code': '', 'stimulus_name': '', 'page': -1, 'sent_idx': -1, 'token': '', 'is_alpha': False, 'is_stop': False, 'is_punct': False, 'lemma': '', 'upos': '', 'xpos': '', 'feats': '', 'head': -1, 'deprel': '', 'deps': '', 'misc': ''}

language_data = {}
for lang_dir in os.listdir(IN_DIR):
    dir_path = os.path.join(IN_DIR, lang_dir)
    if os.path.isdir(dir_path):
        language_csvs = []
        for csv_file in os.listdir(dir_path):
            if csv_file.endswith('.csv'):
                current_df = pd.read_csv(os.path.join(dir_path, csv_file),
                                         keep_default_na=False)
                language_csvs.append(current_df)
        if language_csvs:
            language_data[lang_dir] = pd.concat(language_csvs, axis=0)



LEVEL = sys.argv[1] if len(sys.argv) > 1 else 'page'
# page, stimulus, language

out = {}
limit = 100
for lang_code, _ in tqdm(language_data.items()):
    out[lang_code] = featurize(language_data[lang_code], LEVEL)
    if limit <= 0:
        break
    limit -= 1


FTR = out['yue'].columns.values.tolist()#[:10]
if LEVEL == 'page':
    elems = []
    for k,o in out.items():
        o = o.mean()
        o['stimulus_name'] = k
        o['page'] = -1
        elems.append(o)
    df = pd.concat(elems, axis=1).T
    df.set_index(['stimulus_name', 'page'], inplace=True)
else:
    df = pd.concat(out.values())

df = df[FTR].astype(float)
df.reset_index(inplace=True)
df.drop(columns=['page'], inplace=True)
df.index = df.stimulus_name
df = df.loc[[l for l in LANG_ORDER if l in df.index]]
df.stimulus_name = df.stimulus_name.apply(lambda x:CODE2LANG[x])
df['color'] = [LANG_COLORS[x] for x in df.index]

df = df.round(2)
df.to_csv(f'stats_{LEVEL}.csv', index=False)

print(1)
