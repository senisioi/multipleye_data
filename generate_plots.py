import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from language_constants import LANG_ORDER
from features import featurize
from plot_helpers import (make_combined_figure,
                          make_wide_figure,
                          make_intermediate_features_for_plot,
                          make_single_figure,
                          make_language_label,
                          )


IN_DIR='data/'


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




out = {}
for lang_code, _ in tqdm(language_data.items()):
    out[lang_code] = featurize(language_data[lang_code])




feature = 'function words ratio'
level='page'
#make_combined_figure(out, feature = feature, out_dir='./img', show=False)


# make an index html that links to each html figure
links = {}

for feature in tqdm(out['ro'].columns):
    links[feature] = make_combined_figure(out, feature = feature, out_dir='./img', show=False)
    make_wide_figure(out, feature, level=level, out_dir='./img',  xtitle='', show=False, show_error_y=False)
    #break
    lang_df, doc_dfs, _ = make_intermediate_features_for_plot(out, feature, level)
    make_single_figure(lang_df, feature, level=level, out_dir='./img', show=False, xtitle='All documents')
    for k, docdf in doc_dfs.items():
        make_single_figure(docdf, feature, level=level, out_dir='./img', show=False, xtitle=k, show_error_y=False)



html_csv_out = 'img/data'
html_lang_paths = defaultdict(list)

for lang_dir in os.listdir(IN_DIR):
    dir_path = os.path.join(IN_DIR, lang_dir)
    if os.path.isdir(dir_path):
        language = lang_dir
        out_dir = os.path.join(html_csv_out, language)
        os.makedirs(out_dir, exist_ok=True)
        for csv_file in os.listdir(dir_path):
            if csv_file.endswith('.csv'):
                out_fis = csv_file.replace('.csv', '.html')
                out_fis = os.path.join(out_dir, out_fis)
                html_lang_paths[language].append(out_fis)
                current_df = pd.read_csv(os.path.join(dir_path, csv_file),
                                         keep_default_na=False)                
                current_df.to_html(out_fis)


csv_tables_html = "<h1>Processed Data</h1>\n"
for language in LANG_ORDER:
    files = html_lang_paths[language]
    csv_tables_html += f"<h2>{make_language_label(language)}</h2>\n<ul>\n"
    for html_file in files:
        file_name = os.path.basename(html_file)
        csv_tables_html += f'<li><a href="{html_file}">{file_name}</a></li>\n'
    csv_tables_html += "</ul>\n"

with open('csv_tables.html', 'w') as f:
    f.write(csv_tables_html)


html = "<h1>MultiplEYE Corpus Statistics per page.</h1>\n"
html += '<p><a href="csv_tables.html">Preprocessed Data</a></p>\n'
html += "<h2>Results:</h1>\n"
for text, link in links.items():
    html += f'<p><a href="{link}">{text}</a></p>\n'

with open('index.html', 'w') as f:
    f.write(html)
