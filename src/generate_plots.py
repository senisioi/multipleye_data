import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

os.makedirs("html/img", exist_ok=True)

from language_constants import (LANG_ORDER,
                                LANGUAGES,
                                )
from features import featurize
from plot_helpers import (make_combined_figure,
                          make_wide_figure,
                          make_intermediate_features_for_plot,
                          make_single_figure,
                          make_language_label,
                          make_line_plots,
                          )


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


LEVEL = 'page'  # page, stimulus, language
out = {}
limit = 100
for lang_code, _ in tqdm(language_data.items()):
    out[lang_code] = featurize(language_data[lang_code], LEVEL)
    if limit <= 0:
        break
    limit -= 1

html_csv_ftr_out = 'html/img/features'
html_lang_ftr_paths = defaultdict(list)

for lang_code, dataframe in out.items():
    out_dir = os.path.join(html_csv_ftr_out, lang_code)
    os.makedirs(out_dir, exist_ok=True)
    out_fis = os.path.join(out_dir, f"{lang_code}.html")
    dataframe.to_html(out_fis)
    dataframe.to_csv(out_fis.replace('.html', '.csv'), index=False)
    html_lang_ftr_paths[lang_code].append(
        os.path.relpath(out_fis, start='html'))

feature_tables_html = "<h1>Feature Data</h1>\n"
for language, paths in html_lang_ftr_paths.items():
    for path in paths:
        feature_tables_html += f'<a name="{make_language_label(language)}"></a>\n'
        feature_tables_html += f'<h3><a href="{path}">{make_language_label(language)}</a></h3>\n'

with open('html/feature_tables.html', 'w') as f:
    f.write(feature_tables_html)

feature = 'function words ratio'

#make_combined_figure(out, feature = feature, out_dir='./html/img', show=False)


# make an index html that links to each html figure
links = {}
text_wise_links = {}
#import sys

for feature in tqdm(out['ro'].columns):
    p_comb = make_combined_figure(out, feature=feature, level=LEVEL, out_dir='./html/img', show=False)
    links[feature] = os.path.relpath(p_comb, start='html')

    p_txt = make_line_plots(out, feature, level=LEVEL, out_dir='./html/img', xtitle='', show=False, show_error_y=False)
    text_wise_links[feature] = os.path.relpath(p_txt, start='html')
    lang_df, doc_dfs, _ = make_intermediate_features_for_plot(out, feature, LEVEL)
    make_single_figure(lang_df, feature, level=LEVEL, out_dir='./html/img', show=False, xtitle='All documents')
    for k, docdf in doc_dfs.items():
        make_single_figure(docdf, feature, level=LEVEL, out_dir='./html/img', show=False, xtitle=k, show_error_y=False)

#sys.exit(1)

html_csv_out = 'html/img/data'
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
                if language == "da":
                    # Danish data is not redistributable
                    current_df['token'] = ''
                    current_df['lemma'] = ''
                current_df.to_html(out_fis)
                current_df.to_csv(out_fis.replace('.html', '.csv'), index=False)


csv_tables_html = "<h1>Processed Data</h1>\n"
for language in LANG_ORDER:
    csv_tables_html += f'<a href="#{make_language_label(language)}">{make_language_label(language)}</a>\n<br />'

for language in LANG_ORDER:
    files = html_lang_paths[language]
    csv_tables_html += f'<a name="{make_language_label(language)}"></a>\n'
    csv_tables_html += f"<h2>{make_language_label(language)}</h2>\n<ul>\n"
    for html_file in files:
        file_name = os.path.basename(html_file)
        rel = os.path.relpath(html_file, start='html')
        csv_tables_html += f'<li><a href="{rel}">{file_name}</a></li>\n'
    csv_tables_html += "</ul>\n"

with open('html/csv_tables.html', 'w') as f:
    f.write(csv_tables_html)


html = "<h1>MultiplEYE Corpus Statistics per page.</h1>\n"
html += '<p><a href="csv_tables.html">Preprocessed Data</a></p>\n'
html += '<p><a href="feature_tables.html">Feature Data</a></p>\n'
html += "<h2>Language-wise Results:</h1>\n"
for text, link in links.items():
    html += f'<p><a href="{link}">{text}</a></p>\n'

html += "<h2>Text-wise Results:</h1>\n"
for text, link in text_wise_links.items():
    html += f'<p><a href="{link}">{text}</a></p>\n'

with open('html/index.html', 'w') as f:
    f.write(html)
