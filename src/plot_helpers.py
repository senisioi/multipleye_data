import pandas as pd
import plotly.graph_objects as go
import os

from language_constants import (LANG_COLORS,
                                LANGUAGES,
                                SPACY_LANGUAGES,
                                CODE2LANG,
                                LANG_ORDER,
                                )

DEFAULT_COLOR = "#f97316"

from spacy.util import get_lang_class


def exists_spacy_blank(lang_code):
    try:
        get_lang_class(lang_code)
        return True
    except:
        return False


def make_language_label(lang_code):
    if lang_code in SPACY_LANGUAGES:
        return CODE2LANG.get(lang_code, lang_code)
    elif lang_code in {'he', 'tr', 'zh', 'yue'}:
        return CODE2LANG.get(lang_code, lang_code)
    elif exists_spacy_blank(lang_code):
        return CODE2LANG.get(lang_code, lang_code) + "*"
    return CODE2LANG.get(lang_code, lang_code) + "-"


def doc_arrays(doc_name, doc_stats):
    ys, es = [], []
    for lang in doc_stats.keys():
        df = doc_stats[lang]
        row = df[df["stimulus_name"] == doc_name]
        if row.empty:
            ys.append(None)
            es.append(None)
        else:
            ys.append(row["mean"].values[0])
            es.append(row["sem"].values[0])
    return ys, es


def make_intermediate_features_for_plot(feature_by_lang, feature, level="page"):
    lang_rows = []
    for lang, df in feature_by_lang.items():
        if feature not in df:
            raise ValueError(f"Feature {feature} not found in {df.columns}")
        lang_rows.append(
            {"lang": lang, "mean": df[feature].mean(), "sem": df[feature].sem()}
        )
    lang_df = pd.DataFrame(lang_rows)
    languages_in_order = [lang for lang in LANG_ORDER if lang in lang_df["lang"].values]
    lang_df.index = lang_df["lang"]
    lang_df = lang_df.loc[languages_in_order]

    doc_stats = {}
    for lang, df in feature_by_lang.items():
        agg = (
            df.groupby("stimulus_name")
            .agg(mean=(feature, "mean"), sem=(feature, "sem"))
            .reset_index()
        )
        doc_stats[lang] = agg

    all_docs = sorted(set().union(*[d["stimulus_name"] for d in doc_stats.values()]))
    doc_dfs = {}
    for doc_name in all_docs:
        ys, es = doc_arrays(doc_name, doc_stats)
        current_df = pd.DataFrame(
            {"lang": list(doc_stats.keys()), "mean": ys, "sem": es}
        )
        current_df.index = current_df["lang"]
        current_df = current_df.loc[languages_in_order]
        doc_dfs[doc_name] = current_df
    return lang_df, doc_dfs, doc_stats


def color_for(lang):
    return LANG_COLORS.get(lang, DEFAULT_COLOR)


def make_single_figure(
    lang_df,
    feature,
    level="page",
    out_dir="./",
    xtitle="",
    show=True,
    show_error_y=True,
):
    fig_lang = go.Figure()
    for lang, row in lang_df.iterrows():
        mean_val = row["mean"]
        fig_lang.add_trace(
            go.Bar(
                x=[make_language_label(lang)],
                y=[mean_val],
                error_y=dict(type="data", array=[row["sem"]]) if show_error_y else None,
                name=make_language_label(lang),
                marker_color=color_for(lang),
                visible="legendonly" if mean_val == 0 else True,
            )
        )

    fig_lang.update_layout(
        barmode="group",
        yaxis_title=f"mean {feature} per {level}",
        title=f"{xtitle}: mean {feature} per {level} by language",
        # width=1200,
        # title_y=0.89,
    )
    fig_lang.update_xaxes(
        categoryorder="array", categoryarray=LANG_ORDER, tickangle=45  # 🔹 Rotate labels
    )
    if show:
        fig_lang.show()
    html_out_dir = os.path.join(out_dir, "html", "single", xtitle)
    os.makedirs(html_out_dir, exist_ok=True)
    html_out = os.path.join(html_out_dir, f"{feature}_per_{level}_{xtitle}.html")
    fig_lang.write_html(html_out)
    # remove legend on pdf
    fig_lang.update_layout(showlegend=False)  # , width=800)
    pdf_out_dir = os.path.join(out_dir, "pdf", "single", xtitle)
    os.makedirs(pdf_out_dir, exist_ok=True)
    fig_lang.write_image(
        os.path.join(pdf_out_dir, f"{feature}_per_{level}_{xtitle}.pdf")
    )
    return html_out


def make_line_plots(feature_by_lang,
    feature,
    level="page",
    out_dir="./",
    xtitle="",
    show=True,
    show_error_y=True,
):
    _, doc_dfs, doc_stats = make_intermediate_features_for_plot(
        feature_by_lang, feature, level
    )
    texts = doc_stats['ro'].stimulus_name.values
    fig_all_docs = go.Figure()
    for lang in LANG_ORDER:
        if not lang in doc_stats:
            continue
        df = doc_stats[lang]
        mean_val = df["mean"].iloc[0]
        semm = df['sem']
        fig_all_docs.add_trace(
            go.Scatter(
                x=df["stimulus_name"],
                y=df["mean"],
                error_y=dict(type="data", array=df["sem"]) if show_error_y else None,
                mode='lines+markers',
                name=make_language_label(lang),
                marker_color=color_for(lang),
                visible="legendonly" if mean_val == 0 else True,
                text=texts,
                #hovertemplate = 'Document: %{text}<br> Language: {lang}<br> Mean: %{y:.2f}<br>SEM: %{semm:.2f}<extra></extra>'
                hovertemplate= f'Document: %{{text}}<br> Language: {make_language_label(lang)}<br> Mean: %{{y:.2f}}<br><extra></extra>'
                #hovertemplate= f'Document: %{{text}}<br> Language: {make_language_label(lang)}<br> Mean: %{{y:.2f}}<br>SEM: %{{error_y}}<extra></extra>'
            )
        )
    fig_all_docs.update_layout(
        yaxis_title=f"mean {feature} per {level}",
        title=f"Text {feature} per {level}",
        xaxis=dict(tickfont=dict(size=14)),
        legend=dict(font=dict(size=14)),
    )

    if show:
        fig_all_docs.show()

    html_out_dir = os.path.join(out_dir, "html", "textlevel")
    os.makedirs(html_out_dir, exist_ok=True)
    html_out = os.path.join(html_out_dir, f"{feature}_per_{level}_text.html")
    fig_all_docs.write_html(html_out)
    return html_out



def make_wide_figure(
    feature_by_lang,
    feature,
    level="page",
    out_dir="./",
    xtitle="",
    show=True,
    show_error_y=True,
):
    _, _, doc_stats = make_intermediate_features_for_plot(
        feature_by_lang, feature, level
    )
    fig_all_docs = go.Figure()
    for lang in LANG_ORDER:
        if not lang in doc_stats:
            continue
        df = doc_stats[lang]
        mean_val = df["mean"].iloc[0]
        fig_all_docs.add_trace(
            go.Bar(
                x=df["stimulus_name"],
                y=df["mean"],
                error_y=dict(type="data", array=df["sem"]) if show_error_y else None,
                name=make_language_label(lang),
                marker_color=color_for(lang),
                visible="legendonly" if mean_val == 0 else True,
            )
        )

    fig_all_docs.update_layout(
        barmode="group",
        yaxis_title=f"mean {feature} per {level}",
        title=f"All documents {feature} per {level} by language",
        # template="plotly_dark",
        width=12000,
    )
    if show:
        fig_all_docs.show()

    html_out_dir = os.path.join(out_dir, "html", "wide")
    os.makedirs(html_out_dir, exist_ok=True)
    html_out = os.path.join(html_out_dir, f"{feature}_per_{level}_wide_{xtitle}.html")
    fig_all_docs.write_html(html_out)
    return html_out


def make_combined_figure(
    feature_by_lang, feature, level="page", out_dir="./", show=True
):
    lang_df, doc_dfs, _ = make_intermediate_features_for_plot(
        feature_by_lang, feature, level
    )

    fig_doc = go.Figure()

    ys0 = []
    es0 = []
    languages = lang_df["lang"].tolist()
    # for lang in languages_in_order:
    for lang, row in lang_df.iterrows():
        # row = lang_df[lang_df["lang"] == lang]
        ys0.append(row["mean"])
        es0.append(row["sem"])
        mean_val = row["mean"]
        fig_doc.add_trace(
            go.Bar(
                x=[make_language_label(lang)],
                y=[mean_val],
                error_y=dict(type="data", array=[row["sem"]]),
                name=make_language_label(lang),
                marker_color=color_for(lang),
                visible="legendonly" if mean_val == 0 else True,
            )
        )

    buttons = [
        dict(
            label="All",
            method="restyle",
            args=[
                {
                    "y": [[v] for v in ys0],
                    "error_y.array": [[s if (s is not None) else 0] for s in es0],
                }
            ],
        )
    ]

    for doc_name, doc_df in doc_dfs.items():
        assert doc_df["lang"].tolist() == languages, "Languages do not match!"
        buttons.append(
            dict(
                label=doc_name,
                method="restyle",
                args=[
                    {
                        "y": [[v] for v in doc_df["mean"]],
                        "error_y.array": [None for s in doc_df["sem"]],
                    }
                ],
            )
        )

    fig_doc.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                x=0,
                y=1.15,
                xanchor="left",
                yanchor="top",
            )
        ],
        legend=dict(
            font=dict(
                size=13,
            )
        ),
        yaxis_title=f"mean {feature} per {level}",
        title=f"Document {feature} by {level}:",
        # title_y=0.97,
        # template="plotly_dark",
    )
    if show:
        fig_doc.show()

    html_out_dir = os.path.join(out_dir, "html", "combined")
    os.makedirs(html_out_dir, exist_ok=True)
    html_out = os.path.join(html_out_dir, f"{feature}_per_{level}_comb.html")
    fig_doc.write_html(html_out)
    return html_out
    # fig_doc.update_layout(showlegend=False)
    # pdf_out_dir = os.path.join(out_dir, "pdf", "combined")
    # os.makedirs(pdf_out_dir, exist_ok=True)
    # fig_doc.write_image(os.path.join(pdf_out_dir, f"{feature}_per_{level}_comb.pdf"))
