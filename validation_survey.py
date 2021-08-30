from gensim.models.phrases import Phraser
from gensim.corpora import Dictionary

from get_google import pull_sheet

import dash
import dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import re

import spacy
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

import gensim
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim import corpora, models

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- LOAD DATA ----------- #

monster = pull_sheet('monster_data', 'Sheet1')
youtube = pull_sheet('youtube_data', 'Sheet1')
# NEED TO OUTPUT TABLE WITH LINK AND AND IMAGE
# ALSO NEED TO OUTPUT TABLE WITH


# ----------- DEFINE STOPWORDS ----------- #

stop_words = stopwords.words('english')
stop_words.extend(['from', 'experience', 're', 'edu', 'use','system','team','ability',
                   'excellent','responsible','year','years','requirements','requirement',
                   'skill','skills','candidate','work','level','tool','tools'])


# ----------- LOAD MODELS ----------- #

bigram = Phraser.load('bigram')
dirichlet_dict = Dictionary.load_from_text('dictionary')
lda_model = gensim.models.LdaMulticore.load('lda.model', mmap='r')


# ---------- APP ---------- #

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    nest = []
    for word in texts:
        if word.pos_ in allowed_postags:
            texts_out.append(word.lemma_)
    nest.append(texts_out)
    return nest


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

controls = dbc.Card(
    [
        html.P('NetID'),
        dbc.Input(id="net_id", placeholder="Input NetID (text before @ in email)", type="text"),
        html.Div(id='net_input'),
        dbc.FormGroup(
            [
                dbc.Label("Role"),
                dcc.Dropdown(
                    id="role",
                    options=[
                        {'label': 'Software Engineer', 'value': 'Software Engineer'},
                        {'label': 'Software Developer', 'value': 'Software Developer'},
                        {'label': 'Data Scientist', 'value': 'Data Scientist'},
                        {'label': 'Cloud Engineer', 'value': 'Cloud Engineer'},
                        {'label': 'Data Analyst', 'value': 'Data Analyst'}
                    ],
                    value="Software Engineer",
                ),
            ],
        ),
        dbc.FormGroup(
            [
                dbc.Label("Experience"),
                dcc.Dropdown(
                    id="experience",
                    options=[
                        {'label': 'Entry Level', 'value': 'entry level'},
                        {'label': 'Senior', 'value': 'Senior'},
                        {'label': 'Manager', 'value': 'Manager'}
                    ],
                    value="Entry Level",
                ),
            ],
        ),
        html.Div(
            [
                dbc.Label('Role Responsibilities'),
                dbc.Textarea(id='role_description',
                             className='mb-3',
                             placeholder='Briefly describe the role you would like to be considered for'),
            ]

        ),
        dbc.Button("Submit", outline=True, color="dark", id='button', n_clicks=0),
        html.Div(id='my_output'),
        dbc.Button("Get Videos", outline=True, color="dark", id='button2', n_clicks=0),
    ],
    body=True,
)

app.layout = dbc.Container(
    [
        html.H1("eLearning Recommendation Engine"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                html.Div(id='intermediate_value'),  # Hidden div inside the app that stores the intermediate value
                # html.Div(id='intermediate_value2')
                html.Div(id='job_corpus')
            ],
            align="center",
        ),
    ],
    fluid=True,
)


# ----------- SERVER ----------- #

@app.callback(
    [
        Output('my_output', 'children'),
        Output('intermediate_value', 'children')
    ],
    [
        Input('button', 'n_clicks'),
        State("role", "value"),
        State("experience", "value"),
        State("role_description", "value")
    ],
)
def assign_topic(n_clicks, role, experience, role_description):

    corpus = f'{role} {experience} {role_description}'

    corpus_lower = re.sub('[,\.!?]', '', corpus).lower()
    corpus_cleaned = gensim.utils.simple_preprocess(corpus_lower, deacc=True)
    corpus_no_stop = [word for word in corpus_cleaned if not word in stop_words]

    corpus_bigram = bigram[corpus_no_stop]
    corpus_nlp = nlp(' '.join(corpus_bigram))

    corpus_lemma = lemmatization(corpus_nlp)

    corpus_ld2word = [dirichlet_dict.doc2bow(text) for text in corpus_lemma]
    pred = max(lda_model[corpus_ld2word[0]], key=lambda i: i[1])[0]

    monster_pred = monster[monster['prediction'] == pred].drop_duplicates(subset=['qualifications'], keep='first')

    return dash_table.DataTable(
                style_cell={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                },
                id='table',
                columns=[
                    {"name": i, "id": i, "deletable": True, "selectable": True} for i in monster_pred.columns
                ],
                data=monster_pred.to_dict('records'),
                editable=True,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                row_selectable="multi",
                selected_columns=[],
                selected_rows=[],
                page_action="native",
                page_current=0
            ), pred


@app.callback(
    Output('job_corpus', 'children'),
    [
        Input('button2', 'n_clicks'),
        Input('intermediate_value', 'children'),
        State('table', 'selected_rows'),
    ]
)
def get_videos(n_clicks, intermediate_value, selected_rows):

    # CREATE CLEAN TOKENS FOR VIDEOS
    df_yt = youtube[youtube['prediction'] == intermediate_value]
    df_yt['text'] = np.where(df_yt['text'].isnull, df_yt['transcript'], df_yt['text'])
    df_yt = df_yt[df_yt['text'].notnull()]
    df_yt['com_text'] = np.where((~df_yt['transcript'].isnull()) & (df_yt['transcript'] != df_yt['text']),
                                 df_yt['text'] + df_yt['transcript'], df_yt['text'])
    df_yt['com_text_clean'] = df_yt['com_text'].map(lambda x: re.sub('[,\.!?]', '', x))

    # Convert the titles to lowercase
    df_yt['com_text_clean'] = df_yt['com_text_clean'].map(lambda x: x.lower())
    df_yt['com_text_clean'] = df_yt['com_text_clean'].apply(lambda x: gensim.utils.simple_preprocess(x, deacc=True))

    # Remove Stop Words
    # df_yt['com_text_clean'] = df_yt['com_text_clean'].apply(lambda x: [word for word in x if not word in stop_words])

    # initialize vectorizer

    data = monster[monster['prediction'] == intermediate_value]
    print(data)
    vectorizer = CountVectorizer(stop_words='english')

    if n_clicks != 0:

        selected_jobs = data.iloc[selected_rows]
        print('selected_jobs df')
        print(selected_jobs)

        # create corpus and list to compare jobs to
        combined_text = selected_jobs.qualifications.to_list()

        X = vectorizer.fit_transform(combined_text)
        vid_words = df_yt.com_text_clean.tolist()

        # create dataframe with cosine similarity
        similarity = []
        for text in vid_words:
            y = vectorizer.transform(text)
            similarity.append(cosine_similarity(X, y).mean())

        df_yt['cosine_similarity'] = similarity

        df_sorted = df_yt.sort_values('cosine_similarity', ascending=False)[['video_id','title','img_url']].\
            drop_duplicates(subset=['video_id'], keep='first')

        df_sorted['video_link'] = 'https://www.youtube.com/watch?v=' + df_sorted['video_id'].astype(str)

        print(df_sorted)

        return dash_table.DataTable(
                    style_cell={
                        'whiteSpace': 'normal',
                        'height': 'auto',
                    },
                    id='table_vid',
                    columns=[
                        {"name": i, "id": i, "deletable": True, "selectable": True} for i in df_sorted.columns
                    ],
                    data=df_sorted.to_dict('records'),
                    editable=True,
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    selected_columns=[],
                    selected_rows=[],
                    page_action="native",
                    page_current=0
                )


# functionality is the same for both dropdowns, so we reuse filter_options
def just_print(input):
    return input
app.callback(Output("net_input", "children"), [Input("net_id", "value")])(
    just_print)

# app.callback(Output("y-variable", "options"), [Input("x-variable", "value")])(
#     filter_options
# )


if __name__ == "__main__":
    app.run_server(debug=True, port=8000)