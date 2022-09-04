import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gsdmm import MovieGroupProcess
import re
import unicodedata
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk
import csv
import streamlit as st
import numpy as np
from pandas import DataFrame
import pandas as pd
from keybert import KeyBERT

# For Flair (Keybert)
from flair.embeddings import TransformerDocumentEmbeddings
import seaborn as sns

# For download buttons
from functionforDownloadButtons import download_button
import os
import json

# set seed for reproducibility
np.random.seed(493)


DATASETS = {
    "Trump Tweets": ("trump_tweets.csv", "text"),
    "Stackover Flow Questions": ("stackoverflow.csv", "Title"),
    "ABC News Dataset": ("abcnews-date-text.csv", "headline_text"),
}


nltk.download("stopwords")
nltk.download("wordnet")
ps = nltk.porter.PorterStemmer()


st.set_page_config(
    page_title="BERT Keyword Extractor",
    page_icon="🎈",
)


def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    # st.image("logo.png", width=400)
    st.title("GSDMM - Topic Modelling")
    st.header("")


with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
GSDMM Intro
	    """
    )

    st.markdown("")

st.markdown("")
st.markdown("## **📌 Paste document **")
with st.form(key="my_form"):

    # ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    # with c1:
    Dataset = st.selectbox(
        "Which dataset would you like to use?", tuple(DATASETS.keys())
    )

    alpha = st.slider(
        "alpha",
        min_value=0.1,
        max_value=1.0,
        value=0.1,
        help="_______",
    )
    beta = st.slider(
        "beta",
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        help="_______",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        iterations = st.number_input(
            "Number of Iterations:",
            value=40,
            min_value=20,
            max_value=80,
            help="""_______""",
            # help="Minimum value for the keyphrase_ngram_range. keyphrase_ngram_range sets the length of the resulting keywords/keyphrases. To extract keyphrases, simply set keyphrase_ngram_range to (1, # 2) or higher depending on the number of words you would like in the resulting keyphrases.",
        )

    with col2:
        k = st.number_input(
            "Adjust k:",
            value=30,
            min_value=10,
            max_value=50,
            help="""_______""",
        )

    @st.cache(allow_output_mutation=True)
    def load_model():
        return MovieGroupProcess(K=k, alpha=alpha, beta=beta, n_iters=iterations)

    model = load_model()

    submit_button = st.form_submit_button(label="✨ Get me the data!")


if not submit_button:
    st.stop()


# read the tweets info a dataframe
df = pd.read_csv(DATASETS[Dataset][0])

# remove  null values
df = df.loc[df[DATASETS[Dataset][1]].notnull()]
nltk.download("punkt")


def basic_clean(original):
    word = original.lower()
    word = (
        unicodedata.normalize("NFKD", word)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    word = re.sub(r"[^a-z0-9'\s]", "", word)
    word = word.replace("\n", " ")
    word = word.replace("\t", " ")
    return word


def remove_stopwords(original, extra_words=[], exclude_words=[]):
    stopword_list = stopwords.words("english")

    for word in extra_words:
        stopword_list.append(word)
    for word in exclude_words:
        stopword_list.remove(word)

    words = original.split()
    filtered_words = [w for w in words if w not in stopword_list]

    original_nostop = " ".join(filtered_words)

    return original_nostop


def stem(original):
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in original.split()]
    original_stemmed = " ".join(stems)
    return original_stemmed


docs = []
for sentence in df[DATASETS[Dataset][1]]:
    words = word_tokenize(stem(remove_stopwords(basic_clean(sentence))))
    docs.append(words)

mgp = MovieGroupProcess(K=k, alpha=alpha, beta=beta, n_iters=iterations)

vocab = set(x for doc in docs for x in doc)
n_terms = len(vocab)

y = mgp.fit(docs, n_terms)

doc_count = np.array(mgp.cluster_doc_count)

top_index = doc_count.argsort()[-15:][::-1]


def top_words(cluster_word_distribution, top_cluster, values):
    for cluster in top_cluster:
        sort_dicts = sorted(
            mgp.cluster_word_distribution[cluster].items(),
            key=lambda k: k[1],
            reverse=True,
        )[:values]


top_words(mgp.cluster_word_distribution, top_index, 7)

topic_dict = {}
topic_names = [
    "Topic #1",
    "Topic #2",
    "Topic #3",
    "Topic #4",
    "Topic #5",
    "Topic #6",
    "Topic #7",
    "Topic #8",
    "Topic #9",
    "Topic #10",
    "Topic #11",
    "Topic #12",
    "Topic #13",
    "Topic #14",
    "Topic #15",
]
for i, topic_num in enumerate(top_index):
    topic_dict[topic_num] = topic_names[i]


def create_topics_dataframe(
    data_text=df[DATASETS[Dataset][1]],
    mgp=mgp,
    threshold=0.3,
    topic_dict=topic_dict,
    stem_text=docs,
):
    result = pd.DataFrame(columns=["text", "topic", "stems"])
    for i, text in enumerate(data_text):
        result.at[i, "text"] = text
        result.at[i, "stems"] = stem_text[i]
        prob = mgp.choose_best_label(stem_text[i])
        if prob[1] >= threshold:
            result.at[i, "topic"] = topic_dict[prob[0]]
        else:
            result.at[i, "topic"] = "Other"
    return result


dfx = create_topics_dataframe(
    data_text=df[DATASETS[Dataset][1]],
    mgp=mgp,
    threshold=0.3,
    topic_dict=topic_dict,
    stem_text=docs,
)

dfx.topic.value_counts(dropna=False)


list_of_word_clouds = []


def word_multiply(word, count):
    s = ""
    for _ in range(count):
        s += word + " "
    return s


def top_words(cluster_word_distribution, top_cluster, values):
    for cluster in top_cluster:
        sort_dicts = sorted(
            mgp.cluster_word_distribution[cluster].items(),
            key=lambda k: k[1],
            reverse=True,
        )[:values]
        just_words = [word_multiply(ele[0], ele[1])
                      for ele in sort_dicts if ele[1] > 0]
        if len(just_words) != 0:
            wordcloud = WordCloud(
                width=800, height=800, background_color="white", min_font_size=10
            ).generate(" ".join(just_words))
            list_of_word_clouds.append(wordcloud)
        # print("Cluster %s : %s" % (cluster, sort_dicts))
        # print("-" * 120)


top_words(mgp.cluster_word_distribution, top_index, 7)


st.set_option("deprecation.showPyplotGlobalUse", False)
for word_cloud in list_of_word_clouds:

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(word_cloud)
    plt.axis("off")
    fig = plt.tight_layout(pad=0)

    st.pyplot(fig)


st.markdown("## **🎈 Check & download results **")

st.header("")

st.table(dfx)

st.header("")
