from pprint import pprint
import streamlit as st
import numpy as np
from pandas import DataFrame
import pandas as pd
from keybert import KeyBERT

# For Flair (WordEmbedding)
from flair.embeddings import TransformerDocumentEmbeddings
import seaborn as sns

# For download buttons
from functionforDownloadButtons import download_button
import os
import json
import bertopic

DATASETS = {
    'Trump Tweets': ('trump_tweets.csv', 'text'),
    'Stackover Flow Questions': ('stackoverflow.csv', 'Title'),
    'ABC News Dataset': ('abcnews-date-text.csv', 'headline_text')
}

MODELS = {
    'Default (BERT)': 'bert-base-uncased',
    'XLNET': 'xlnet-base-uncased',
    'DistilRoberta': 'distilroberta-base',
    'Albert': 'albert-base-v2'

}

st.set_page_config(
    page_title="BERTopic",
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

# c30, c31, c32 = st.columns([2.5, 1, 3])


st.title("BERTopic")
st.header("")


with st.expander("ℹ️ About", expanded=True):

    st.write(
        """     
- BERTopic is a topic modeling technique that leverages hugs transformers and c-TF-IDF to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic descriptions.

- BERTopic supports guided, (semi-) supervised, and dynamic topic modeling. It even supports visualizations similar to LDAvis!
	    """
    )

    st.markdown("")

st.markdown("")
st.markdown("## 📌 Try it out!")

with st.form(key="my_form"):

    # ce, c1, ce, c2, c3 = st.columns([1, 1, 1, 5, 1])
    # with c1:

    # uploaded_file = st.file_uploader("Choose a file")

    option = st.selectbox(
        'Which dataset would you like to use?',
        tuple(DATASETS.keys())
    )

    num_topics = st.slider(
        "# of topics:",
        min_value=20,
        max_value=100,
        value=30,
        help="You can choose the number of topics to reduce to. Between 20 and 100, default number is 30.",
    )

    min_topic_size = st.slider(
        "Minimum size of each topic cluster:",
        min_value=1,
        max_value=40,
        value=10,
        help="You can choose the number of topics to display. Between 20 and 100, default number is 30.",
    )

    diversity = st.slider(
        "How diverse should the topics be? (0 for no diversity and 1 for maximum diversity):",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        help="You can choose the number of topics to display. Between 20 and 100, default number is 30.",
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        min_Ngrams = st.number_input(
            "Minimum Ngram",
            min_value=1,
            max_value=6,
            value=2,
            help="""The minimum value for the ngram range.

    *Topics_ngram_range* sets the length of the resulting topics.

    To extract topics, simply set *Topics_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting topics""",
        )
    with c2:
        max_Ngrams = st.number_input(
            "Maximum Ngram",
            min_value=1,
            max_value=6,
            value=3,
            help="""The maximum value for the keyphrase_ngram_range.

    *Topics_ngram_range* sets the length of the resulting topics.

    To extract topics, simply set *topics_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting topics.""",
        )

    StopWordsCheckbox = st.checkbox(
        "Remove stop words",
        value=True,
        help="Tick this box to remove stop words from the dataset (currently English only)",
    )

    title = st.text_input('Enter a keyword to search for:', '')

    ModelType = st.radio(
        "Choose your model",
        MODELS.keys(),
        help="At present, you can choose between 4 models to embed your text. More to come!",
    )

    @st.cache(allow_output_mutation=True)
    def load_model():
        return bertopic.BERTopic(
            calculate_probabilities=True
        )

    model = load_model()

    # col1, col2 = st.columns([1,1], gap='small')

    


    submit_button = st.form_submit_button(label="✨ Show me the Topics!")

if not submit_button:
    st.stop()

if min_Ngrams > max_Ngrams:
    st.warning("min_Ngrams can't be greater than max_Ngrams")
    st.stop()



docs = pd.read_csv(DATASETS[option][0])[DATASETS[option][1]]

model = bertopic.BERTopic(
    embedding_model=TransformerDocumentEmbeddings(
        MODELS[str(ModelType)]),
    n_gram_range=(
        min_Ngrams, max_Ngrams
    ),
    nr_topics=num_topics,
    diversity=diversity,
    calculate_probabilities=True
)

topics, probs = model.fit_transform(
    docs
)
all_topics = model.get_topic_info()

coherence_visual = model.visualize_topics()

# documents_visual = model.visualize_documents(docs)

# hierarchical_topics = model.hierarchical_topics(docs, topics)
# document_hierarchy_visual = model.visualize_hierarchical_documents(
#     docs=docs, hierarchical_topics=hierarchical_topics)

topic_hierarchy_visual = model.visualize_hierarchy()

# topic_distribution_visual = model.visualize_distribution(probs)

topic_similarity_visual = model.visualize_heatmap(width=1000, height=1000)

topic_c_tf_idf_scores_visual = model.visualize_barchart()

term_rank_visual = model.visualize_term_rank()

# term_rank_visual = model.visualize_term_rank()

# similar_topics, similarity = model.find_topics("vehicle", num_topics=5)

# for topic in similar_topics:
#   print("-" * 50)
#   pprint(model.get_topic(topic))
#   print("-" * 50)


# topics_per_class_visual = model.visualize_topics_per_class()

st.markdown("## 🎈 Check & download results ")

st.header("")


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Topics", "Coherence", "c-TF-IDF Scores", "Topic Hierarchies", "Term Rank", "Topic Similarities", "Search"])

with tab1:
    st.header("Topics")
    CSVButton = download_button(all_topics, "topics.csv", "📥 Download (.csv)")
    st.table(all_topics)

with tab2:
    st.header("Coherence")
    st.plotly_chart(coherence_visual)

with tab3:
    st.header("c-TF-IDF Scores")
    st.plotly_chart(topic_c_tf_idf_scores_visual)

with tab4:
    st.header("Topic Hierarchy")
    st.plotly_chart(topic_hierarchy_visual)

with tab5:
    st.header("Term Rank Decline")
    st.plotly_chart(term_rank_visual)

with tab6:
    st.header("Topic Similarities")
    st.plotly_chart(topic_similarity_visual)

with tab7:
    st.header("Find a topic of your interest!")
    similar_topics, similarity = model.find_topics(title)
    st.table(similar_topics)
    # st.plotly_chart(topic_distribution_visual)

# with tab6:
#     st.header("Topic Hierarchy")
#     st.plotly_chart(topics_per_class_visual)
