import streamlit as st
from streamlit_option_menu import option_menu
from local_css import local_css
import pandas as pd
import streamlit as st
import numpy as np

# For download buttons
from functionforDownloadButtons import download_button

# BERTopic
import bertopic

# GSDMM
from gsdmm import MovieGroupProcess
import re
import unicodedata
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

DATASETS = {
    'ABC News Dataset': ('datasets/abcnews.csv', 'headline_text'),
    'Stackover Flow Questions': ('datasets/stackoverflow.csv', 'Title'),
    'Trump Tweets': ('datasets/trump_tweets.csv', 'text'),
}

local_css("styles.css")


def _max_width_(width=1000):
    max_width_str = f"max-width: {width}px;"
    st.markdown(
        f"""
    <style>
     .block-container{{
        {max_width_str}
        padding-top: 36px;
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_(1000)

with st.sidebar:
    selected = option_menu(
        menu_title="Topic Modelling",
        menu_icon="chat-right-dots",
        options=["Home", "Datasets", "GSDMM", "BERTopic",
                 "Run GSDMM", "Run BERTopic", "References"],
        icons=["house", "clipboard-data", "grid", "cpu", "arrow-right-circle", "arrow-right-circle",
               "file-text"],
        styles={
            "nav-link-selected": {"background-color": "#ffdeb4;", "color": "black"},
            "menu-title": {"color": "", "font-weight": "900", "text-decoration": "bolder"}
        }
    )

if selected == "Home":
    # Introduction
    st.title("Short Text Topic Modelling 💬")
    st.image("images/intro.jpeg", width=970)

    intro = """
        <span class='highlight'>Short texts </span>  have become an important information source
        including <span class='highlight'>news headlines, status updates, web page snippets,
        tweets, question/answer pairs, feedback, </span>  etc. Short text analysis has been
        attracting increasing attention in recent years due to the ubiquity
        of short text in the real-world.
        Topic modeling is an <span class='highlight'>unsupervised machine learning technique
        </span>  that's capable of <span class='highlight'>scanning </span>  a set of documents,
        <span class='highlight'>detecting </span> word and phrase patterns within them, and 
        automatically<span class='highlight'>clustering word groups  </span>  and similar 
        expressions that best characterize a set of documents." Topic modeling is an unsupervised
        machine learning technique, in other words, one that <span class='highlight'>doesn't 
        require training. </span>
    """
    why = "Why is it a great area of research? 🔬"
    reason = """
        In <span class='highlight'>traditional topic modelling algorithms </span> , each document 
        may be viewed as a mixture of various topics and each topic is characterized by a distribution
        over all the words. However, traditional topic models experience large performance degradation 
        over short texts due to the <span class='highlight'> lack of word co-occurrence information
        in each short text. </span>  Therefore, short text topic modeling has already <span 
        class='highlight'>attracted much attention </span>  from the machine learning research community
        in recent years, which aims at <span class='highlight'>overcoming the problem of sparseness in
        short texts. </span> 
    """
    goal = "Our Goal 🎯"
    our_work = """
        We will focus on classifying specific short text datasets with the two Short Text Topic Modelling algorithms:
        - <span class='highlight'>Gibbs Sampling for Dirichlet Multinomial Mixture (GSDMM)</span>
        - <span class='highlight'>BERTopic </span>
        and then evaluate and compare the results.
    """
    st.write(intro, unsafe_allow_html=True)
    st.header(why)
    st.write(reason, unsafe_allow_html=True)

    st.image("images/intro2.png")

    st.header(goal)
    st.markdown(our_work, unsafe_allow_html=True)

if selected == "Datasets":
    # Datasets
    st.title(f"About our Datasets 📊")

    # ABC News
    df = pd.read_csv("datasets/abcnews.csv")
    st.header("ABC News Headlines")
    abc_news = """
        This contains data of <span class='highlight'>news headlines </span>  published over a period of years. 
        Sourced from the reputable <span class='highlight'>Australian news source ABC (Australian Broadcasting 
        Corporation) </span> This includes the entire <span class='highlight'>corpus of articles published by 
        the abcnews website </span> . With a volume of two hundred articles per day and a good focus on international 
        news, we can be <span class='highlight'>fairly certain that every event of significance has been 
        captured in this dataset </span>. Digging into the keywords, one can see all the important episodes 
        shaping the last decade and how they evolved over time.<br/><br/> We have selected the <span 
        class='highlight'>top 1000 headline samples </span>  from this dataset to run our algorithm.
        """
    st.write(abc_news, unsafe_allow_html=True)
    with st.expander("View the contents of the dataset"):
        st.dataframe(df)

    # Stackoverflow
    df = pd.read_csv("datasets/stackoverflow.csv")
    st.header("Stackoverflow Questions")
    stackoverflow = """
        <span class='highlight'>Stack Overflow </span>  is the <span class='highlight'>largest online community
        for programmers </span>  to learn, share their knowledge, and advance their careers. This is a dataset
        containing <span class='highlight'>1,000 Stack Overflow questions </span> . Questions are classified 
        into three categories:<br/>
        <span class='highlight'>HQ </span> : High-quality posts without a single edit.<br/>
        <span class='highlight'>LQ_EDIT </span> : Low-quality posts with a negative score, and multiple 
        community edits. However, they still remain open after those changes.<br/>
        <span class='highlight'>LQ_CLOSE </span> : Low-quality posts that were closed by the community without
        a single edit.<br/><br/>
        However, for STTM, we are only interested in the <span class='highlight'>'Title' column </span>  of 
        this dataset.
    """
    st.write(stackoverflow, unsafe_allow_html=True)
    with st.expander("View the contents of the dataset"):
        st.dataframe(df)

    # Trump Twitter Archive
    df = pd.read_csv("datasets/trump_tweets.csv")
    st.header("Trump Twitter Archive")
    trump_twitter = """
        The former US president <span class='highlight'>Donald Trump </span>  was <span class='highlight'>notoriously
        active on Twitter </span> . On January 8th, 2021, the platform decided to suspend his account, citing
        'the risk of further incitement of violence' following the violent riots at the US Capitol building 
        on Jan 6th. <span class='highlight'>Trump's Twitter activity </span>  constitutes an <span class='highlight'>
        important documentation of escalating polarisation </span>  in the US political and societal discourse
        during the second decade of the 2000s. This dataset contains Trump's tweets since November 2019 to December
        2019. It has a total of 931 samples.

        It was <span class='highlight'>collected from the website 'The Trump Archive' </span>  who did all the work
        in periodically scraping Trump's Twitter account until his suspension in 2021.
    """
    st.write(trump_twitter, unsafe_allow_html=True)
    with st.expander("View the contents of the dataset"):
        st.dataframe(df)


if selected == "GSDMM":
    # GSDMM Explaination
    st.title(f"What is GSDMM? 🤔")
    gsdmm = """
        GSDMM <span class='highlight'>(Gibbs Sampling Dirichlet Multinomial Mixture) </span>  is a short text clustering
        model proposed by Jianhua Yin and Jianyong Wang in a paper a few years ago. The model claims to <span 
        class='highlight'>solve the sparsity problem of short text clustering </span>  while also <span 
        class='highlight'>displaying word topics like LDA.</span> GSDMM can <span class='highlight'>infer the number
        of clusters automatically </span>  with a good balance between the <span class='highlight'>completeness and
        homogeneity </span>  of the clustering results, and is fast to converge. The <span class='highlight'>basic
        principle </span>  of GSDMM is described using an analogy called <span class='highlight'>'Movie Group
        Approach'.</span>
    """
    movie_group_approach = """
        Imagine a <span class='highlight'>group of students (documents) </span>  who all have a <span class='highlight'>
        list of favorite movies (words). </span>  The students are randomly assigned to <span class='highlight'>K tables
        .</span>
        At the instruction of a professor the students must shuffle tables with 2 goals in mind:
        <ul>
        <li> <span class='highlight'>Find a table with more students. </span> </li>
        <li> <span class='highlight'>Pick a table where your film interests align with those at the table. </span> </li>
        <li> <span class='highlight'>Rinse and repeat until you reach a plateau where the number of clusters does not change.
        </span> </li>
        </ul>
    """

    st.write(gsdmm, unsafe_allow_html=True)

    st.header("The Movie Group Process (MGP)")
    st.write(movie_group_approach, unsafe_allow_html=True)
    st.header("Influence of parameters in GSDMM")
    st.subheader("α and β")
    st.image("images/param.png")
    st.subheader("Analogy of Influence of α and β")
    st.image("images/influence.png")


if selected == "BERTopic":
    # BERTopic Explanation
    bert = """
        BERTopic is a topic modeling technique that <span class='highlight'>leverages BERT embeddings and c-TF-IDF </span>
        to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic 
        descriptions. The <span class='highlight'>two greatest advantages </span>  to BERTopic are arguably its <span
        class='highlight'>straight forward out-of-the-box usability </span>  and its <span class='highlight'>novel 
        interactive visualization methods.</span> Having an overall picture of the topics that have been learned by 
        the model allows us to generate an internal perception of the model's quality and the most notable themes 
        encapsulated in the corpus.
    """

    st.title("What is BERTopic? 🧐")
    st.write(bert, unsafe_allow_html=True)

    st.header("Algorithmic Stages in BERTopic")
    st.image("images/bert.png", width=800)

_max_width_(1000)

if selected == "Run GSDMM":
    # Find topics with GSDMM
    np.random.seed(493)
    ps = nltk.porter.PorterStemmer()

    st.title(f"Check out GSDMM Yourself! 👨‍💻")

    c30, c31, c32 = st.columns([2.5, 1, 3])
    with st.form(key="gsdmm"):
        Dataset = st.selectbox(
            'Which dataset would you like to use?',
            tuple(DATASETS.keys())
        )

        alpha = st.slider(
            "Set alpha α ",
            min_value=0.1,
            max_value=1.0,
            value=0.1,
            help="The prior probability of choosing a cluster.",
        )

        beta = st.slider(
            "Set beta β",
            min_value=0.1,
            max_value=1.0,
            value=1.0,
            help="The prior probability of choosing a cluster when there are no similar words.",
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            iterations = st.number_input(
                "Number of Iterations:",
                value=30,
                min_value=20,
                max_value=80,
                help="How many times should the process be run?",
            )

        with col2:
            k = st.number_input(
                "Adjust k:",
                value=15,
                min_value=10,
                max_value=50,
                help="The number of clusters to stop with.",
            )

        @st.cache(allow_output_mutation=True)
        def load_model():
            return MovieGroupProcess(K=k, alpha=alpha, beta=beta, n_iters=iterations)

        model = load_model()

        submit_button = st.form_submit_button(label="✨ Run GSDMM!")

    if not submit_button:
        st.stop()

    # read the tweets info a dataframe
    df = pd.read_csv(DATASETS[Dataset][0])

    # remove  null values
    df = df.loc[df[DATASETS[Dataset][1]].notnull()]

    def basic_clean(original):
        word = original.lower()
        word = unicodedata.normalize('NFKD', word)\
            .encode('ascii', 'ignore')\
            .decode('utf-8', 'ignore')
        word = re.sub(r"[^a-z0-9'\s]", '', word)
        word = word.replace('\n', ' ')
        word = word.replace('\t', ' ')
        return word

    def remove_stopwords(original, extra_words=[], exclude_words=[]):
        stopword_list = stopwords.words('english')

        for word in extra_words:
            stopword_list.append(word)
        for word in exclude_words:
            stopword_list.remove(word)

        words = original.split()
        filtered_words = [w for w in words if w not in stopword_list]

        original_nostop = ' '.join(filtered_words)

        return original_nostop

    def stem(original):
        wnl = WordNetLemmatizer()
        stems = [wnl.lemmatize(word) for word in original.split()]
        original_stemmed = ' '.join(stems)
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

    top_index = doc_count.argsort()[:][::-1]

    list_of_word_clouds = []

    def top_words(cluster_word_distribution, top_cluster, values):
        for cluster in top_cluster:
            sort_dicts = sorted(
                mgp.cluster_word_distribution[cluster].items(),
                key=lambda k: k[1],
                reverse=True,
            )[:values]
            frequencies = {}
            for i in sort_dicts:
                frequencies[i[0]] = i[1]
            if len(frequencies) != 0:
                wordcloud = WordCloud(
                    width=800, height=800, background_color="white", min_font_size=10
                ).generate_from_frequencies(frequencies)
                list_of_word_clouds.append(wordcloud)

    st.header("🎈 Topic Wordclouds")
    cols = st.columns(3)

    top_words(mgp.cluster_word_distribution, top_index, 100)
    for index, word_cloud in enumerate(list_of_word_clouds):
        with cols[index % 3]:
            st.image(word_cloud.to_image(), caption=f'Topic #{index}')

    topic_dict = {}
    topic_names = [f"Topic #{i}" for i in range(k)]

    for i, topic_num in enumerate(top_index):
        topic_dict[topic_num] = topic_names[i]

    def create_topics_dataframe(data_text=df[DATASETS[Dataset][1]],  mgp=mgp, threshold=0.3, topic_dict=topic_dict, stem_text=docs):
        result = pd.DataFrame(columns=['text', 'topic', 'stems'])
        for i, text in enumerate(data_text):
            result.at[i, 'text'] = text
            result.at[i, 'stems'] = stem_text[i]
            prob = mgp.choose_best_label(stem_text[i])
            if prob[1] >= threshold:
                result.at[i, 'topic'] = topic_dict[prob[0]]
            else:
                result.at[i, 'topic'] = 'Other'
        return result

    dfx = create_topics_dataframe(
        data_text=df[DATASETS[Dataset][1]],  mgp=mgp, threshold=0.3, topic_dict=topic_dict, stem_text=docs)

    dfx.topic.value_counts(dropna=False)
    st.header("🎈 Check & Download results")
    CSVButton = download_button(
        dfx, "gsdmm_topics.csv", "📥 Download (.csv)")
    st.dataframe(dfx)


if selected == "Run BERTopic":
    # Find topics with BERTopic
    st.title(f"Check out BERTopic Yourself! 👨‍💻")

    with st.form(key="bertopic"):

        option = st.selectbox(
            'Which dataset would you like to use?',
            tuple(DATASETS.keys())
        )

        top_words = st.slider(
            "Top `n` words:",
            min_value=1,
            max_value=50,
            value=10,
            help="""
                You can choose the number of words that represent each topic cluster.
                The top words are words with the highest c-TF-IDF scores
            """,
        )

        min_topic_size = st.slider(
            "Minimum size of each topic cluster:",
            min_value=1,
            max_value=40,
            value=10,
            help="You can choose the number of topics to display. Between 20 and 100, default number is 30.",
        )

        diversity = st.slider(
            "How diverse should the topics be?",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            help="""
                It uses MMR to calculate the similarity between words in a topic cluster.
                
                Setting this to 1 makes sure that dissimilar words are chosen to represent a topic and
                to 0 makes sure that similar words are chosen to represent a topic.
            """,
        )

        c1, c2 = st.columns([1, 1])

        with c1:
            min_Ngrams = st.number_input(
                "Minimum Ngram",
                min_value=1,
                max_value=6,
                value=2,
                help="""
                    The minimum value for the ngram range.
                    *Topics_ngram_range* sets the length of the resulting topics.

                    To extract topics, simply set *Topics_ngram_range* to (1, 2) or higher depending
                    on the number of words you would like in the resulting topics
                """,
            )
        with c2:
            max_Ngrams = st.number_input(
                "Maximum Ngram",
                min_value=1,
                max_value=6,
                value=3,
                help="""
                    The maximum value for the keyphrase_ngram_range.
                    *Topics_ngram_range* sets the length of the resulting topics.

                    To extract topics, simply set *topics_ngram_range* to (1, 2) or higher depending
                    on the number of words you would like in the resulting topics.
                """,
            )

        @st.cache(allow_output_mutation=True)
        def load_model():
            return bertopic.BERTopic(
                calculate_probabilities=True
            )

        model = load_model()

        submit_button = st.form_submit_button(label="✨ Run BERTopic!")

    if not submit_button:
        st.stop()

    if min_Ngrams > max_Ngrams:
        st.warning("min_Ngrams can't be greater than max_Ngrams")
        st.stop()

    docs = pd.read_csv(DATASETS[option][0])[DATASETS[option][1]]

    model = bertopic.BERTopic(
        n_gram_range=(
            min_Ngrams, max_Ngrams
        ),
        top_n_words=top_words,
        diversity=diversity,
        calculate_probabilities=True
    )

    topics, probs = model.fit_transform(
        docs
    )

    all_topics = model.get_topic_info()

    coherence_visual = model.visualize_topics()

    topic_hierarchy_visual = model.visualize_hierarchy()

    topic_similarity_visual = model.visualize_heatmap(width=1000, height=1000)

    topic_c_tf_idf_scores_visual = model.visualize_barchart()

    term_rank_visual = model.visualize_term_rank()

    st.markdown("## 🎈 Check & download results ")

    st.header("")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Topics", "Coherence Visualisation", "c-TF-IDF Scores", "Topic Hierarchies", "Term Rank Decline", "Topic Similarities"])

    with tab1:
        # st.header("Topics")
        CSVButton = download_button(
            all_topics, "bertopic_topics.csv", "📥 Download (.csv)")
        st.dataframe(all_topics)

    with tab2:
        # st.header("Coherence")
        st.plotly_chart(coherence_visual)

    with tab3:
        # st.header("c-TF-IDF Scores")
        st.plotly_chart(topic_c_tf_idf_scores_visual)

    with tab4:
        # st.header("Topic Hierarchy")
        st.plotly_chart(topic_hierarchy_visual)

    with tab5:
        # st.header("Term Rank Decline")
        st.plotly_chart(term_rank_visual)

    with tab6:
        # st.header("Topic Similarities")
        st.plotly_chart(topic_similarity_visual)

_max_width_()

if selected == "References":
    st.title(f"References 📝")

    survey_paper = """
        Qiang, J., Qian, Z., Li, Y., Yuan, Y., & Wu, X. (2020).<br/>
        Short text topic modeling techniques, applications, and performance: a survey.<br/>
        IEEE Transactions on Knowledge and Data Engineering, 34(3), 1427-1445.<br/>
    """

    gsdmm_paper = """
        Yin, J., & Wang, J. (2014, August).<br/>
        A dirichlet multinomial mixture model-based approach for short text clustering.<br/>
        In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and
        data mining (pp. 233-242).<br/>
    """

    bertopic_paper = """
        Grootendorst, M. (2022).<br/>
        BERTopic: Neural topic modeling with a class-based TF-IDF procedure.<br/>
        arXiv preprint arXiv:2203.05794.<br/>
    """

    st.subheader(
        "[Topic Modelling Survey](https://ieeexplore.ieee.org/abstract/document/9086136/)")
    st.markdown(survey_paper, unsafe_allow_html=True)

    st.subheader("[GSDMM](https://dl.acm.org/doi/abs/10.1145/2623330.2623715)")
    st.markdown(gsdmm_paper, unsafe_allow_html=True)

    st.subheader("[BERTopic](https://arxiv.org/abs/2203.05794)")
    st.markdown(bertopic_paper, unsafe_allow_html=True)
