import base64
import streamlit as st
from streamlit_option_menu import option_menu
from local_css import local_css
import json
import pandas as pd
from pprint import pprint
import streamlit as st
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer

# For Flair (WordEmbedding)
from keybert import KeyBERT
from flair.embeddings import TransformerDocumentEmbeddings
import seaborn as sns

# For download buttons
from functionforDownloadButtons import download_button
import os
import json
import bertopic

from gsdmm import MovieGroupProcess
import re
import unicodedata
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk
local_css("styles.css")

with st.sidebar :
    selected = option_menu(
            menu_title="STTM",
            menu_icon="chat-square-text-fill",
            options=["Home", "Datasets", "GSDMM", "Run GSDMM", "BERTopic", "Run BERTopic", "KeyBERT", "Run KeyBERT", "References"],
            icons=["house-fill", "server", "code-square", "terminal-fill", "code-square", "terminal-fill", "code-square", "terminal-fill", "file-text-fill"],
            styles={
                "nav-link-selected" : {"background-color" : "#ffa429", "color" : "black" },
            }
        )

if selected == "Home" :    
    st.title("Short Text Topic Modelling 💬")
    with st.expander("Topic Modelling on Short Text", expanded=True):
        st.image("img/intro.jpeg")
    text = """
            <span class='highlight'>Short texts</span> have become an important information source
            including <span class='highlight'>news headlines, status updates, web page snippets,
            tweets, question/answer pairs, feedback,</span> etc. Short text analysis has been
            attracting increasing attention in recent years due to the ubiquity
            of short text in the real-world.</br>
        """
    text2 = "<div>Topic modeling is an<span class='highlight'>unsupervised machine learning technique</span>that's capable of <span class='highlight'>scanning</span> a set of documents,<span class='highlight'>detecting</span>word and phrase patterns within them, and automatically<span class='highlight'>clustering word groups</span>and similar expressions that best characterize a set of documents</div>"
    text3 = "<br/>Topic modeling is an unsupervised machine learning technique, in other words, one that <span class='highlight'>doesn't require training</span>."
    text4 = "Why is STTM a great area of research? 🔬"
    text5 = """
            In <span class='highlight'>traditional topic modelling algorithms</span>, each document
            may be viewed as a mixture of various topics and each topic
            is characterized by a distribution over all the words. <br/><br/>
            However, traditional topic models experience
            large performance degradation over short texts due to the <span class='highlight'>lack
            of word co-occurrence information in each short text</span>. <br/><br/>
            Therefore, short text topic modeling has already<span class='highlight'>attracted much attention</span> from the machine learning research community in recent
            years, which aims at <span class='highlight'>overcoming the problem of sparseness in
            short texts</span>.
        """
    text6 = "Our Goal 🎯"
    text7 = "Here we will focus on running dedicated STTM algorithms on specific short datasets.<br/>We will then evaluate and compare their results.<br/>"
    text8 = "<span class='highlight'>Gibbs Sampling for Dirichlet Multinomial Mixture (GSDMM)<br/>&nbsp;&nbsp;Bidirectional Encoder Representations from Transformers & TF-IDF (BERTopic)</span>"
    text9 = "<span class='highlight'>Twitter Trump Archives Dataset<br/>&nbsp;&nbsp;ABC News Headlines Dataset<br/>&nbsp;&nbsp;Stack Overflow Dataset</span>"
    st.write(text, unsafe_allow_html=True)
    st.write(text2, unsafe_allow_html=True)
    st.write(text3, unsafe_allow_html=True)
    st.header(text4)
    st.write(text5, unsafe_allow_html=True)
    with st.expander("Topic Modelling Workflow", expanded=True):
        st.image("img/intro2.png")
    st.header(text6)
    st.write(text7, unsafe_allow_html=True)
    st.subheader("STTM Algorithms")
    st.write(text8, unsafe_allow_html=True)
    st.subheader("Short Text Datasets")
    st.write(text9, unsafe_allow_html=True)

if selected == "Datasets" :
    f = open("datasets/trump_archive.json",  encoding="mbcs")
    data = json.load(f)
    df = pd.read_csv("datasets/stackoverflow.csv", encoding="mbcs")

    st.title(f"About our Datasets 📊")
    st.write("We are considering 3 popular datasets for our research!")

    st.header("Trump Twitter Archive Dataset")
    text = "The former US president <span class='highlight'>Donald Trump</span> was <span class='highlight'>notoriously active on Twitter</span>. On January 8th, 2021, the platform decided to suspend his account, citing 'the risk of further incitement of violence' following the violent riots at the US Capitol building on Jan 6th. <span class='highlight'>Trump's Twitter activity</span> constitutes an <span class='highlight'>important documentation of escalating polarisation</span> in the US political and societal discourse during the second decade of the 2000s."
    text1 = "This dataset contains Trump's tweets since November 2019 to December 2019. It has a total of 931 samples. It was <span class='highlight'>copied from the website 'The Trump Archive'</span> who did all the work in periodically scraping Trump's Twitter account until his suspension in 2021."
    st.write(text, unsafe_allow_html=True)
    st.write(text1, unsafe_allow_html=True)
    with st.expander("View Trump Twitter Archive Dataset"):
        st.json(data["trump_tweets"])

    st.header("Stackoverflow Dataset")
    text = "<span class='highlight'>Stack Overflow</span> is the <span class='highlight'>largest online community for programmers</span> to learn, share their knowledge, and advance their careers."
    text1 = """
        This is a dataset containing <span class='highlight'>1,000 Stack Overflow questions</span>. Questions are classified into three categories:<br/>
        <span class='highlight'>HQ</span>: High-quality posts without a single edit.<br/>
        <span class='highlight'>LQ_EDIT</span>: Low-quality posts with a negative score, and multiple community edits. However, they still remain open after those changes.<br/>
        <span class='highlight'>LQ_CLOSE</span>: Low-quality posts that were closed by the community without a single edit.<br/><br/>
        However, for STTM, we are only interested in the <span class='highlight'>'Title' column</span> of this dataset.
        """
    st.write(text, unsafe_allow_html=True)
    st.write(text1, unsafe_allow_html=True)
    with st.expander("View Stackoverflow Dataset"):
        st.dataframe(df)

    df = pd.read_csv("datasets/abcnews.csv", encoding="mbcs")
    st.header("ABC News Headlines Dataset")
    text = "This contains data of <span class='highlight'>news headlines</span> published over a period of years. Sourced from the reputable <span class='highlight'>Australian news source ABC (Australian Broadcasting Corporation)</span>"
    text1 = """
        This includes the entire <span class='highlight'>corpus of articles published by the abcnews website</span>.<br/>
        With a volume of two hundred articles per day and a good focus on international news, we can be <span class='highlight'>fairly certain that every event of significance has been captured in this dataset</span>.
        Digging into the keywords, one can see all the important episodes shaping the last decade and how they evolved over time.<br/>
        We have selected the <span class='highlight'>top 1000 headline samples</span> from this dataset to run our algorithm.
        """
    st.write(text, unsafe_allow_html=True)
    st.write(text1, unsafe_allow_html=True)
    with st.expander("View ABC News Headlines Dataset"):
        st.dataframe(df)

if selected == "GSDMM" :
    st.title(f"What is GSDMM? 🤔")
    text = "GSDMM <span class='highlight'>(Gibbs Sampling Dirichlet Multinomial Mixture)</span> is a short text clustering model proposed by Jianhua Yin and Jianyong Wang in a paper a few years ago. The model claims to <span class='highlight'>solve the sparsity problem of short text clustering</span> while also <span class='highlight'>displaying word topics like LDA</span>."
    text1 = """
        GSDMM can <span class='highlight'>infer the number of clusters automatically</span> with a good balance between the <span class='highlight'>completeness and homogeneity</span> of the clustering results, and is fast to converge.
        The <span class='highlight'>basic principle</span> of GSDMM is described using an analogy called <span class='highlight'>'Movie Group Approach'</span>.
        """
    text2 = """
        Imagine a <span class='highlight'>group of students (documents)</span> who all have a <span class='highlight'>list of favorite movies (words)</span>. The students are randomly assigned to <span class='highlight'>K tables</span>.
        At the instruction of a professor the students must shuffle tables with 2 goals in mind:<br/><br/>
        -> <span class='highlight'>Find a table with more students</span>.<br/>
        -> <span class='highlight'>Pick a table where your film interests align with those at the table</span>.<br/>
        -> <span class='highlight'>Rinse and repeat until you reach a plateau where the number of clusters does not change</span>.<br/>
        """
    st.write(text, unsafe_allow_html=True)
    st.write(text1, unsafe_allow_html=True)
    st.header("The Movie Group Process (MGP)")
    st.write(text2, unsafe_allow_html=True)
    st.header("Influence of parameters in GSDMM")
    with st.expander("α and β", expanded=True):
        st.image("img/param.png")
    with st.expander("Analogy of Influence of α and β", expanded=True):
        st.image("img/influence.png")

if selected == "BERTopic" :
    st.title(f"What is BERTopic? 🧐")
    text = "BERTopic is a topic modeling technique that <span class='highlight'>leverages BERT embeddings and c-TF-IDF</span> to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic descriptions.<br/>"
    text1 = "The <span class='highlight'>two greatest advantages</span> to BERTopic are arguably its <span class='highlight'>straight forward out-of-the-box usability</span> and its <span class='highlight'>novel interactive visualization methods<span>.<br/>"
    text2 = "Having an overall picture of the topics that have been learned by the model allows us to generate an internal perception of the model's quality and the most notable themes encapsulated in the corpus."

    st.write(text, unsafe_allow_html=True)
    st.write(text1, unsafe_allow_html=True)
    st.write(text2, unsafe_allow_html=True)

    st.header("Stages involved in Topic Modelling with BERTopic")
    with st.expander("View the Stages", expanded=True):
        st.image("img/bert.png")

if selected == "Run GSDMM" :
    st.title(f"Check out GSDMM Yourself! 👨‍💻")
    np.random.seed(493)


    DATASETS = {
        'Trump Tweets': ('trump_tweets.csv', 'text'),
        'Stackover Flow Questions': ('stackoverflow.csv', 'Title'),
        'ABC News Dataset': ('abcnews-date-text.csv', 'headline_text')
    }


    nltk.download('stopwords')
    nltk.download('wordnet')
    ps = nltk.porter.PorterStemmer()


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

    with st.form(key="my_form"):

        # ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
        # with c1:
        Dataset = st.selectbox(
            'Which dataset would you like to use?',
            tuple(DATASETS.keys())
        )

        alpha = st.slider(
            "Set alpha α ",
            min_value=0.1,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="_______",
        )
        beta = st.slider(
            "Set beta β",
            min_value=0.1,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help="_______",
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            iterations = st.number_input(
                "Number of Iterations:",
                value=30,
                step=5,
                min_value=20,
                max_value=80,
                help="""_______""",
                # help="Minimum value for the keyphrase_ngram_range. keyphrase_ngram_range sets the length of the resulting keywords/keyphrases. To extract keyphrases, simply set keyphrase_ngram_range to (1, # 2) or higher depending on the number of words you would like in the resulting keyphrases.",
            )

        with col2:
            k = st.number_input(
                "Adjust k:",
                value=15,
                min_value=10,
                max_value=50,
                step=5,
                help="""_______""",
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
    nltk.download('punkt')


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

    top_index = doc_count.argsort()[-15:][::-1]

    list_of_word_clouds = []


    def word_multiply(word, count):
        s = ""
        for _ in range(count):
            s += word + " "
        return s

    # def top_words(cluster_word_distribution, top_cluster, values):
    #     for cluster in top_cluster:
    #         sort_dicts = sorted(mgp.cluster_word_distribution[cluster].items(
    #         ), key=lambda k: k[1], reverse=True)[:values]

    def top_words(cluster_word_distribution, top_cluster, values):
        for cluster in top_cluster:
            sort_dicts = sorted(
                mgp.cluster_word_distribution[cluster].items(),
                key=lambda k: k[1],
                reverse=True,
            )[:values]
            frequencies = {}
            for i in sort_dicts :
                frequencies[i[0]] = i[1]
            if len(frequencies) != 0:
                wordcloud = WordCloud(
                    width=800, height=800, background_color="white", min_font_size=10
                ).generate_from_frequencies(frequencies)
                list_of_word_clouds.append(wordcloud)

    cols = st.columns(3)
    top_words(mgp.cluster_word_distribution, top_index, 100)
    for index, word_cloud in enumerate(list_of_word_clouds):        
        with cols[index % 3]:            
            st.image(word_cloud.to_image(), caption=f'Topic #{index}')

    topic_dict = {}
    topic_names = ['Topic #0',
                'Topic #1',
                'Topic #2',
                'Topic #3',
                'Topic #4',
                'Topic #5',
                'Topic #6',
                'Topic #7',
                'Topic #8',
                'Topic #9',
                'Topic #10',
                'Topic #11',
                'Topic #12',
                'Topic #13',
                'Topic #14',
                ]
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
    st.dataframe(dfx)


if selected == "Run BERTopic" :
    st.title(f"Check out BERTopic Yourself! 👨‍💻")
    DATASETS = {
        'Trump Tweets': ('trump_tweets.csv', 'text'),
        'Stackover Flow Questions': ('stackoverflow.csv', 'Title'),
        'ABC News Dataset': ('abcnews-date-text.csv', 'headline_text')
    }

    MODELS = {
        'BERTopic': 'bert-base-uncased',
    }

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

        # StopWordsCheckbox = st.checkbox(
        #     "Remove stop words",
        #     value=True,
        #     help="Tick this box to remove stop words from the dataset (currently English only)",
        # )

        # title = st.text_input('Enter a keyword to search for:', '')

        ModelType = st.radio(
            "",
            MODELS.keys(),
            
        )

        @st.cache(allow_output_mutation=True)
        def load_model():
            return bertopic.BERTopic(
                calculate_probabilities=True
            )

        model = load_model()

        # col1, col2 = st.columns([1,1], gap='small')

        


        submit_button = st.form_submit_button(label="✨ Run BERTopic!")

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

    topic_hierarchy_visual = model.visualize_hierarchy()

    # topic_distribution_visual = model.visualize_distribution(probs)

    topic_similarity_visual = model.visualize_heatmap(width=1000, height=1000)

    topic_c_tf_idf_scores_visual = model.visualize_barchart()

    term_rank_visual = model.visualize_term_rank()

    st.markdown("## 🎈 Check & download results ")

    st.header("")


    tab1, tab2, tab3, tab4, tab5, tab6= st.tabs(
        ["Topics", "Coherence", "c-TF-IDF Scores", "Topic Hierarchies", "Term Rank", "Topic Similarities"])

    with tab1:
        st.header("Topics")
        CSVButton = download_button(all_topics, "topics.csv", "📥 Download (.csv)")
        st.dataframe(all_topics)

    with tab2:
        st.header("Coherence")
        st.plotly_chart(coherence_visual, use_container_width=True)

    with tab3:
        st.header("c-TF-IDF Scores")
        st.plotly_chart(topic_c_tf_idf_scores_visual, use_container_width=True)

    with tab4:
        st.header("Topic Hierarchy")
        st.plotly_chart(topic_hierarchy_visual, use_container_width=True)

    with tab5:
        st.header("Term Rank Decline")
        st.plotly_chart(term_rank_visual, use_container_width=True)

    with tab6:
        st.header("Topic Similarities")
        st.plotly_chart(topic_similarity_visual, use_container_width=True)

if selected == "Run KeyBERT" :
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

    st.title(f"Check out KeyBERT Yourself! 👨‍💻")
    with st.form(key="my_form"):
        ModelType = st.radio(
            "Choose your model",
            ["DistilBERT (Default)", "Flair"],
            help="At present, you can choose between 2 models (Flair or DistilBERT) to embed your text. More to come!",
        )

        if ModelType == "Default (DistilBERT)":
            # kw_model = KeyBERT(model=roberta)

            @st.cache(allow_output_mutation=True)
            def load_modell():
                return KeyBERT(model=roberta)

            kw_model = load_modell()

        else:
            @st.cache(allow_output_mutation=True)
            def load_modell():
                return KeyBERT("distilbert-base-nli-mean-tokens")

            kw_model = load_modell()

        top_N = st.slider(
            "# of results",
            min_value=1,
            max_value=5,
            value=3,
            help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
        )
        min_Ngrams = st.number_input(
            "Minimum Ngram",
            min_value=1,
            max_value=6,
            help="""The minimum value for the ngram range.

*Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.

To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
            # help="Minimum value for the keyphrase_ngram_range. keyphrase_ngram_range sets the length of the resulting keywords/keyphrases. To extract keyphrases, simply set keyphrase_ngram_range to (1, # 2) or higher depending on the number of words you would like in the resulting keyphrases.",
        )

        max_Ngrams = st.number_input(
            "Maximum Ngram",
            value=2,
            min_value=1,
            max_value=6,
            help="""The maximum value for the keyphrase_ngram_range.

*Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.

To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
        )

        StopWordsCheckbox = st.checkbox(
            "Remove stop words",
            help="Tick this box to remove stop words from the document (currently English only)",
        )

        use_MMR = st.checkbox(
            "Use MMR",
            value=True,
            help="You can use Maximal Margin Relevance (MMR) to diversify the results. It creates keywords/keyphrases based on cosine similarity. Try high/low 'Diversity' settings below for interesting variations.",
        )

        Diversity = st.slider(
            "Keyword diversity (MMR only)",
            value=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            help="""The higher the setting, the more diverse the keywords.
            
Note that the *Keyword diversity* slider only works if the *MMR* checkbox is ticked.

""",
        )

        doc = st.text_area(
            "Paste your text below (max 100 words)",
            height=200,
        )

        MAX_WORDS = 100
        import re
        res = len(re.findall(r"\w+", doc))
        if res > MAX_WORDS:
            st.warning(
                "⚠️ Your text contains "
                + str(res)
                + " words."
                + " Only the first 500 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
            )

            doc = doc[:MAX_WORDS]

        submit_button = st.form_submit_button(label="✨ Get me the topic!")

        if use_MMR:
            mmr = True
        else:
            mmr = False

        if StopWordsCheckbox:
            StopWords = "english"
        else:
            StopWords = None

    if not submit_button:
        st.stop()

    if min_Ngrams > max_Ngrams:
        st.warning("min_Ngrams can't be greater than max_Ngrams")
        st.stop()

    keywords = kw_model.extract_keywords(
        doc,
        keyphrase_ngram_range=(min_Ngrams, max_Ngrams),
        use_mmr=mmr,
        stop_words=StopWords,
        top_n=top_N,
        diversity=Diversity,
    )

    st.markdown("## **🎈 Check & download results **")

    st.header("")

    cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

    with c1:
        CSVButton2 = download_button(keywords, "Data.csv", "📥 Download (.csv)")
    with c2:
        CSVButton2 = download_button(keywords, "Data.txt", "📥 Download (.txt)")
    with c3:
        CSVButton2 = download_button(keywords, "Data.json", "📥 Download (.json)")

    st.header("")

    df = (
        DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
        .sort_values(by="Relevancy", ascending=False)
        .reset_index(drop=True)
    )

    df.index += 1

    # Add styling
    cmGreen = sns.light_palette("green", as_cmap=True)
    cmRed = sns.light_palette("red", as_cmap=True)
    df = df.style.background_gradient(
        cmap=cmGreen,
        subset=[
            "Relevancy",
        ],
    )

    c1, c2, c3 = st.columns([1, 3, 1])

    format_dictionary = {
        "Relevancy": "{:.1%}",
    }

    df = df.format(format_dictionary)

    with c2:
        st.table(df)

if selected == "References" :
    st.title(f"References 📝")
    
    with st.expander("Topic Modelling Reference Paper", expanded=False):
        def show_pdf(file_path):
            with open(file_path,"rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

        show_pdf('papers/topic_modelling_paper.pdf')
    
    with st.expander("GSDMM Reference Paper", expanded=False):
        def show_pdf(file_path):
            with open(file_path,"rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

        show_pdf('papers/gsdmm_paper.pdf')

    with st.expander("BERTopic Reference Paper", expanded=False):
        def show_pdf(file_path):
            with open(file_path,"rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

        show_pdf('papers/bertopic_paper.pdf')