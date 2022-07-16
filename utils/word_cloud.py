import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
        just_words = [word_multiply(ele[0], ele[1]) for ele in sort_dicts if ele[1] > 0]
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
