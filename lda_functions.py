from requirements import *

def get_lda(df, stops, num_topics):
    # zbiór tekstów odczytany w Unicode
    corpus = df['lemmatized_article'].values.astype('U')

    # Reprezentacja bag-of-words korpusu (zliczone występowanie słów)
    vectorizer = CountVectorizer(stop_words=stops, lowercase=True)
    X = vectorizer.fit_transform(corpus)

    print("Fitting LDA model...")
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42, verbose=True)
    lda.fit(X)

    # Najczęstsze słowa w danych tematach
    n_top_words = 7
    feature_names = vectorizer.get_feature_names_out()
    topic_dict = get_top_words(lda, feature_names, n_top_words)

    # Dopasowanie najbardziej prawdopodobnego tematu do artykułu
    topic_distribution = lda.transform(X)
    df['topic'] = topic_distribution.argmax(axis=1)
    df["topic_words"] = df["topic"].map(topic_dict)

    return df

def get_top_words(model, feature_names, n_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict[topic_idx] = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return topic_dict
