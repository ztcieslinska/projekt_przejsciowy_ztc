from requirements import *

df = pd.read_pickle("pickle_data_lemm.pickle")

# domyślnie usunięte stopwords z dodanymi słowami specyficznymi dla zbioru
stops = list(ENGLISH_STOP_WORDS)
stops.extend(['say', 'said', 'will',
              'electric', 'car', 'vehicle',
              'make', 'company', 'new', 'year'])

def get_wordcloud(articles_series, filename):
    long_string = ','.join(list(articles_series.values))
    wordcloud = WordCloud(background_color="white",
                          max_words=8000,
                          colormap='plasma',
                          stopwords=stops,
                          scale=4,
                          collocations=False,
                          random_state=42
                          )
    wordcloud.generate(long_string)
    wordcloud.to_file(filename)


get_wordcloud(df['lemmatized_article'], "all_articles_wordcloud.png")

# jakie działy pojawiają się na stronach? jacy wydawcy?
sections = df['section'].value_counts()
sections = sections[sections>10]
publishers = df['publication'].value_counts()

# co się pojawiało w temacie biznesu, a co w technologii?
df_mod = df[df['section']=='business']
get_wordcloud(df_mod['lemmatized_article'], "business_wordcloud.png")

df_mod = df[(df['section']=='technology')]
get_wordcloud(df_mod['lemmatized_article'], "technology_wordcloud.png")

# co dominowało w roku 2016, a co w 2019?
df_mod = df[df['year']==2016]
get_wordcloud(df_mod['lemmatized_article'], "2016_wordcloud.png")

df_mod = df[df['year']==2019]
get_wordcloud(df_mod['lemmatized_article'], "2019_wordcloud.png")

# technologia: co dominowało w roku 2016, a co w 2019?
df_tech = df[df['section']=='tech']
df_mod = df_tech[df_tech['year']==2016]
get_wordcloud(df_mod['lemmatized_article'], "2016_tech_wordcloud.png")

df_mod = df_tech[df_tech['year']==2019]
get_wordcloud(df_mod['lemmatized_article'], "2019_tech_wordcloud.png")