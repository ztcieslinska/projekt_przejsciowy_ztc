from requirements import *
from lda_functions import *

base_df = pd.read_pickle("pickle_data_lemm.pickle")
stops = list(ENGLISH_STOP_WORDS)
stops.extend(['car', 'vehicle', 'electric', 'say', 'said', 'like',
              'new', 'ha', 'wa', 'year', 'reuters'])
n_top = 5

# rozważałam dodatkowe uproszczenie, połączenie wszystkich tech i
# wszystkich business, ale mozliwe, że stracę istotne informacje
base_df['section'] = base_df['section'].str.replace("news", "").str.strip()
base_df['section'] = base_df['section'].str.replace("technology", "tech", regex=True)


############# podstawowa analiza ################

# oszczędność obliczeniowa poprzez zapisywanie wyniku i odczytywanie go
df = get_lda(base_df, stops, num_topics=n_top)
df.to_pickle('lda_topickled_'+str(n_top)+'.pickle')
df = pd.read_pickle('lda_topickled_'+str(n_top)+'.pickle')

# wybieram najpopularniejsze działy
df_freq_sec = df.loc[df.groupby('section')['section'].transform('size') >= 100]

# normalizacja względem kolumn - które tematy są najpopularniejsze w danym dziale
topic_distro_big = pd.crosstab(df_freq_sec['topic_words'], df_freq_sec['year'], normalize='columns')

# czy ludzie interesujący się konkretnym działem nigdy nie ujrzą jakiegoś tematu?
sns.set(rc={'figure.figsize': (15, 4)})
heatmap = sns.heatmap(topic_distro_big, annot=False, cmap='magma_r')
fig = heatmap.get_figure()
fig.set_tight_layout(True)
fig.savefig('topic_in_sections_'+str(n_top)+'.png')


############# wyniki LDA w podziale na sekcje ################

# wybieram najczęstsze działy - widzę, że powyżej 300 wsytąpień jest rozsądną granicą
sections = base_df['section'].value_counts()
# ponad 2/3 artykułów pochodziło z tych najpopularniejszych sekcji.
df_freq_sec = base_df.loc[base_df.groupby('section')['section'].transform('size') >= 300]

# jakie są najczęstsze tematy w każdym dziale?
sections_dict = {}
for section, df in df_freq_sec.groupby('section'):
    print("Calculating LDA for section " + section)
    s_df = get_lda(df, stops, num_topics=3)
    themes = s_df['topic_words'].value_counts()
    sections_dict[section] = themes

    # widać, że warto, by były oddzielne stopwords w każdym temacie.

# w jakich latach występowały jakie tematy?
years_dist = {}
for section, df in df_freq_sec.groupby('section'):
    print("Calculating LDA for section " + section)
    s_df = get_lda(df, stops, num_topics=3)
    topic_distro = pd.crosstab(s_df['topic_words'], s_df['year'])
    years_dist[section] = topic_distro


# o czym się mówiło w mniejszych wydawnictwach?
df_small_sec = base_df.loc[base_df.groupby('section')['section'].transform('size') < 300]
df_lda_small = get_lda(df_small_sec, stops, num_topics=n_top)
# w podziale na lata:
topic_distro_small = pd.crosstab(df_lda_small['topic_words'], df_lda_small['year'], normalize='columns')
