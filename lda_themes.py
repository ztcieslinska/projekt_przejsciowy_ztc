from requirements import *
from lda_functions import *

base_df = pd.read_pickle("pickle_data_lemm.pickle")
stops = list(ENGLISH_STOP_WORDS)
stops.extend(['car', 'vehicle', 'electric', 'say', 'said', 'like',
              'new', 'ha', 'wa', 'year', 'reuters'])
n_top = 5

def make_heatmap(crosstab, filename):
    plt.figure(figsize=(15, 5))
    sns.heatmap(crosstab, annot=False, cmap='magma_r')
    plt.tight_layout()
    plt.savefig(filename)

############# podstawowa analiza ################

# oszczędność obliczeniowa poprzez zapisywanie wyniku i odczytywanie go
df = get_lda(base_df, stops, num_topics=n_top)
df.to_pickle('lda_topickled_'+str(n_top)+'.pickle')
df = pd.read_pickle('lda_topickled_'+str(n_top)+'.pickle')

# wybieram najpopularniejsze działy
df_freq_sec = df.loc[df.groupby('section')['section'].transform('size') >= 300]

# normalizacja względem kolumn - które tematy są najpopularniejsze w danym dziale
topic_distro_big = pd.crosstab(df_freq_sec['topic_words'], df_freq_sec['section'], normalize='columns')

# różnice między większymi publikującymi a mniej popularnymi stronami
df_big_pub = df[(df['publication'].isin(["Reuters", "CNBC", "The New York Times", "The Verge", "TechCrunch"]))]
topic_year_all = pd.crosstab(df_big_pub['topic_words'], df_big_pub['year'], normalize="all")
make_heatmap(topic_year_all, "big_topics_year.png")

df_small_pub = df[~(df['publication'].isin(["Reuters", "CNBC", "The New York Times", "The Verge", "TechCrunch"]))]
topic_distro_pub = pd.crosstab(df_small_pub['topic_words'], df_small_pub['year'], normalize='all')
make_heatmap(topic_distro_pub, "small_topics_year.png")
# czy ludzie interesujący się konkretnym działem nigdy nie ujrzą jakiegoś tematu?

# co zawierają te konkretne tematy?
df_small_pub[df_small_pub['topic_words']=='make people company work just think time']['article'].to_csv("dziwny_temat.csv")

# jakie dokładniej podziały tematyczne można znaleźć wewnątrz tematów związanych z litem i z paliwem?
eco_df = df[df['topic_words'].isin(['make people company work just think time',
                                    'energy trump climate emission change oil state'])]
eco_df = eco_df.drop(columns=['topic',"topic_words"])
eco_lda_df = get_lda(eco_df, stops, num_topics=n_top)
topic_year_eco = pd.crosstab(eco_lda_df['topic_words'], eco_lda_df['year'], normalize="all")
make_heatmap(topic_year_eco, "eco_topics_year.png")


# wyniki poza głównym nurtem:
df_small_pub = df[~(df['publication'].isin(["Reuters", "CNBC", "The New York Times", "The Verge", "TechCrunch"]))]
    #.loc[df.groupby('section')['section'].transform('size') >= 300]
topic_distro_pub = pd.crosstab(df_small_pub['topic_words'], df_small_pub['section'], normalize='columns')
plt.figure(figsize=(15, 5))
sns.heatmap(topic_distro_pub, annot=False, cmap='magma_r')
plt.tight_layout()
plt.savefig(f'topic_section_small_pub_{n_top}.png')
plt.show()

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
    # oraz oddzielne liczby tematów

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


# dokładniejszy podział tematów w 2019 roku
df_2019 = base_df[base_df['year']==2019]
df_lda_2019 = get_lda(df_2019, stops, num_topics=n_top)
# w podziale na lata:
topic_distro_small = pd.crosstab(df_lda_small['topic_words'], df_lda_small['year'], normalize='columns')