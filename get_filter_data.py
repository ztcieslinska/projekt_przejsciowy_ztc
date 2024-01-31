from requirements import *

# https://components.one/datasets/all-the-news-2-news-articles-dataset/
filename = "all-the-news-2-1.csv"
key_phrases = ['electric car', 'electric vehicle']
all_data = []

# ze względu na rozmiar danych, czytanie i filtrowanie po interesujących artykułach odbywa się w porcjach:
for chunk in tqdm(pd.read_csv(filename, chunksize=100000)):
    filter_condition = chunk['article'].str.contains('|'.join(key_phrases), case=False, na=False)
    interesting_articles = chunk[filter_condition]
    all_data.append(interesting_articles)
    print(len(interesting_articles))

df = pd.concat(all_data, ignore_index=True)
df.to_pickle("pickle_data.pickle")

df = pd.read_pickle("pickle_data.pickle")

# sprowadzenie słów do ich prostszych postaci
# lemmatyzacja za pomocą spacy jest znacznie wolniejsza, ale NLTK nie działało.
print("Lemmatizing articles...")
df['lemmatized_article'] = df['article'].str.translate(str.maketrans('', '', string.punctuation + '”’'))\
    .str.lower()\
    .progress_apply(nlp)

df['lemmatized_article'] = df['lemmatized_article'].progress_apply(lambda doc: ' '.join([token.lemma_ for token in doc]))

# sprowadzam wszystkie działy do małej litery
df['section'] = df['section'].str.lower().fillna("no section")
df['section'] = df['section'].str.replace("news", "").str.strip()
df['section'] = df['section'].str.replace("technology", "tech", regex=True)

df.to_pickle("pickle_data_lemm.pickle")
