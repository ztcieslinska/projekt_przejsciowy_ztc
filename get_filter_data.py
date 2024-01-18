import pandas as pd
from tqdm import tqdm

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
df.to_pickle("pickled_data.pickle")