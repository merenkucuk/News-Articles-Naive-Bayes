import nltk
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from wordcloud import WordCloud

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("omw-1.4")


def read_factorize_data():
    data = pd.read_csv("English Dataset.csv")
    data["CategoryId"] = data["Category"].factorize()[0]
    return data


def plot_category_counts(data):
    data.groupby("Category").CategoryId.value_counts().plot(kind="bar", color=["pink", "brown", "red", "gray", "blue"])
    plt.xlabel("News Category")
    plt.title("Distribution of News by Category")
    plt.show()


def wordcloud_draw(dataset, category_name):
    print("Drawing words in category - " + category_name + "...")
    words = ' '.join(dataset)
    cleaned_word = ' '.join([word for word in words.split() if (word != "news" and word != "text")])
    wordcloud = WordCloud(stopwords=stop_w, background_color="white", width=2500, height=2500).generate(cleaned_word)
    plt.figure(1, figsize=(5.12, 5.12))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("WordCloud - " + category_name, fontdict={"fontsize": 24})
    plt.show()
    print("Drawing complete!")


def read_stopwords_from_txt(filename):
    gist_file = open(filename, "r")

    try:
        content = gist_file.read()
        stopwords_read = content.split(",")
    finally:
        gist_file.close()

    return set([i.replace('"', "").strip() for i in stopwords_read])


stop_w = stopwords.words("english")

df = read_factorize_data()
plot_category_counts(df)

category_names = df.Category.unique()

for i in range(0, 5):
    category_texts = df[df['CategoryId'] == i]["Text"]
    wordcloud_draw(category_texts, category_names[i].upper())
