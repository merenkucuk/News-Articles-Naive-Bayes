import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, ENGLISH_STOP_WORDS, TfidfVectorizer

# the function takes 3 parameters
# first parameter is category. "business", "politics" etc.
# second parameter is pres_or_abs. you should decide why to use presence or absence.
# third parameter is withStopWord. you should decide use StopWord or not.
def analyzeWithStopWords(category, pres_or_abs, withStopWord):
    print("\n" + category, pres_or_abs, withStopWord, "\n")
    # read the file with pandas library
    df = pd.read_csv('English Dataset.csv')
    # make dataframe to numpy array
    df = df.to_numpy()
    # we initialize a list which have all category and count
    categoryList = list(df[:, 2])
    # we initialize a list for each category
    # make for loop which indexes assigned to which category
    businessList, politicsList, techList, entList, sportList = [], [], [], [], [],
    for i in categoryList:
        if i == "business":
            businessList.append(True)
            politicsList.append(False)
            techList.append(False)
            entList.append(False)
            sportList.append(False)
        elif i == "politics":
            businessList.append(False)
            politicsList.append(True)
            techList.append(False)
            entList.append(False)
            sportList.append(False)
        elif i == "tech":
            businessList.append(False)
            politicsList.append(False)
            techList.append(True)
            entList.append(False)
            sportList.append(False)
        elif i == "entertainment":
            businessList.append(False)
            politicsList.append(False)
            techList.append(False)
            entList.append(True)
            sportList.append(False)
        elif i == "sport":
            businessList.append(False)
            politicsList.append(False)
            techList.append(False)
            entList.append(False)
            sportList.append(True)
    # make our lists to numpy array to be faster
    businessList = np.array(businessList)
    politicsList = np.array(politicsList)
    techList = np.array(techList)
    entList = np.array(entList)
    sportList = np.array(sportList)
    if category == "business":
        categoryList = businessList
    elif category == "politics":
        categoryList = politicsList
    elif category == "tech":
        categoryList = techList
    elif category == "entertainment":
        categoryList = entList
    elif category == "sport":
        categoryList = sportList
    ngram = (0, 0)
    if pres_or_abs == "presence":
        ngram = (1, 1)
    elif pres_or_abs == "absence":
        ngram = (2, 2)
    data = df[categoryList]
    # Now we are going to compute the IDF values by calling
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    # With Tfidfvectorizer you compute the word counts, idf and tf-idf values all at once
    vectorizer = TfidfVectorizer(use_idf=True)
    # then we decide use stop word or not
    if withStopWord == "withStopWord":
        vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, ngram_range=ngram)
    elif withStopWord == "withoutStopWord":
        vectorizer = TfidfVectorizer(use_idf=True, stop_words=ENGLISH_STOP_WORDS, smooth_idf=True, ngram_range=ngram)
    # this steps generates word counts for the words
    word_count_vector = vectorizer.fit_transform(data[:, 1])
    # get the all vector out
    tf_idf_vector = np.sum(word_count_vector, axis=0)
    # place tf-idf values in a pandas data frame
    df = pd.DataFrame(tf_idf_vector.A1, index=vectorizer.get_feature_names_out(), columns=["tfidf"])
    # list the scores for just first 10 words as you mention in pdf.
    df_category = df.sort_values(by=["tfidf"], ascending=False)
    return df_category.head(10)


print(analyzeWithStopWords("business", "presence", "withStopWord"))
print(analyzeWithStopWords("business", "presence", "withoutStopWord"))
print(analyzeWithStopWords("business", "absence", "withoutStopWord"))
print(analyzeWithStopWords("business", "absence", "withStopWord"))

print(analyzeWithStopWords("politics", "presence", "withStopWord"))
print(analyzeWithStopWords("politics", "presence", "withoutStopWord"))
print(analyzeWithStopWords("politics", "absence", "withoutStopWord"))
print(analyzeWithStopWords("politics", "absence", "withStopWord"))

print(analyzeWithStopWords("tech", "presence", "withStopWord"))
print(analyzeWithStopWords("tech", "presence", "withoutStopWord"))
print(analyzeWithStopWords("tech", "absence", "withoutStopWord"))
print(analyzeWithStopWords("tech", "absence", "withStopWord"))

print(analyzeWithStopWords("entertainment", "presence", "withStopWord"))
print(analyzeWithStopWords("entertainment", "presence", "withoutStopWord"))
print(analyzeWithStopWords("entertainment", "absence", "withoutStopWord"))
print(analyzeWithStopWords("entertainment", "absence", "withStopWord"))

print(analyzeWithStopWords("sport", "presence", "withStopWord"))
print(analyzeWithStopWords("sport", "presence", "withoutStopWord"))
print(analyzeWithStopWords("sport", "absence", "withoutStopWord"))
print(analyzeWithStopWords("sport", "absence", "withStopWord"))
