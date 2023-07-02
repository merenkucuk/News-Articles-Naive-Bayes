import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# read the file with pandas library
df = pd.read_csv('English Dataset.csv')
# make dataframe to numpy array
df = df.to_numpy()
# shuffle the numpy array
np.random.shuffle(df)
# split the data as percentage 8 / 2 to train and test set
# the percentage of split is discussed on piazza and TA say yes to 80, 20
train, test = train_test_split(df, test_size=0.2)


# the function makes matrix from vector
# take vector as parameter
# we can use the function if np.sum axis 0 does not exist
# the TA can also use that for make matrix
# but program gives output after 5-6 min.
def makeMatrix(vector):
    _, W = vector.shape
    t = [[0] * W for k in range(1)]
    t = np.array(t)
    for i in range(W):
        column = np.sum(vector[:, i])
        t[0][i] = column
    return t

# Naive Bayes Method
def naiveBayes(mode):
    # we firstly specify our ngram_range
    # if we select mode as unigram then ngram_range will be (1,1)
    # if we select mode as bigram then ngram_range will be (2,2)
    ngram = (0, 0)
    if mode == "unigram":
        ngram = (1, 1)
    elif mode == "bigram":
        ngram = (2, 2)
    elif mode == "uni-bigram":
        ngram = (1, 2)
    else:
        print("set valid mode")
    # we initialize CountVectorizer with ngram_mode
    vectorizer = CountVectorizer(ngram_range=ngram)
    # we initialize a vector which includes all words and their counts for each item
    train_vector = vectorizer.fit_transform(train[:, 1])
    # we make vector to array to use
    count_vector = train_vector.toarray()
    # we initialize a list which have all category and count
    categoryList = list(train[:, 2])
    # we initialize a list and variable for each category
    # to use on probability
    businessList, politicsList, techList, entList, sportList = [], [], [], [], [],
    bus, pol, tech, ent, spo = 0, 0, 0, 0, 0
    for i in categoryList:
        if i == "business":
            bus += 1
            businessList.append(True)
            politicsList.append(False)
            techList.append(False)
            entList.append(False)
            sportList.append(False)
        elif i == "politics":
            pol += 1
            businessList.append(False)
            politicsList.append(True)
            techList.append(False)
            entList.append(False)
            sportList.append(False)
        elif i == "tech":
            tech += 1
            businessList.append(False)
            politicsList.append(False)
            techList.append(True)
            entList.append(False)
            sportList.append(False)
        elif i == "entertainment":
            ent += 1
            businessList.append(False)
            politicsList.append(False)
            techList.append(False)
            entList.append(True)
            sportList.append(False)
        elif i == "sport":
            spo += 1
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
    # we initialize vectors for each category
    # the train_vector has all words and counts for each category
    # so we just take it from the train_vector for related category
    business_vector = train_vector[businessList]
    politics_vector = train_vector[politicsList]
    tech_vector = train_vector[techList]
    entertainment_vector = train_vector[entList]
    sport_vector = train_vector[sportList]
    # instead of a loop we can use this method
    # to get number of each category from sum of vector
    n_business = np.sum(business_vector)
    n_politics = np.sum(politics_vector)
    n_tech = np.sum(tech_vector)
    n_entertainment = np.sum(entertainment_vector)
    n_sport = np.sum(sport_vector)

    # bus_mtx = makeMatrix(business_vector)
    # pol_mtx = makeMatrix(politics_vector)
    # tech_mtx = makeMatrix(tech_vector)
    # ent_mtx = makeMatrix(entertainment_vector)
    # spo_mtx = makeMatrix(sport_vector)

    # we initialize shape of count vector for Laplace Smoothing
    _, D = count_vector.shape
    # we initialize alpha for Laplace Smoothing
    alpha = 1
    # we sum the column values of the vectors and get the matrix for each category
    # we make the progress of sum column values with axis = 0
    bus_mtx = np.sum(business_vector, axis=0)
    pol_mtx = np.sum(politics_vector, axis=0)
    tech_mtx = np.sum(tech_vector, axis=0)
    ent_mtx = np.sum(entertainment_vector, axis=0)
    spo_mtx = np.sum(sport_vector, axis=0)
    # apply laplace smoothing
    # we sum alpha and matrix values for each matrix and divide to sum of number of each category and count_vector shape
    # then, we take the log value of this process and we get matrix
    # lastly, we make the matrix to array
    bus_arr = (np.log((bus_mtx + alpha) / (n_business + D))).A1
    pol_arr = (np.log((pol_mtx + alpha) / (n_politics + D))).A1
    tech_arr = (np.log((tech_mtx + alpha) / (n_tech + D))).A1
    ent_arr = (np.log((ent_mtx + alpha) / (n_entertainment + D))).A1
    spo_arr = (np.log((spo_mtx + alpha) / (n_sport + D))).A1
    # we initialize test_vector from our test set to use on prediction
    test_vector = vectorizer.transform(test[:, 1])
    # we initialize a list which name is preds to store predict values.
    preds = []
    for i in test_vector:
        colList = []
        for _, col in zip(*i.nonzero()):
            colList.append(col)
        colList = np.array(colList)
        # we calculate our probability for each category
        allCategory = bus + pol + tech + ent + spo
        prob_business = np.log(bus / allCategory)
        prob_politics = np.log(pol / allCategory)
        prob_tech = np.log(tech / allCategory)
        prob_entertainment = np.log(ent / allCategory)
        prob_sport = np.log(spo / allCategory)
        # we calculate sum of arr colList for each category
        sum_business = bus_arr[colList].sum()
        sum_politics = pol_arr[colList].sum()
        sum_tech = tech_arr[colList].sum()
        sum_sport = spo_arr[colList].sum()
        sum_entertainment = ent_arr[colList].sum()
        # Then we added sums to the probability for each category
        prob_sport += sum_sport
        prob_tech += sum_tech
        prob_politics += sum_politics
        prob_business += sum_business
        prob_entertainment += sum_entertainment
        a = prob_entertainment
        b = prob_business
        c = prob_tech
        d = prob_politics
        e = prob_sport
        # then we initialize as variable the probabilities
        # and find the which is higher than others to append to prediction list as category to predict
        # finally, we find accuracy with test set and prediction list through accuracy score.
        if a > b and a > c and a > d and a > e:
            preds.append("entertainment")
        elif b > a and b > c and b > d and b > e:
            preds.append("business")
        elif c > a and c > b and c > d and c > e:
            preds.append("tech")
        elif d > a and d > b and d > c and d > e:
            preds.append("politics")
        elif e > a and e > b and e > c and e > d:
            preds.append("sport")

    return accuracy_score(test[:, 2], preds)


print("Accuracy is", naiveBayes("unigram"), "for mode unigram")
print("Accuracy is", naiveBayes("bigram"), "for mode bigram")
print("Accuracy is", naiveBayes("uni-bigram"), "for mode uni-bigram")
