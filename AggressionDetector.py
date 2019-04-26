import csv

import keras
import nltk
import numpy as np
from gensim.models import KeyedVectors
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.ensemble import RandomForestRegressor

from Utils import f1, idf, ncce, evaluate

# Configure to change language
lang = "hi"

def load_data(path):
    rows = []
    with open(path, newline="\n", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="\"")
        for row in reader:
            rows.append(DataRow(row[1], row[2]))

    return rows


class DataRow:

    def __init__(self, text, label):
        if not (label == "NAG" or label == "CAG" or label == "OAG"):
            raise ValueError(f"label must be 'NAG', 'CAG' or 'OAG', but was {label}")

        self.text = text
        self.label = label
        self.words = [w for w in nltk.word_tokenize(self.text)]
        if label == "NAG":
            self.label = 0
        elif label == "CAG":
            self.label = 1
        else:
            self.label = 2

    def embeddings(self, model, idf):
        embed_words = [w for w in self.words if w in model.vocab]
        if len(embed_words) < 1:
            return np.zeros(300)
        else:
            embeddings = []
            for w in embed_words:
                embeddings.append(model.get_vector(w) * idf[w])

            summed = np.array(embeddings).sum(axis=0)
            return summed


print("Loading training data ...")
train = load_data(f"data/agr_{lang}_train.csv")
test = load_data(f"data/agr_{lang}_dev.csv")

print("Loading model ...")
embedding_model = KeyedVectors.load_word2vec_format(f"models/cc.{lang}.300.vec", limit=500000)
embedding_dim = 300

print("Loading IDF scores ...")
idf = idf(train)

# Fetch embeddings for the rows in the data set
X_train = np.array([row.embeddings(embedding_model, idf) for row in train])
Y_train = keras.utils.to_categorical(([e.label for e in train]), 3)
X_test = np.array([row.embeddings(embedding_model, idf) for row in test])
Y_test = keras.utils.to_categorical(([e.label for e in test]), 3)


def create_and_train_network():
    print("Training neural network ...")
    m = Sequential()
    m.add(Dense(120, input_shape=(embedding_dim,), activation='sigmoid'))
    m.add(Dropout(0.5))
    m.add(Dense(3, activation='softmax'))
    m.compile(loss=ncce, optimizer=keras.optimizers.Adam(0.001), metrics=[f1])
    m.fit(X_train, Y_train, epochs=50, batch_size=1024, verbose=0, shuffle=True)
    return m


def create_and_train_forest():
    print("Training random forest ...")
    rf = RandomForestRegressor(n_estimators=50, max_features='auto', max_depth=10, min_samples_split=5,
                               min_samples_leaf=4)
    rf.fit(X_train, Y_train)
    return rf


random_forest = create_and_train_forest()
neural_network = create_and_train_network()

evaluate(random_forest, X_test, Y_test)
evaluate(neural_network, X_test, Y_test)
