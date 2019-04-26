import math
from functools import partial
from itertools import product

import keras.backend as K
import nltk
import numpy as np
from sklearn.metrics import f1_score


def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (K.cast(weights[c_t, c_p], K.floatx()) * K.cast(y_pred_max_mat[:, c_p], K.floatx()) * K.cast(
            y_true[:, c_t], K.floatx()))

    return K.categorical_crossentropy(y_true, y_pred) * final_mask


loss_weights = np.ones((3, 3))

loss_weights[2, 1] = 0.7
loss_weights[1, 2] = 0.7
ncce = partial(w_categorical_crossentropy, weights=loss_weights)


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + K.epsilon())

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def idf(documents):
    # Computation of TF
    df = nltk.defaultdict(lambda: 0)

    for row in documents:
        lower = [w for w in row.words]
        distinct_words = list(set(lower))
        for word in distinct_words:
            df[word] += 1


    # Computation of IDF
    n = len(documents)
    default_idf = 0.0
    idf = nltk.defaultdict(lambda: default_idf)
    for term, freq in df.items():
        idf[term] = math.log(n / (df[term] + 1))

    return idf

def evaluate(model, X_test, Y_test):
    def predict(model):
        Y_pred_probs = model.predict(X_test)
        Y_pred = np.zeros_like(Y_pred_probs)
        Y_pred[np.arange(len(Y_pred_probs)), Y_pred_probs.argmax(1)] = 1
        return Y_pred

    Y_pred = predict(model)
    score = f1_score(Y_test, Y_pred, average='weighted')
    print('F1-score: ' + str(score))
    return score

