# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import logging
import multiprocessing
import re
from zipfile import ZipFile

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (Activation, Conv1D, Dense, Dropout,
                                     Embedding, GlobalMaxPooling1D, Input,
                                     concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from config import DATA_DIR

archive_name = '12.7.zip'
archive_path = DATA_DIR.joinpath('external').joinpath(archive_name)
path_positive = '12.7/positive.csv'
path_negative = '12.7/negative.csv'

# =============================================================================
# best_tree_size = min(scores, key=scores.get)
# =============================================================================

names = 'id date name text typr rep rtw faw stcount foll frien listcount'.split()

with ZipFile(archive_path).open(path_positive) as f:
    kwargs = {
        'filepath_or_buffer': f,
        'sep': ';',
        'header': None,
        'names': names,
        'usecols': ['text']
    }
    df_positive = pd.read_csv(**kwargs)

print(df_positive.head())

with ZipFile(archive_path).open(path_negative) as f:
    kwargs = {
        'filepath_or_buffer': f,
        'sep': ';',
        'header': None,
        'names': names,
        'usecols': ['text']
    }
    df_negative = pd.read_csv(**kwargs)

print(df_negative.head())

sample_size = min(df_positive.shape[0], df_negative.shape[0])

raw_data = np.concatenate(
    (
        df_positive['text'].values[:sample_size],
        df_negative['text'].values[:sample_size]
    ),
    axis=0
)
labels = [1] * sample_size + [0] * sample_size


def preprocessing_text(text):
    text = text.lower().replace('ё', 'е')
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    text = re.sub('@[^\s]+', 'USER', text)
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    return re.sub(' +', ' ', text).strip()


data = [preprocessing_text(t) for t in raw_data]

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=.2, random_state=42)


NUM_WORDS = 100_000
SENTENCE_LENGTH = 50


def get_sequences(tokenizer, x):
    sequences = tokenizer.texts_to_sequences(x)
    return pad_sequences(sequences, maxlen=SENTENCE_LENGTH)


tokenizer = Tokenizer(NUM_WORDS)
tokenizer.fit_on_texts(X_train)

X_train_seq = get_sequences(tokenizer, X_train)
X_test_seq = get_sequences(tokenizer, X_test)

# =============================================================================
# ?
# =============================================================================
logging.basicConfig(format='%(asctime)s: %(levelname)s', level=logging.INFO)

model = Word2Vec(data, vector_size=200, window=5, min_count=3,
                 workers=multiprocessing.cpu_count())
model.save('modeltweet.w2v')
print(model.wv.most_similar('хорошо'))

w2v_model = Word2Vec.load('modeltweet.w2v')
DIM = w2v_model.vector_size

embedding_matrix = np.zeros((NUM_WORDS, DIM))

for word, i in tokenizer.word_index.items():
    if i >= NUM_WORDS:
        break
    if word in w2v_model.wv.vocab.keys():
        embedding_matrix[i] = w2v_model.wv[word]

tweet_input = Input(shape=(SENTENCE_LENGTH,), dtype='int32')
tweet_encoder = Embedding(NUM_WORDS, DIM, input_length=SENTENCE_LENGTH,
                          weights=[embedding_matrix], trainable=False)(tweet_input)

branches = []
x = Dropout(.2)(tweet_encoder)
for size, filter_count in ((2, 10), (3, 10), (4, 10), (5, 10)):
    for i in range(filter_count):
        branch = Conv1D(filters=1, kernel_size=size,
                        padding='valid', activation='relu')(x)
        branch = GlobalMaxPooling1D()(branch)
        branches.append(branch)

x = concatenate(branches, axis=1)
x = Dropout(.2)(x)
x = Dense(30, activation='relu')(x)
x = Dense(1)(x)
output = Activation('sigmoid')(x)

model = Model(inputs=tweet_input, outputs=[output])
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(np.array(X_train_seq), y_train, batch_size=32,
                    epochs=10, validation_split=.25, verbose=1)
