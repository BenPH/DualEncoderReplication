import os
import pickle

import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

DATA_DIRECTORY = '../data/csv'
PROCESSED_DIRECTORY = '../data/processed'
MAX_SEQUENCE_LENGTH = 100

def get_filtered_length_map(df, max_length):
    """ Returns map where rows with sequences longer than `max_length`
    are filtered out.

    Parameters
    df: Dataframe
        The dataframe to filter.
    max_length: int
        The maximum length of sequence desired.
    """
    return (df.apply(lambda seq: seq.str.split().apply(len)) <= max_length).all(axis=1)


def build_tokenizer(texts_df, count_threshold=7):
    """ Builds a keras tokenizer from the texts in the training data
    and removes infrequent words.

    Parameters
    train_df: Dataframe
        The training data.
    count_threshold: int
        The minimum number of times a word must appear
        in the data for it to be included in the vocabulary.
    """
    tokenizer = Tokenizer(filters='', lower=False, oov_token='__oov__')
    for col in texts_df.columns:
        tokenizer.fit_on_texts(texts_df[col])

    # Removing the least frequent words from word_index
    low_count_words = [w for w, c in tokenizer.word_counts.items() if c < count_threshold]
    for w in low_count_words:
        del tokenizer.word_index[w]
        del tokenizer.word_docs[w]
        del tokenizer.word_counts[w]

    return tokenizer


def preprocess_data(df, tokenizer, train_data=False):
    """ Transforms the data into padded, tokenized sequences.
    Returns tuple of numpy arrays of the context, utterance, and labels.

    Parameters
    df: Dataframe
        The dataframe to process.
    tokenizer: Tokenizer
        Tokenizer used to transform the texts to sequences.
    train_data: boolean
        Whether the dataframe is training or testing data.
    """
    if train_data:
        labels = df['Label']
        df = df[df.columns[:-1]]
    filtered_map = get_filtered_length_map(df, MAX_SEQUENCE_LENGTH)
    df = df[filtered_map]

    tokenized_texts = df.apply(lambda text: tokenizer.texts_to_sequences(text))
    padded_sequences = tokenized_texts.apply(lambda row: pad_sequences(row, MAX_SEQUENCE_LENGTH), axis=1).to_numpy()

    if train_data:
        contexts = np.asarray([row[0] for row in padded_sequences])
        utterances = np.asarray([row[1] for row in padded_sequences])
        labels = labels[filtered_map].to_numpy()
    else:
        contexts = np.asarray([row[0] for row in padded_sequences]).repeat(10, axis=0)
        utterances = np.asarray([row[1:] for row in padded_sequences]).reshape(-1, MAX_SEQUENCE_LENGTH)
        labels = np.tile([1] + [0] * 9, df.shape[0]) # Ground truth is followed by 9 false examples

    return contexts, utterances, labels


train = pd.read_csv(os.path.join(DATA_DIRECTORY, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIRECTORY, 'test.csv'))
valid = pd.read_csv(os.path.join(DATA_DIRECTORY, 'valid.csv'))

tokenizer = build_tokenizer(train[train.columns[:-1]], count_threshold=7)

pickle.dump(preprocess_data(train, tokenizer, train_data=True), \
    open(os.path.join(PROCESSED_DIRECTORY, 'train.pkl'), 'wb'))
pickle.dump(preprocess_data(test, tokenizer), \
    open(os.path.join(PROCESSED_DIRECTORY, 'test.pkl'), 'wb'))
pickle.dump(preprocess_data(valid, tokenizer), \
    open(os.path.join(PROCESSED_DIRECTORY, 'valid.pkl'), 'wb'))
pickle.dump([MAX_SEQUENCE_LENGTH, tokenizer.word_index], \
    open(os.path.join(PROCESSED_DIRECTORY, 'params.pkl'), 'wb'))
