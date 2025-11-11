import numpy as np 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_len = 200
max_words = 10000

def prepare_tokenizer(X_train):
    X_train = [str(x) for x in X_train]
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    return tokenizer

def preprocess_texts(X, tokenizer):
    X = [str(x) for x in X]
    sequences = tokenizer.texts_to_sequences(X)
    padded = pad_sequences(sequences, maxlen=max_len)
    return padded
