from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
import numpy as np
import pandas as pd

def tokenize_document(document):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokenizer_ = RegexpTokenizer('[a-zA-Z]+')
    
    words = []
    for sentence in sent_tokenize(document):
        tokens = [lemmatizer.lemmatize(t.lower()) for t in tokenizer_.tokenize(sentence)\
                  if t.lower() not in stop_words]
        words += tokens
     
    words_ = str()
    for word in words:
        words_ = words_ + " " + word
    return words_


class DocTokenizer(BaseEstimator, TransformerMixin):
        
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        X = pd.Series(X)
        return X.apply(tokenize_document).tolist()

    
class WordsEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, top_words = 20000):
        self.top_words = top_words

    def fit(self, X, y=None, **fit_params):
        encoder = Tokenizer(self.top_words)
        encoder.fit_on_texts(X)
        self.encoder_ = encoder
        
        return self

    def transform(self, X, **transform_params):

        return self.encoder_.texts_to_sequences(X)
    
    
class Padder(BaseEstimator, TransformerMixin):
    def __init__(self, max_sequence_length = 500):
        self.max_sequence_length = max_sequence_length
    
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        return sequence.pad_sequences(np.array(X), maxlen = self.max_sequence_length)
