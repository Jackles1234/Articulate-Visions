import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class BagOfWordsTextEncoder():
    def __init__(self):
        self.total_words = 0
        self.idx_to_word = {}
        self.word_to_idx = {}
        self.all_words = set()

    def fit(self, text_array):
        for sentence in text_array:
            words = [word.lower() for word in sentence.split()]
            for word in words:
                # Add all words. Since its a set, there won't be duplicates.
                self.all_words.add(word)

        for idx, word in enumerate(self.all_words):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
            # Set the vocab size.
        self.total_words = len(self.all_words)

    def encode(self, text):
        if type(text) == str:
            encoded = self._transform_sentence(text.split())
        elif type(text) == list and type(text[0]) == str:
            encoded = np.empty((len(text), self.total_words))
            # Iterate over all sentences - this can be parallelized.
            for row, sentence in enumerate(text):
                # Substitute each row by the sentence BoW.
                encoded[row] = self._transform_sentence(sentence.split())
        else:
            raise TypeError(f"You must pass either a string or list of strings for transformation. type is {type(text)}")
        return encoded

    def _transform_sentence(self, list_of_words):
        transformed = np.zeros(self.total_words)
        for word in list_of_words:
            if word in self.all_words:
                word_idx = self.word_to_idx[word]
                transformed[word_idx] += 1
        return transformed


class TF_IDF:
    def __init__(self):
        self.tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')

    def fit(self, text_array):
        self.tfidf_wm = self.tfidfvectorizer.fit_transform(text_array)
        self.tfidf_tokens = self.tfidfvectorizer.get_feature_names()

    def encode(self, text):
        df_tfidfvect = pd.DataFrame(data=self.tfidf_wm.toarray(), index=['Doc1', 'Doc2'], columns=self.tfidf_tokens)
        return df_tfidfvect.to_numpy()
