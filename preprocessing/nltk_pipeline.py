from sklearn.base import BaseEstimator, TransformerMixin
import string
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pandas as pd
import numpy as np


def normalize_input(X):
    if isinstance(X, str):
        return [X]
    if X is None:
        raise TypeError("Input cannot be None.")
    if isinstance(X, (pd.Series, np.ndarray)):
        return X.astype(str).tolist()
    if isinstance(X, (list, tuple)):
        return [str(x) for x in X]
    if isinstance(X, dict):
        raise TypeError("Dictionaries are not valid input.")
    if hasattr(X, "__iter__"):
        try:
            return [str(x) for x in X]
        except:
            raise TypeError(f"Invalid iterable input: {type(X)}")
    raise TypeError(f"Input type not supported: {type(X)}")


class TextPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(
            self,
            language="english",
            remove_html=True,
            remove_urls=True,
            lower=True,
            expand_contr=False,
            remove_punct=True,
            remove_sw=True,
            stem=True,
            sw_add=None,
            sw_remove=None,
    ):
        self.language = language
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.lower = lower
        self.expand_contr = expand_contr
        self.remove_punct = remove_punct
        self.remove_sw = remove_sw
        self.stem = stem
        self.sw_add = sw_add
        self.sw_remove = sw_remove

        self._url_re = re.compile(r"https?://\S+|www\.\S+")
        self._punct_re = re.compile(f"[{re.escape(string.punctuation)}]")
        self._html_re = re.compile(r"<[^>]+>")
        self._token_re = re.compile(r"\b\w+\b")

    def fit(self, X, y=None):
        sw = set(stopwords.words(self.language))
        if self.sw_add:
            sw.update(self.sw_add)
        if self.sw_remove:
            for w in self.sw_remove:
                sw.discard(w)
        self._stopwords = sw
        self._stemmer = SnowballStemmer(self.language)
        return self

    def transform(self, X):
        X = normalize_input(X)
        processed = []

        # LOCALIZE FUNCTIONS FOR SPEED
        url_re_sub = self._url_re.sub
        punct_re_sub = self._punct_re.sub
        html_re_sub = self._html_re.sub
        token_re_findall = self._token_re.findall
        stopwords = self._stopwords
        stem = self._stemmer.stem

        expand_contr = self.expand_contr
        if expand_contr:
            import contractions
            fix_contr = contractions.fix

        for text in X:

            if self.remove_html:
                text = html_re_sub(" ", text)

            if self.remove_urls:
                text = url_re_sub(" ", text)

            if expand_contr:
                text = fix_contr(text)

            if self.lower:
                text = text.lower()

            if self.remove_punct:
                text = punct_re_sub(" ", text)

            tokens = token_re_findall(text)

            if self.remove_sw and self.stem:
                tokens = [stem(t) for t in tokens if t not in self._stopwords]
            elif self.remove_sw:
                tokens = [t for t in tokens if t not in self._stopwords]
            elif stem:
                tokens = [stem(t) for t in tokens]

            processed.append(" ".join(tokens))

        return processed
