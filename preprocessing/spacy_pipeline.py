import spacy
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class SpacyPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        language="en_core_web_sm",
        remove_html=True,
        remove_urls=True,
        lower=True,
        remove_punct=True,
        remove_sw=True,
        lemma=True,
        batch_size=2000,
        n_process=2,
        sw_add=None,
        sw_remove=None
    ):
        self.language = language
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.lower = lower
        self.remove_punct = remove_punct
        self.remove_sw = remove_sw
        self.lemma = lemma
        self.batch_size = batch_size
        self.n_process = n_process
        self.sw_add = sw_add
        self.sw_remove = sw_remove

    def fit(self, X, y=None):

        self.nlp = spacy.load(self.language, disable=["parser", "ner", "textcat"])
        sw = self.nlp.Defaults.stop_words

        if self.sw_add:
            sw.update(self.sw_add)
        if self.sw_remove:
            sw.difference_update(self.sw_remove)

        self._stopwords = sw

        return self

    def transform(self, X):

        if isinstance(X, pd.Series):
            X = X.astype(str).tolist()
        elif isinstance(X, np.ndarray):
            X = X.astype(str).tolist()
        elif isinstance(X, str):
            X = [X]
        else:
            X = list(map(str, X))

        docs = self.nlp.pipe(
            X,
            batch_size=self.batch_size,
            n_process=self.n_process
        )

        processed = []

        for doc in docs:
            tokens = []

            for t in doc:

                if self.remove_sw and t.text.lower() in self._stopwords:
                    continue
                if self.remove_punct and t.is_punct:
                    continue
                if self.remove_urls and t.like_url:
                    continue
                if self.remove_html and t.like_email:
                    continue

                tok = t.lemma_ if self.lemma else t.text
                if self.lower:
                    tok = tok.lower()

                if tok.strip():
                    tokens.append(tok)

            processed.append(" ".join(tokens))

        return processed
