import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_X_y


class EnsembleModel(BaseEstimator):
    
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.models = [
            MultiOutputClassifier(SGDClassifier(max_iter=10), n_jobs=-1),
            MultiOutputClassifier(Perceptron(), n_jobs=-1),
            MultiOutputClassifier(PassiveAggressiveClassifier(), n_jobs=-1),
            MultiOutputClassifier(MultinomialNB(alpha=0.01), n_jobs=-1),
            MultiOutputClassifier(LinearSVC(), n_jobs=-1),
            MultiOutputClassifier(AdaBoostClassifier(n_estimators=100), n_jobs=-1)
        ]
    def fit(self, X, y):
        y_ = self.mlb.fit_transform(y)
        check_X_y(X, y_, accept_sparse=True, multi_output=True)
        
        for clf in self.models:
            clf.fit(X, y_)
            
    def transform(self, X):
        return [clf.predict(X) for clf in self.models]
    
    def predict(self, X):
        preds = self.transform(X)
        join_preds = np.maximum.reduce(preds)
        return pd.DataFrame(data=join_preds, columns=self.mlb.classes_)