import os

import joblib
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier


class TreeModel():
    def __init__(self, **model_kwargs):
        """
        Wrapper around DecisionTreeClassifier to conform to DeepPurpose's interface
        """
        self.model = DecisionTreeClassifier(**model_kwargs)

    def train(self, train_dataset, val_dataset, test_dataset=None):
        train_X = np.stack(train_dataset['drug_encoding'], axis=0)
        train_y = train_dataset['Label']

        self.model.fit(train_X, train_y)

    def save_model(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'model.joblib')
        joblib.dump(self.model, model_path)

    def model_pretrained(self, model_dir):
        path = os.path.join(model_dir, 'model.joblib')
        self.model = joblib.load(path)

    def predict(self, dataset):
        """
        Return probability of class 1.
        """
        test_X = np.stack(dataset['drug_encoding'], axis=0)

        pred_y_proba = self.model.predict_proba(test_X)[:, 1]
        return pred_y_proba
