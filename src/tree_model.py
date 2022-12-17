import os

import joblib
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

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
        Actually, compute ROC-AUC, Average precision, and F1 on a test dataset. True labels are required.

        The naming conforms to DeepPurpose's interface.
        """
        test_X = np.stack(dataset['drug_encoding'], axis=0)
        test_y = dataset['Label']

        pred_y_proba = self.model.predict_proba(test_X)
        pred_y = self.model.predict(test_X)

        roc_auc = roc_auc_score(test_y, pred_y_proba)
        ap_score = average_prediction_score(test_y, pred_y)
        f1 = f1_score(test_y, pred_y)

        return roc_auc, ap_score, f1, pred_y
