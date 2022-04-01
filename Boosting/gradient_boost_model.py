from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np


class GradientBoostModel:
    def __init__(self, se):
        self.gradient_booster = GradientBoostingClassifier(n_estimators=se.n_estimators, learning_rate=se.learning_rate,
                                                           max_depth=se.max_depth,
                                                           random_state=se.random_state,
                                                           subsample=se.subsample)

    def train(self, x_train, y_train):
        self.gradient_booster.fit(x_train, y_train)
        print("end")

    def evaluate(self, x_test, y_test):
        return classification_report(y_test, self.gradient_booster.predict(x_test))

    def error_rate(self, x, y):
        y_prediction = self.gradient_booster.predict(x)
        error_rate = np.count_nonzero(y - y_prediction) / len(y_prediction)
        return error_rate
