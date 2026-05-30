from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

import joblib
import os


def train_models(
        X_train,
        y_train
):

    linear_model = LinearRegression()

    linear_model.fit(
        X_train,
        y_train
    )

    tree_model = (
        DecisionTreeClassifier(
            random_state=42
        )
    )

    tree_model.fit(
        X_train,
        y_train
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        linear_model,
        "models/linear_regression_model.pkl"
    )

    joblib.dump(
        tree_model,
        "models/decision_tree_model.pkl"
    )

    return (
        linear_model,
        tree_model
    )