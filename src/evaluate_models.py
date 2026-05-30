from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score
)

import matplotlib.pyplot as plt
import os


def evaluate_models(
        linear_model,
        tree_model,
        X_test,
        y_test
):

    linear_predictions = (
        linear_model.predict(X_test)
    )

    tree_predictions = (
        tree_model.predict(X_test)
    )

    mse = mean_squared_error(
        y_test,
        linear_predictions
    )

    r2 = r2_score(
        y_test,
        linear_predictions
    )

    accuracy = accuracy_score(
        y_test,
        tree_predictions
    )

    print("\nLinear Regression Results")
    print("=" * 40)

    print("MSE:", mse)
    print("R2 Score:", r2)

    print("\nDecision Tree Results")
    print("=" * 40)

    print("Accuracy:", accuracy)

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    models = [
        "Linear Regression",
        "Decision Tree"
    ]

    scores = [
        r2,
        accuracy
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        models,
        scores
    )

    plt.title(
        "Model Comparison"
    )

    plt.savefig(
        "outputs/model_comparison.png"
    )

    plt.close()

    print(
        "\noutputs/model_comparison.png"
    )