import joblib
import pandas as pd


def predict_survival(input_data):

    model = joblib.load(
        "models/decision_tree_model.pkl"
    )

    input_df = pd.DataFrame(
        [input_data]
    )

    prediction = model.predict(
        input_df
    )

    if prediction[0] == 1:
        return "Survived"

    return "Did Not Survive"