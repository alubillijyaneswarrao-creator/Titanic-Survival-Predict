import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath):

    df = pd.read_csv(filepath)

    print("\nDataset Shape:")
    print(df.shape)

    X = df.drop("Survived", axis=1)

    y = df["Survived"]

    X = X.select_dtypes(
        include=["int64", "float64", "bool"]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )