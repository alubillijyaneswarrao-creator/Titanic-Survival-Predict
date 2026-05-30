from src.data_preprocessing import (
    load_data
)

from src.train_models import (
    train_models
)

from src.evaluate_models import (
    evaluate_models
)

def main():

    filepath = (
        "dataset/cleaned_titanic.csv"
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_data(filepath)

    (
        linear_model,
        tree_model
    ) = train_models(
        X_train,
        y_train
    )

    evaluate_models(
        linear_model,
        tree_model,
        X_test,
        y_test
    )
    

if __name__ == "__main__":
    main()