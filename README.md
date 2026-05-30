# Titanic Survival Prediction Using Linear Regression and Decision Tree

## Project Overview

This project applies supervised machine learning techniques on the Titanic dataset to predict passenger survival. The dataset is first cleaned and transformed using data preprocessing and feature engineering techniques. Two machine learning models, Linear Regression and Decision Tree, are trained and evaluated to compare their predictive performance.

---

## Objectives

* Load and preprocess the Titanic dataset.
* Train Linear Regression and Decision Tree models.
* Evaluate model performance using regression and classification metrics.
* Compare model performance visually.
* Save trained models for future use.

---

## Dataset Information

Dataset: Titanic Dataset

Target Variable:

* Survived

  * 0 = Did Not Survive
  * 1 = Survived

Features Used:

* Passenger Class
* Gender
* Age
* Fare
* Family Size
* Embarked Information
* Title Information
* Age Group Information

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Joblib
* VS Code

---

## Project Structure

Titanic_Model_Training_Project/

├── dataset/

│ └── cleaned_titanic.csv

├── models/

│ ├── linear_regression_model.pkl

│ └── decision_tree_model.pkl

├── outputs/

│ └── model_comparison.png

├── src/

│ ├── data_preprocessing.py

│ ├── train_models.py

│ ├── evaluate_models.py

│ └── predict.py

├── main.py

├── requirements.txt

└── README.md

---

## Machine Learning Models

### Linear Regression

Used to predict survival values and evaluated using:

* Mean Squared Error (MSE)
* R² Score

### Decision Tree Classifier

Used for classification and evaluated using:

* Accuracy Score

---

## Results

### Linear Regression

* MSE: 0.1371
* R² Score: 0.4347

### Decision Tree

* Accuracy: 75.98%

---

## Generated Outputs

* model_comparison.png

The graph visually compares model performance.

---

## How to Run

### Install Dependencies

pip install -r requirements.txt

### Run Project

python main.py

---

## Future Improvements

* Logistic Regression
* Random Forest Classifier
* XGBoost
* Hyperparameter Tuning
* Cross Validation
* Streamlit Deployment

---

## Author

A. Jyaneswar Rao

B.Tech Computer Science (AI & ML)

Parul University
