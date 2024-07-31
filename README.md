# Heart Disease Prediction

## Overview
This project is based on a Kaggle dataset where we need to predict heart disease using the given training, validation, and test datasets. The following machine learning techniques were applied to achieve the best prediction accuracy: Logistic Regression, K-Nearest Neighbors (KNN), K-Nearest Neighbors with Principal Component Analysis (PCA), and Random Forest. 

## Dataset
The dataset includes training, validation, and test datasets with features such as age, sex, cholesterol levels, etc., and a label indicating the presence of heart disease. The datasets are taken from Kaggle. 

- [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets)

## Installation
To run this project, you need Python 3.x and the following libraries:
- numpy
- pandas
- matplotlib
- scikit-learn

You can install the required libraries using pip:

    pip install numpy pandas matplotlib scikit-learn

## Usage

Clone this repository:

    git clone https://github.com/KrishnaHPatel/ML_Heart_Disease_Predictor.git 

    cd ML_Heart_Disease_Predictor.git
    
Run the main script:

    python3 ml_main.py

## Modeling Approach

Data Preprocessing
- Imputation: Missing values are filled using the mean.
- Standardization: Features are standardized to have zero mean and unit variance.
- CA: Principal Component Analysis is applied to reduce dimensionality.

Models Used
- Logistic Regression
- K-Nearest Neighbors (KNN)
- KNN with PCA
- Random Forest
  
  Hyperparameter Tuning: Grid search is used to find the best hyperparameters for the Random Forest model.

## Results

Models are evaluated based on their accuracy on the validation dataset. The best model is used to make predictions on the test dataset. 

The performance of each model on the validation dataset is printed, and the best performing model is used to generate predictions for the test dataset. The results are saved in a predictions.csv file.