import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import sys

print('You\'re running python %s' % sys.version.split(' ')[0])

# Load data
train_data = pd.read_csv("train.csv")
validation_data = pd.read_csv("validation.csv")
test_data = pd.read_csv("test.csv")

# Identify and remove unnecessary columns
columns_to_remove = ["id"]  # Example column names that are not needed
train_data = train_data.drop(columns=columns_to_remove, errors='ignore')
validation_data = validation_data.drop(columns=columns_to_remove, errors='ignore')
test_data = test_data.drop(columns=columns_to_remove, errors='ignore')

# Separate features and labels
xTr = train_data.drop(columns="label")
yTr = train_data["label"]
xVal = validation_data.drop(columns="label")
yVal = validation_data["label"]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
xTr = imputer.fit_transform(xTr)
xVal = imputer.transform(xVal)
test_data = imputer.transform(test_data)

# Standardize Features
scaler = StandardScaler()
xTrScaled = scaler.fit_transform(xTr)
xValScaled = scaler.transform(xVal)
test_data_scaled = scaler.transform(test_data)

# PCA
pca = PCA(n_components=13)
xTrPCA = pca.fit_transform(xTrScaled)
xValPCA = pca.transform(xValScaled)
test_data_pca = pca.transform(test_data_scaled)

# Train and evaluate models
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "KNN with PCA": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Fit models
models["Logistic Regression"].fit(xTrScaled, yTr)
models["KNN"].fit(xTrScaled, yTr)
models["KNN with PCA"].fit(xTrPCA, yTr)
models["Random Forest"].fit(xTrScaled, yTr)

# Hyperparameter tuning for Random Forest
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

RFGrid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=rf_params, cv=5, scoring='accuracy')
RFGrid.fit(xTrScaled, yTr)

# Evaluate models
for name, model in models.items():
    if name == "KNN with PCA":
        yValPred = model.predict(xValPCA)
    else:
        yValPred = model.predict(xValScaled)
    
    accuracy = accuracy_score(yVal, yValPred)
    print(f"{name} Accuracy: {accuracy}")

# Best Random Forest model from Grid Search
best_rf_model = RFGrid.best_estimator_
RFAccuracy = accuracy_score(yVal, best_rf_model.predict(xValScaled))
print(f"Best Random Forest Accuracy: {RFAccuracy}")

# Prepare test data for submission
test_preds = models["KNN with PCA"].predict(test_data_pca)

# Create submission file
submission = pd.DataFrame({"id": pd.read_csv("test.csv")["id"], "label": test_preds})
submission.to_csv("predictions.csv", index=False)
