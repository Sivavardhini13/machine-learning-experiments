import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Introduce missing values (simulation)
X.iloc[5:10, 2] = np.nan

# Handle missing values
X.fillna(X.mean(), inplace=True)

# Function to run pipeline
def run_pipeline(scaler, X, y, name):
    print(f"\n===== {name} =====")

    # Scaling
    X_scaled = scaler.fit_transform(X)

    # PCA (2 components)
    X_pca = PCA(n_components=2).fit_transform(X_scaled)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Run experiments
run_pipeline(StandardScaler(), X, y, "StandardScaler + PCA + Logistic Regression")
run_pipeline(MinMaxScaler(), X, y, "MinMaxScaler + PCA + Logistic Regression")