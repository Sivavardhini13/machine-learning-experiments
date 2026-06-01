from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling (important for MLP)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Base MLP Model
base_model = MLPClassifier(max_iter=1000, random_state=42)
base_model.fit(X_train, y_train)

base_pred = base_model.predict(X_test)

print("\n===== Base MLP Model =====")
print("Accuracy:", accuracy_score(y_test, base_pred))
print("Classification Report:\n", classification_report(y_test, base_pred))

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(10,), (50,), (10, 10)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam']
}

grid = GridSearchCV(
    MLPClassifier(max_iter=1000, random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

# Best Model Evaluation
best_model = grid.best_estimator_
grid_pred = best_model.predict(X_test)

print("\n===== Best MLP Model (GridSearchCV) =====")
print("Best Parameters:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, grid_pred))
print("Classification Report:\n", classification_report(y_test, grid_pred))