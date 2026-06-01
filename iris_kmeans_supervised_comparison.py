from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# K-Means Clustering (Unsupervised)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_train)

cluster_labels = kmeans.labels_
print("KMeans Cluster Labels (Train Data):")
print(cluster_labels)

# Logistic Regression (Supervised)
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)
log_acc = accuracy_score(y_test, log_pred)

print("\n===== Logistic Regression =====")
print("Accuracy:", log_acc)

# Decision Tree (Supervised)
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)
tree_acc = accuracy_score(y_test, tree_pred)

print("\n===== Decision Tree =====")
print("Accuracy:", tree_acc)