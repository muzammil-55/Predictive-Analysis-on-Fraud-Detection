import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# Generate a synthetic fraud dataset with 10000 samples and 20 features (adjust as needed)
X, y = make_classification(n_samples=10000, n_features=20, 
n_informative=10, n_redundant=5, n_classes=2,
                           weights=[0.99, 0.01], random_state=42)
# DataFrame Creation:
columns = [f"feature_{i}" for i in range(20)]
fraud_data = pd.DataFrame(data=X, columns=columns)
fraud_data['is_fraud'] = y
# Split the data into training and testing sets (80% for training, 20% for testing)
X = fraud_data.drop('is_fraud', axis=1)
y = fraud_data['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42)
# Create and train the Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
conf_matrix = confusion_matrix(y_test, y_pred)
print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("Confusion Matrix:")
print(conf_matrix)