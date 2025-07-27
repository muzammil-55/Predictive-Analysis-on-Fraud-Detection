# Predictive-Analysis-on-Fraud-Detection
Academic Project

🛡️ Fraud Detection using Random Forest (Sample Code)
This mini project demonstrates how to build a fraud detection model using a Random Forest Classifier trained on a synthetic dataset created with make_classification.

🧠 Technologies Used
Python
NumPy
Pandas
scikit-learn

📌 What This Code Does
✅ Generates a synthetic dataset mimicking fraudulent and genuine transactions
✅ Splits the dataset into training and testing sets
✅ Trains a Random Forest Classifier
✅ Predicts outcomes on test data
✅ Evaluates the model using:

Accuracy
Precision
Recall
F1 Score
ROC AUC
Confusion Matrix

Evaluation Metrics:
Accuracy: 0.9870
Precision: 0.9231
Recall: 0.7969
F1 Score: 0.8559
ROC AUC: 0.9873
Confusion Matrix:
[[1963   10]
 [  15   59]]
(Note: Actual values may vary slightly each time due to randomness.)

📁 Dataset
The dataset is synthetically generated using sklearn.datasets.make_classification() with:
10,000 samples
20 features
Imbalanced class distribution (99% non-fraud, 1% fraud)

Author
Muzammil Khan
