# Logistic Regression on Iris Binary Dataset (Setosa vs Non-Setosa)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("iris_binary_dataset.csv")

# Features and target
X = df.drop(columns=["is_setosa"])
y = df["is_setosa"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation
cm = confusion_matrix(y_test, y_pred)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Setosa", "Setosa"], yticklabels=["Not Setosa", "Setosa"])
plt.title("Confusion Matrix - Logistic Regression (Binary Iris)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
