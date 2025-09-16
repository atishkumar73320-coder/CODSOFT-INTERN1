import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")
print("Dataset loaded successfully!\n")
print(df.head())

# 2. Handle missing values
df = df.drop(["Cabin"], axis=1)  # Drop cabin (too many missing)
df = df.dropna(subset=["Embarked"])  # Drop missing Embarked

# Fill missing Age with median
df["Age"].fillna(df["Age"].median(), inplace=True)

# 3. Encode categorical variables
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])  # male=1, female=0
df["Embarked"] = le.fit_transform(df["Embarked"])

# 4. Select features (X) and target (y)
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df["Survived"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Build logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 7. Predictions
y_pred = model.predict(X_test)

# 8. Evaluation
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
