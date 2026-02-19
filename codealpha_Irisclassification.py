import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    df = pd.read_csv('iris.csv')
    print("Dataset 'iris.csv' loaded successfully!\n")
except FileNotFoundError:
    print("Error: 'iris.csv' file is not found. please check in your system.")
    exit()


if 'ID' in df.columns:
    df = df.drop('ID', axis=1)
    print("'ID' column removed from the dataset.\n")
X = df[['SepalLengthCM', 'SepalWidthCM', 'PetalLengthCM', 'PetalWidthCM']]
y = df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data successfully split into Training (80%) and Testing (20%) sets.\n")

print("Training the Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully!\n")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("--- Evaluation Results ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nGenerating Confusion Matrix Graph...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
# Creating a heatmap for the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix - Iris Classification')
plt.xlabel('Predicted Species')
plt.ylabel('Actual Species')

plt.show() 

print("Project Execution Completed!")
