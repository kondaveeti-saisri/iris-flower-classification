# iris_classification.py
# My First ML Project: Iris Flower Classifier 

print(" Starting Iris Flower Classification Project...\n")

# Step 1: Load Data
from sklearn import datasets
import pandas as pd

print(" Loading Iris dataset...")
iris = datasets.load_iris()

# Convert to DataFrame for easy viewing
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print(" First 5 rows:")
print(df.head())

# Step 2: Explore Data
print("\n Data Summary:")
print(df.describe())

print("\n Class distribution:")
print(df['species'].value_counts())

# Step 3: Train Model
print("\n Training Machine Learning Model...")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = iris.data  # Features (measurements)
y = iris.target  # Target (species)

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 4: Evaluate Model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n Model trained! Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 5: Predict New Flower
print("\n Predicting a New Flower...")
new_flower = [[5.7, 3.1, 4.5, 1.4]]  # Your new flower
prediction = model.predict(new_flower)
predicted_species = iris.target_names[prediction[0]]

print(f"Measurements: {new_flower[0]}")
print(f"Predicted species: {predicted_species}")

# Optional: Show a plot
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='species', s=100)
plt.title("Iris Species by Petal Size")
plt.legend(title="Species")
plt.show(block=True)  # This will show the plot window


print("\n Project Complete! You've built your first AI model!")
