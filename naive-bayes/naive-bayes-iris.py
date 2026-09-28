import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# 1. Load the Iris Flower Dataset
iris = load_iris(as_frame=True)
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target classes: 0 = Setosa, 1 = Versicolor, 2 = Virginica

# 2. Split the data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Initialize and Train the Naive Bayes Classifier
# We use 'GaussianNB' because our flower measurements are continuous numbers (like 5.1cm, 3.5cm)
bayes_robot = GaussianNB()
bayes_robot.fit(X_train, y_train)

print(" Naive Bayes model has successfully studied the flowers! \n")

# 4. Create a completely new, unknown mystery flower to test our model
# We create a proper pandas DataFrame right away to avoid that 'feature names' warning!
mystery_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],  # Example measurements
    columns=X.columns,
)

# 5. Predict the flower type and look at the confidence percentages!
prediction = bayes_robot.predict(mystery_flower)
probabilities = bayes_robot.predict_proba(mystery_flower)[0]

# Map the numbers (0, 1, 2) back to the actual flower names
flower_names = iris.target_names

print("---  MYSTERY FLOWER REPORT  ---")
print(f" Prediction: This flower is an IRIS-{flower_names[prediction[0]].upper()}!")
print("\n Mathematical Probability Breakdown:")
for i, name in enumerate(flower_names):
    print(f"   -> Chance it is {name.capitalize()}: {probabilities[i] * 100:.2f}%")

# 6. Check the overall test exam score
y_pred = bayes_robot.predict(X_test)
print(f"\n Overall Model Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.1f}%")
