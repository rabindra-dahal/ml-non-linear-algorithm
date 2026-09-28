import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Download the real banknote data from the internet
print(" Fetching the banknote dataset... Please wait...")
banknote = fetch_openml(name="Banknote-Authentication", version=1, as_frame=True)
X = banknote.data       # The 4 clues: Variance, Skewness, Kurtosis, Entropy
y = banknote.target.astype(int)  # The answers: 0 = Genuine, 1 = Forged

# 2. Split into "School Data" (80%) and "Exam Data" (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 3. Create and train our Decision Tree Robot
# We limit max_depth=3 so the flowchart stays simple and doesn't over-complicate rules
robot_brain = DecisionTreeClassifier(max_depth=3, random_state=42)
robot_brain.fit(X_train, y_train)

print("✅ Robot training complete!\n")

# 4. Create an imaginary "Mystery Banknote" to test the model
# Let's give it values: [Variance, Skewness, Kurtosis, Entropy]
# Remember: Low/negative numbers usually mean blurry/forged!
mystery_bill = pd.DataFrame(
    [[-2.5, -1.2, 3.4, -0.5]], 
    columns=X.columns
)

# 5. Predict the answer AND check how confident the robot is!
prediction = robot_brain.predict(mystery_bill)
confidence = robot_brain.predict_proba(mystery_bill) # Gives percentages

# 6. Reveal the final score
genuine_chance = confidence[0][0] * 100
forged_chance = confidence[0][1] * 100

print("---  INSPECTION REPORT  ---")
if prediction[0] == 0:
    print(f" Robot says: This bill looks GENUINE! ")
else:
    print(f"Robot says: Warning! This bill looks FORGED! ")

print(f"Confidence breakdown:")
print(f"   -> Chance it is Genuine: {genuine_chance:.1f}%")
print(f"   -> Chance it is Forged:  {forged_chance:.1f}%")
