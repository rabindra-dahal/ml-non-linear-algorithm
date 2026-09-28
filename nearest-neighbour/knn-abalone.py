import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# 1. Load the Abalone Dataset directly from the official web URL
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
column_names = [
    "Sex", "Length", "Diameter", "Height", "Whole_weight", 
    "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"
]
df = pd.read_csv(url, names=column_names)

# 2. Preprocess Data: Remove the text column ("Sex") to keep only pure measurements
# (KNN measures geometrical distances, so it requires pure numbers!)
df = df.drop(columns=["Sex"])

# 3. Separate features (X) from the target score we want to guess (y)
X = df.drop(columns=["Rings"])
y = df["Rings"]

# 4. Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 5. CRITICAL STEP FOR KNN: Scaling the Data!
# Because weight is measured in grams and length in millimeters, the numbers have different scales.
# Scaling makes sure a feature with large numbers doesn't drown out a feature with small numbers.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Initialize and Train the KNN Regressor
# We set n_neighbors=5, meaning it will check the 5 closest matches
knn_robot = KNeighborsRegressor(n_neighbors=5)
knn_robot.fit(X_train_scaled, y_train)

print(" KNN Robot has stored the abalone library map! \n")

# 7. Test it out with a brand new Mystery Abalone measurement!
# [Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight]
mystery_abalone = pd.DataFrame(
    [[0.53, 0.42, 0.13, 0.67, 0.28, 0.14, 0.22]], 
    columns=X.columns
)

# Scale the mystery shell just like we scaled the training data!
mystery_scaled = scaler.transform(mystery_abalone)

# Predict the age
predicted_rings = knn_robot.predict(mystery_scaled)
estimated_age = predicted_rings[0] + 1.5  # Actual age is roughly Rings + 1.5 years

print("---  ABALONE AGING REPORT  ---")
print(f" Predicted Number of Rings: {predicted_rings[0]:.1f}")
print(f" Estimated Biological Age:  {estimated_age:.1f} years old")
