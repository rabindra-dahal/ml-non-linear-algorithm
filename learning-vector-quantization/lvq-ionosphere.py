import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


# 1. Fetch the Ionosphere dataset from OpenML
print(" Fetching the Ionosphere dataset...")
ionosphere = fetch_openml(name="ionosphere", version=1, as_frame=True)
X = ionosphere.data.to_numpy()
y = ionosphere.target

# 2. Split into Training (80%) and Testing (20%) datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Scale the continuous radar metrics (vital for distance-based sorting!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Initialize and Train the Nearest Centroid Classifier
# This acts exactly like our "Guard Dog / Prototype" logic!
print(" Positioning the class prototypes...")
model = NearestCentroid(metric="euclidean")
model.fit(X_train_scaled, y_train)
print(" Training complete!\n")

# 5. Evaluate the model performance
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f" Overall Model Accuracy: {accuracy * 100:.2f}%")
print("\n--- Detailed Classification Metrics ---")
print(classification_report(y_test, y_pred))


# 1. Create two new mystery radar signals for prediction 
# (Each must have 34 numbers because the Ionosphere dataset has 34 features)
# Signal A: Mostly high, steady positive values
mystery_signal_A = np.ones(34) * 0.8  

# Signal B: Mostly negative, chaotic values
mystery_signal_B = np.ones(34) * -0.5 

# 2. Combine them into a single test batch DataFrame with column names
new_signals_df = pd.DataFrame(
    [mystery_signal_A, mystery_signal_B],
    columns=ionosphere.data.columns
)

# 3. CRITICAL: Scale the new signals exactly like we scaled our training school data!
new_signals_scaled = scaler.transform(new_signals_df.to_numpy())

# 4. Ask the model to make its final guess
predictions = model.predict(new_signals_scaled)

# 5. Print the Inspection Results
print("\n---  LIVE RADAR PREDICTION REPORT  ---")
for i, pred in enumerate(predictions):
    signal_type = "GOOD SIGNAL (g) " if pred == 'g' else "BAD SIGNAL (b) "
    print(f" Object {i+1}: The model predicts this is a {signal_type}")



# 1. Compute the exact raw distances from our new signals to the two trained centers
# This creates a perfect grid: [Distance to Bad, Distance to Good]
raw_distances = pairwise_distances(new_signals_scaled, model.centroids_)

print("\n---  RAW GEOMETRIC DISTANCE METRICS  ---")

# Print the scores for Object 1 (Mystery Signal A)
print(f"Object 1 (High/Steady Signal):")
print(f"   -> Distance to 'Bad' Center (b) : {raw_distances[0][0]:.2f}")
print(f"   -> Distance to 'Good' Center (g): {raw_distances[0][1]:.2f}")
print(f"    Decision: It picks 'Good' because it has the shorter distance!\n")

# Print the scores for Object 2 (Low/Chaotic Signal)
print(f"Object 2 (Negative/Chaotic Signal):")
print(f"   -> Distance to 'Bad' Center (b) : {raw_distances[1][0]:.2f}")
print(f"   -> Distance to 'Good' Center (g): {raw_distances[1][1]:.2f}")
print(f"    Decision: It picks 'Bad' because it has the shorter distance!")
