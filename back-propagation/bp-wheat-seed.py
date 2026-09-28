import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# 1. Fetch the real Wheat Seeds dataset from OpenML
print(" Fetching the Wheat Seeds dataset...")
seeds = fetch_openml(name="seeds", version=1, as_frame=True, parser="auto")
X = seeds.data.to_numpy()
y = seeds.target.astype(int)  # Labels: 1 = Kama, 2 = Rosa, 3 = Canadian

# 2. Split into Training (80%) and Testing (20%) datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. CRITICAL: Scale your features!
# Neural networks use backpropagation gradients which fail if data scales vary wildly.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Initialize the Neural Network (Backpropagation Model)
# - hidden_layer_sizes=(10, 5): 10 neurons in layer one, 5 neurons in layer two
# - max_iter=500: Give the archer up to 500 practice epochs to adjust their aim
# - learning_rate_init=0.01: Step size for weight adjustments
print(" Initiating forward shots and backpropagation adjustments...")
mlp_network = MLPClassifier(
    hidden_layer_sizes=(10, 5),
    activation="relu",
    solver="adam",
    learning_rate_init=0.01,
    max_iter=500,
    random_state=42
)

# Train the network!
mlp_network.fit(X_train_scaled, y_train)
print(" Training complete! The weights have been optimized.\n")

# 5. Evaluate the model performance
y_pred = mlp_network.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f" Overall Neural Network Accuracy: {accuracy * 100:.2f}%")
print("\n--- Detailed Variety Classification Metrics ---")
# Mapping back to real wheat variety names
variety_names = ["Kama (1)", "Rosa (2)", "Canadian (3)"]
print(classification_report(y_test, y_pred, target_names=variety_names))



# 1. Create two new mystery wheat seeds for prediction (each must have 7 continuous numerical features): 
# Each seed needs exactly 7 continuous numerical features matching the dataset:
# [Area, Perimeter, Compactness, Length, Width, Asymmetry_Coefficient, Groove_Length]

# Seed A: Very large area and perimeter (Likely Rosa)
mystery_seed_A = [18.5, 16.2, 0.88, 6.2, 3.7, 2.1, 6.0]

# Seed B: Much smaller area, very elongated (Likely Canadian)
mystery_seed_B = [11.2, 13.0, 0.82, 5.2, 2.5, 5.1, 4.8]

# 2. Convert into a proper DataFrame with matching features
new_seeds_df = pd.DataFrame(
    [mystery_seed_A, mystery_seed_B],
    columns=seeds.data.columns
)

# 3. CRITICAL: Strip the names to match the scaler format and transform
new_seeds_scaled = scaler.transform(new_seeds_df.to_numpy())

# 4. Run the forward pass to get final predictions
predictions = mlp_network.predict(new_seeds_scaled)

# 5. Extract probabilities to see the neural network's internal confidence
probabilities = mlp_network.predict_proba(new_seeds_scaled)

# Variety lookup mapping
variety_names = {1: "Kama ", 2: "Rosa ", 3: "Canadian "}

print("\n---  NEURAL NETWORK SEED PREDICTION REPORT  ---")
for i, pred in enumerate(predictions):
    print(f"\n Mystery Seed {i+1}:")
    print(f"    Final Prediction: {variety_names[pred].upper()}")
    print("    Network Confidence Breakdown:")
    print(f"      -> Kama    (1): {probabilities[i][0] * 100:.1f}%")
    print(f"      -> Rosa    (2): {probabilities[i][1] * 100:.1f}%")
    print(f"      -> Canadian(3): {probabilities[i][2] * 100:.1f}%")

