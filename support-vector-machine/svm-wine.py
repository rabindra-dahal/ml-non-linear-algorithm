import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Load the Wine Dataset
wine = load_wine(as_frame=True)
X = wine.data
y = wine.target

# 2. Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Scale the features (Absolute requirement for SVM distance metrics!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.to_numpy())
X_test_scaled = scaler.transform(X_test.to_numpy())

# 4. Initialize the SVM base model with an RBF Kernel
base_svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)

# 5. Wrap the SVM inside CalibratedClassifierCV to get clean probability mapping
# ensemble=False matches the classic prediction workflow perfectly without warning logs
print(" Bending the data dimensions with a Calibrated RBF Kernel SVM...")
svm_model = CalibratedClassifierCV(estimator=base_svm, ensemble=False)
svm_model.fit(X_train_scaled, y_train)
print(" Training complete!\n")

# 6. Evaluate the model performance
y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f" Overall SVM Accuracy: {accuracy * 100:.2f}%")
print("\n--- Detailed Wine Classification Metrics ---")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# =====================================================================
# 7. LIVE PREDICTION FOR AN UNKNOWN MYSTERY WINE
# =====================================================================
mystery_wine = pd.DataFrame(
    [[14.0, 1.7, 2.4, 19.5, 105.0, 2.8, 3.0, 0.3, 2.0, 5.5, 1.0, 3.4, 1050.0]],
    columns=X.columns
)

# Scale using raw numpy arrays to ensure a stable prediction workflow
mystery_scaled = scaler.transform(mystery_wine.to_numpy())

# Make the warning-free prediction and check confidence breakdown
prediction = svm_model.predict(mystery_scaled)
probabilities = svm_model.predict_proba(mystery_scaled)

print("\n---  LIVE WINE PREDICTION REPORT  ---")
print(f" Final Prediction: CULTIVAR_{wine.target_names[prediction[0]].upper()}")
print(" Modern RBF Calibrated Confidence Breakdown:")
for i, name in enumerate(wine.target_names):
    print(f"   -> {name.capitalize()}: {probabilities[0][i] * 100:.1f}%")
