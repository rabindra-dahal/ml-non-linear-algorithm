from sklearn.tree import DecisionTreeClassifier, export_text

# 1. Our Data (Features): [Variance, Skewness] of the banknote image
# Let's say: Genuine notes have high numbers. Fake notes have low numbers.
X = [
    [4.5, 3.2],  # Note A: High variance, high skewness (Genuine)
    [-2.1, -1.5],  # Note B: Low variance, low skewness (Forged)
]

# 2. Our Labels (Targets): 0 means Genuine, 1 means Forged
y = [0, 1]

# 3. Create the "Tree" model and let it study the data (fit)
tree_model = DecisionTreeClassifier()
tree_model.fit(X, y)

# 4. Use the model to test a brand new, unknown banknote!
# This new note has low numbers [-1.0, -0.5]. What will the model guess?
# new_banknote = [[-1.0, -0.5]]
new_banknote = [[58.0, 4.5]]
prediction = tree_model.predict(new_banknote)
print(f"Prediction for the new banknote {new_banknote}: {prediction[0]}")

# 5. Print the result
if prediction[0] == 0:
    print("The model says: This note is GENUINE!")
else:
    print("The model says: This note is FORGED!")

# 6. See the "Flowchart" logic the computer built
print("\n--- The Tree's Internal Flowchart ---")
print(export_text(tree_model, feature_names=["Variance", "Skewness"]))
