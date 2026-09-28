# 🤖 Machine Learning Playground: Non-Linear Algorithms

Welcome to the **Machine Learning Playground**! This repository tracks a step-by-step educational journey exploring how non-linear algorithms process complex datasets—from structural decision flowcharts to geometric prototype shifting and multi-layer neural networks.

---

## 📚 What is a Non-Linear Model?
Linear models try to split data using a single straight line, which often fails in real-world scenarios. **Non-linear models** can bend, box, or isolate boundaries, allowing them to capture intricate, complex feature interactions without assuming a simple linear relationship.

---

## 🛠️ Complete Implementation Log

Here are the concrete algorithms, concepts, and real-world datasets implemented during this journey:

### 1. 🌲 Classification and Regression Trees (CART)
* **Dataset:** [UCI Banknote Authentication Dataset](https://uci.edu) 💸
* **Core Concept:** Works exactly like a **Google Search Filter** or a game of "20 Questions." It builds an orthogonal, step-like decision flowchart using **Gini Impurity** (`criterion="gini"`) to systematically reduce data chaos and separate clean subgroups.
* **Key Visual Insight:** Decision spaces are cut strictly vertically and horizontally, carving out "stairs" on a graph to catch complex counterfeiting patterns.

### 2. 🌸 Probabilistic Classifiers (Naive Bayes)
* **Dataset:** [Iris Flower Dataset](https://scikit-learn.org) 🌸
* **Core Concept:** Predicts categories using individual probabilities (Bayes' Theorem). It is called **"Naive"** because it innocently assumes all clues (like petal width vs. length) are completely independent of each other. 
* **Variant Used:** `GaussianNB` (ideal for continuous decimal values like millimeter measurements).

### 3. 🐚 Proximity Clustering (K-Nearest Neighbors)
* **Dataset:** [UCI Abalone Dataset](https://uci.edu) 🐚
* **Core Concept:** Looks at the physical characteristics of an unknown shell, checks the `K` closest matches (neighbors), and calculates their average to accurately predict the biological age (rings).

### 4. 🧭 Prototype Shifting (Learning Vector Quantization / Nearest Centroid)
* **Dataset:** [UCI Ionosphere Radar Dataset](https://uci.edu) 📡
* **Core Concept:** Works like training **Guard Dogs** to claim structural territories. It anchors marker coordinates ("prototypes") in space and dynamically recalculates their boundaries using Euclidean distance measurements.
* **Model Used:** `NearestCentroid` (Bypassed package version conflicts cleanly with a stellar **92.96% validation accuracy**).

### 🧠 5. Deep Learning (Backpropagation Neural Network)
* **Dataset:** [OpenML Wheat Seeds Dataset](https://openml.org) 🌾
* **Core Concept:** Works like an archer taking practice shots. It performs a forward guess (**Forward Pass**), computes the prediction variance error, and flows backward (**Backpropagation**) using the `adam` solver to optimize weights and biases across non-linear `relu` activation layers.
* **Model Layout:** `MLPClassifier(hidden_layer_sizes=(10, 5))` resulting in an outstanding **90.48% classification success rate** and up to **100.0% targeted confidence metrics** on live queries.

### 🍷 6. Dimensional Bending (Support Vector Machine)
* **Dataset:** [UCI Wine Dataset](https://uci.edu) 🍷
* **Core Concept:** Works like a math **"Forcefield."** It uses a non-linear **Radial Basis Function (RBF) Kernel** to mathematically raise tangled, un-splittable data points into a higher dimension where a flat boundary sheet can cleanly separate them.
* **Model Layout:** Wrapped cleanly inside the modern `CalibratedClassifierCV` wrapper to maintain robust, thread-safe probability predictions without syntax warnings, achieving a peak performance of **97.22% evaluation accuracy**.

---

## ⚙️ Project Requirements & Setup

To run any of the standalone scripts in this directory, set up your Python virtual environment and install the verified core scientific stack:

```bash
pip install scikit-learn pandas numpy matplotlib
```

### 💡 Core Engineering Lessons Learned
1. **`StandardScaler` is Mandatory:** Distance-based models (like KNN, Nearest Centroid, SVM, and Neural Networks) calculate spatial vectors. If you do not normalize your scales, large unit features will completely drown out decimal weights.
2. **Handle Incomplete Training Data:** Machine learning models can only choose from answers they have studied. Unseen anomalous input combinations require a fallback safety tier or explicit classification mapping to prevent erroneous target forced-assignments.
3. **Data Leakage Warnings:** When feeding live single array queries (`.predict()`) into scaled model environments, pass them securely wrapped as a `pd.DataFrame` matching your training `.to_numpy()` signature blocks to completely eliminate compilation feature name mismatches.
4. **Calibrate for Confidence:** For algorithms like SVM that operate on spatial boundaries rather than pure logs, using wrappers like `CalibratedClassifierCV` ensures stable, future-proof probability mappings without deprecation friction.
