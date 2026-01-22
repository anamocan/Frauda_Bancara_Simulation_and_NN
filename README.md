# Credit Card Fraud Detection & Banking Simulation

This project implements a complete data science pipeline to detect fraudulent credit card transactions and simulates the business impact of different detection policies. It transitions from a baseline Gaussian statistical model to a high-performance Deep Learning MLP (Multi-Layer Perceptron) built with PyTorch.

## Features
* **Deep Learning Pipeline:** End-to-end implementation using PyTorch, including data scaling and handling extreme class imbalance.
* **Weighted Loss Logic:** Custom BCEWithLogitsLoss weights to penalize missed frauds (False Negatives) more heavily.
* **Simulation Engine:** A discrete-event simulation built with SimPy to model banking operations during Normal and Christmas high-traffic periods.
* **Policy Comparison:** Comparative analysis of Strict vs. Relaxed thresholds and their impact on financial loss vs. customer satisfaction.

## Tech Stack
* **Languages:** Python
* **Deep Learning:** PyTorch
* **Data Analysis:** Pandas, NumPy, Scikit-Learn
* **Simulation:** SimPy
* **Visualization:** Matplotlib

## Project Structure

### Phase 1: Data Preprocessing
* **Dataset:** Kaggle Credit Card Fraud Detection dataset.
* **Normalization:** Standardization of Amount and Time features using StandardScaler.
* **Splitting:** Stratified train-test split to maintain the 0.17% fraud ratio across sets.

### Phase 2: Deep Learning Model
* **Architecture:** 3-layer MLP with ReLU activation and Dropout (0.2 - 0.3) for regularization.
* **Optimization:** AdamW optimizer with weight decay to prevent overfitting.
* **Evaluation Metrics:** Prioritization of Precision-Recall AUC over standard accuracy to better reflect performance on imbalanced data.
* **Early Stopping:** Automated training termination to save the best-performing model state.



### Phase 3: Banking Simulation (SimPy)
The project simulates two banking archetypes to evaluate decision thresholds:
1. **Strict Bank (Low Threshold):** Prioritizes capital security, minimizing stolen money but increasing False Positives (blocked legitimate customers).
2. **Relaxed Bank (High Threshold):** Prioritizes customer experience, lowering friction but accepting a higher risk of financial loss.

## Results
The MLP model demonstrated superior decision boundary efficiency compared to basic statistical assumptions.
* **Risk Polarization:** Legitimate transactions consistently score near 0.0, while frauds polarize toward 1.0.
* **Operational Impact:** The model reduces the "gray area" of uncertainty, allowing banks to maintain high detection rates with minimal customer interruption.

## Installation and Usage
1. Clone the repository:
   `git clone https://github.com/your-username/fraud-detection-simulation.git`
2. Install dependencies:
   `pip install torch pandas numpy scikit-learn matplotlib simpy kagglehub`
3. Run the project:
   Execute the main script or notebook to train the model and generate the simulation plots.
