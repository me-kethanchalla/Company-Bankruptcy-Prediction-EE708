# Company Bankruptcy Prediction

## Overview
This project aims to predict whether a company is likely to go bankrupt based on its financial indicators. The prediction framework integrates a **Deep Neural Network (DNN)** and a **Gaussian Naive Bayes (GNB)** classifier, leveraging the strengths of both models. The hybrid ensemble approach combines probabilistic inference and deep feature learning, achieving robust performance on an imbalanced dataset.

## Features
- **Exploratory Data Analysis (EDA):** Insights into data patterns, feature correlations, and class imbalance.
- **Data Preprocessing:**
  - Feature selection using ANOVA.
  - Oversampling the minority class using SMOTE.
  - Standardization of features for uniform scaling.
- **Model Architecture:**
  - DNN with 3 hidden layers and dropout for regularization.
  - GNB for probabilistic classification.
  - Ensemble approach using soft voting to combine predictions.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score.

---

## Dataset
- **Source:** `Train.csv`
- **Size:** 5,455 rows × 96 columns.
- **Target Variable:** `Bankrupt?` (Binary: 0 = Non-bankrupt, 1 = Bankrupt).
- **Class Distribution:**
  - Non-bankrupt: 5,301 (97.2%)
  - Bankrupt: 154 (2.8%)

---

## Data Preprocessing
1. **Feature Selection:**
   - Used ANOVA F-scores to select the top 30 features relevant to bankruptcy prediction.
2. **Handling Class Imbalance:**
   - Applied SMOTE to oversample the minority class in the training data.
3. **Standardization:**
   - Scaled features to have zero mean and unit variance using `StandardScaler`.

---

## Model Architecture
### Deep Neural Network (DNN)
- Input Layer: Takes selected features as input.
- Hidden Layers:
  - Layer 1: 256 neurons, ReLU activation, Batch Normalization, Dropout (50%).
  - Layer 2: 128 neurons, ReLU activation, Batch Normalization, Dropout (50%).
  - Layer 3: 64 neurons, ReLU activation, Batch Normalization, Dropout (40%).
- Output Layer: Single neuron with Sigmoid activation for binary classification.
- Optimizer: Adam (learning rate = 0.0005).
- Loss Function: Binary Crossentropy.

### Gaussian Naive Bayes (GNB)
- Probabilistic model assuming feature independence.
- Calculates posterior probabilities using Gaussian likelihoods.

### Ensemble Approach
- Combined predictions from DNN and GNB using soft voting.
- Fine-tuned decision threshold to maximize F1-score.

---

## Evaluation Metrics
The model was evaluated on a test set (20% split from the dataset) which was trained on heavy class imbalance of ratio 1:33 , achieving the following results:
- **Accuracy:** 97.23%
- **Precision:** 48.57%
- **Recall:** 54.84%
- **F1-Score:** 51.52%
- **Best Threshold:** 0.45

---

## Files in Repository
| File Name                     | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `Code.ipynb`                  | Contains the complete implementation of the project.                       |
| `Evaluation.ipynb`            | Loads user input data and uses saved models (`pkl` files) for predictions. |
| `GaussianNB_model.pkl`        | Pre-trained Gaussian Naive Bayes model.                                    |
| `dnn_model.h5`                | Pre-trained Deep Neural Network model.                                     |
| `scaler.pkl`                  | Scaler object used for standardizing input features.                       |
| `Train.csv`                   | Dataset used for training and evaluation.                                  |
| `Company-Bankruptcy-Prediction.pdf` | Detailed project report summarizing the approach and results.         |
| `Report_IEEE.pdf`             | IEEE-style report for academic purposes.                                   |

---

## How to Run
1. Clone this repository:
    ```bash
   git clone https://github.com/your-repo/Company-Bankruptcy-Prediction.git
2. Install required dependencies:
   ```bash    
   pip install -r requirements.txt
3. Run `Evaluation.ipynb` to load user input data and predict bankruptcy status.

---

## Results & Conclusion
The hybrid framework effectively addresses class imbalance and combines probabilistic reasoning with deep learning capabilities. With an accuracy of **97.23%** and an F1-score of **51.52%**, this model demonstrates its potential for accurate bankruptcy prediction despite severe class imbalance.

Future improvements could include exploring advanced ensemble techniques or incorporating additional financial indicators for better predictions.
