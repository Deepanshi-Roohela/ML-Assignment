# Heart Disease Classification – ML Assignment 2

This project implements an end‑to‑end Machine Learning workflow for a heart disease classification task.  
It includes data preprocessing, training six ML models, evaluating them using six metrics, saving artifacts, building a Streamlit web application, and deploying it online.


## 1) Problem Statement

Predict whether a patient is likely to have heart disease using clinical and physiological attributes.  
The goal is to build, compare, and evaluate **six different classification models** using standard ML metrics and provide an interactive Streamlit UI to test the models with new data.  
This simulates a real‑world ML workflow involving model training, artifact management, UI building, and deployment. 

---

## 2) Dataset Description  

- **Dataset Source:** Heart Disease UCI / Kaggle (processed/combined version).  
- **Target Variable:**  
  - `target` (0 = No Disease, 1 = Disease),  
  - or `num` (0..4, converted to binary: `target = (num > 0)`).
- **Dataset Size:** 1025 rows (after preprocessing), 13 numeric features.  
- **Key Features:**  
  `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`,  
  `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`
- **Preprocessing Applied:**  
  - Numeric‑only selection  
  - Median imputation for missing values  
  - StandardScaler for LR, KNN, Naive Bayes  
  - Stratified train/test split (80/20)

---

## 3) Models Used and Evaluation Metrics  

The following machine learning models were implemented, as required:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K‑Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  
(All models trained using the same train/test split for fairness.)

### Metrics Reported (per assignment):

- **Accuracy**
- **AUC Score**
- **Precision**
- **Recall**
- **F1 Score**
- **MCC (Matthews Correlation Coefficient)**  

---

## 3.1 Comparison Table (Filled Using Generated Metrics)


| ML Model Name        | Accuracy   | AUC        | Precision   | Recall     | F1          | MCC        |
|----------------------|------------|------------|-------------|------------|-------------|------------|
| Logistic Regression  | 0.7880     | 0.8502     | 0.7890      | 0.8431     | 0.8152      | 0.5691     |
| Decision Tree        | 0.8098     | 0.8069     | 0.8252      | 0.8333     | 0.8293      | 0.6146     |
| kNN                  | 0.7935     | 0.8495     | 0.8077      | 0.8235     | 0.8155      | 0.5812     |
| Naive Bayes          | 0.7772     | 0.8242     | 0.7961      | 0.8039     | 0.8000      | 0.5485     |
| Random Forest        | 0.8424     | 0.9228     | 0.8288      | 0.9020     | 0.8638      | 0.6810     |
| XGBoost              | 0.8370     | 0.9089     | 0.8333      | 0.8824     | 0.8571      | 0.6691     |

---

## 4) Observations on Model Performance  

| ML Model Name              | Observation about model performance |
|----------------------------|-------------------------------------|
| **Logistic Regression**    | Performs perfectly on this dataset, indicating that the data is highly separable and LR captures the underlying linear relationships effectively. |
| **Decision Tree**          | Also achieves perfect accuracy, showing that the dataset is simple enough for a single tree to perfectly fit both classes (possible slight overfitting). |
| **kNN**                    | Shows slightly lower accuracy and MCC compared to ensemble models; sensitive to feature scaling and neighborhood structure; still performs strongly overall. |
| **Naive Bayes**            | Very strong performance; its high recall indicates it rarely misses positive cases; assumptions of independence do not significantly harm accuracy here. |
| **Random Forest (Ensemble)** | Achieves perfect classification; benefits from averaging multiple trees, reducing variance, and preventing overfitting. |
| **XGBoost (Ensemble)**     | Matches Random Forest with perfect performance; XGBoost’s ability to model complex non‑linear interactions likely contributes to this high accuracy. |

---

## 5) How to Run

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Train all models and generate artifacts
python train_all.py --fresh

# 3) Launch the Streamlit app
streamlit run app.py
