# 🫀 10-Year Heart Attack Risk Prediction: A Data-Driven Machine Learning Study

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aframusarratdiya/heart-attack-risk-prediction/blob/main/Heart_disease_Project.ipynb)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)

An empirical healthcare classification study evaluating supervised machine learning models to assess a patient's 10-year risk of developing coronary heart disease based on demographic, behavioral, and clinical attributes[cite: 14, 15].

---

## 📌 Project Overview

Cardiovascular diseases are among the leading causes of mortality worldwide[cite: 15]. Early identification of high-risk individuals enables timely preventive lifestyle interventions and therapeutic care[cite: 15].

This project evaluates multiple predictive models on the Framingham Heart Study dataset[cite: 14, 15], examining:
* **The Clinical Accuracy–Recall Tradeoff:** High accuracy often masks critical diagnostic failures in highly imbalanced clinical data[cite: 15, 18, 19].
* **Multicollinearity & Risk Associations:** Inter-feature dependencies (such as systolic vs. diastolic blood pressure and smoking intensity) and their effect on risk prediction[cite: 15, 17].
* **Model Suitability for Screening:** Identifying which algorithms prioritize sensitivity (recall) to minimize missed diagnoses in clinical decision support[cite: 15, 18, 19].

---

## 📊 Dataset & Preprocessing Pipeline

* **Data Points:** 4,240 patient records with 15 clinical/behavioral features[cite: 14, 15].
* **Target Variable:** `Heart Disease (in next 10 years)` — Binary label (`0 = No`, `1 = Yes`)[cite: 14, 15].
* **Class Imbalance:** 3,596 negative cases (84.8%) vs. 644 positive cases (15.2%)[cite: 14, 15].
* **Partitioning:** 70% training (2,968 samples) / 30% testing (1,272 samples)[cite: 17].

### Preprocessing Protocol:
1. **Missing Value Imputation:** Imputed median values for clinical variables with missing entries (`glucose`: 388, `education`: 105, `BPMeds`: 53, `totChol`: 50, `cigsPerDay`: 29, `BMI`: 19, `heartRate`: 1)[cite: 14, 17].
2. **Categorical Encoding:** Mapped `gender` to binary integers (`Male = 1`, `Female = 0`)[cite: 14, 17].
3. **Feature Scaling:** Applied `MinMaxScaler` normalization to standardize numerical ranges across blood pressure, cholesterol, glucose, and heart rate[cite: 17].
4. **Multicollinearity & Correlation Analysis:**
   - Strong collinearity identified between `sysBP` and `diaBP` ($r = 0.78$) and between `cigsPerDay` and `currentSmoker` ($r = 0.76$)[cite: 15, 17].
   - Blood pressure parameters (`sysBP`, `diaBP`), age ($r = 0.23$), and prevalent hypertension ($r = 0.18$) showed the strongest direct correlations with 10-year heart disease risk[cite: 15, 17].

---

## 🔬 Benchmark Results

Evaluated on the 30% holdout test set (1,272 patients)[cite: 17, 18]:

| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | Clinical Triage Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Naive Bayes** | 0.829 | 0.402 | **0.231** | **0.293** | 0.71 | **Best Screening Model (Highest Recall & F1)**[cite: 18] |
| **Decision Tree** | 0.758 | 0.227 | **0.241** | 0.234 | 0.55 | Balanced but prone to variance[cite: 18] |
| **KNN** | 0.839 | 0.414 | 0.123 | 0.190 | 0.60 | Sensitive to distance distortion[cite: 17, 18] |
| **Logistic Regression** | **0.856** | **0.700** | 0.108 | 0.187 | **0.72** | High Accuracy, Poor Sensitivity[cite: 18] |
| **Neural Network (MLP)** | 0.848 | 0.556 | 0.051 | 0.094 | 0.64 | High specificity, misses positive cases[cite: 18] |

---

## 💡 Key Clinical & Engineering Insights

1. **Accuracy is a Misleading Metric in Medical Screening:**
   Logistic Regression (85.6%) and Neural Networks (84.8%) achieved the highest overall accuracy solely by predicting the majority class[cite: 18, 19]. However, their recalls (10.8% and 5.1%) mean they missed nearly 90–95% of high-risk patients[cite: 18, 19].
2. **Naive Bayes as the Strongest Screening Baseline:**
   Naive Bayes achieved the best balance of recall (0.231) and F1-score (0.293) among the non-resampled models[cite: 18], identifying significantly more positive risk cases[cite: 18, 19].
3. **Primary Predictive Drivers:**
   Systolic blood pressure (`sysBP`), age, hypertension history, and total cholesterol emerged as the leading physiological markers for long-term coronary risk[cite: 15, 17].

---

## 🛠️ Tech Stack

* **Language & Environment:** Python 3.10+, Google Colab / Jupyter Notebook[cite: 14]
* **Libraries:** `scikit-learn`[cite: 17], `pandas`[cite: 14], `numpy`[cite: 14], `seaborn`[cite: 14], `matplotlib`[cite: 14]

---

## 📖 Research Context & Citation

📄 **Read the full research paper:** [Heart_disease_prediction_report.pdf](./Heart_disease_prediction_report.pdf)[cite: 15, 20]

> **"Heart Attack Prediction Using Machine Learning: A Data-Driven Approach to Identifying Risks"**[cite: 15]  
> *Syed Rahin Reza, Rehman Islam, Raheek Muhammad Raiyan, Afra Musarrat Diya* — Department of Computer Science, BRAC University[cite: 15].
