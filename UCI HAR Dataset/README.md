# 📱 Human Activity Recognition using Machine Learning

## 📌 Project Overview
This project focuses on classifying human activities using smartphone sensor data from the **UCI Human Activity Recognition (HAR) Dataset**. The goal is to predict activities like walking, sitting, and standing using machine learning models.

---

## 📊 Dataset Information
The dataset contains data collected from **30 volunteers (age 19–48)** performing different activities while wearing a smartphone on their waist.

### 📡 Sensor Details:
- Accelerometer (3-axis)
- Gyroscope (3-axis)
- Sampling rate: **50Hz**

### 🧠 Activities:
- Walking  
- Walking Upstairs  
- Walking Downstairs  
- Sitting  
- Standing  
- Laying  

### 📈 Data Features:
- 561 features (time + frequency domain)
- Sliding window: 2.56 seconds
- 70% training / 30% testing split  

📎 Dataset Reference: :contentReference[oaicite:0]{index=0}

---

## 🎯 Objectives
- Perform data preprocessing and scaling
- Apply multiple machine learning models
- Optimize models using hyperparameter tuning
- Compare performance using evaluation metrics

---

## ⚙️ Tech Stack
- Python
- NumPy, Pandas
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Matplotlib, Seaborn

---

## 🤖 Models Implemented
- Logistic Regression
- Support Vector Machine (SVM)
- PCA + SVM
- XGBoost Classifier
- Artificial Neural Network (ANN)
- Voting Ensemble Model

---

## 📈 Results
| Model                  | Accuracy |
|-----------------------|---------|
| Logistic Regression   | ~95%    |
| SVM (Tuned)           | ~95.7%  |
| XGBoost               | ~95%    |
| ANN                   | ~94–96% |
| Voting Ensemble       | **~96.4% (Best)** |

---

## 📊 Evaluation Metrics
- Confusion Matrix
- ROC Curve & AUC
- Accuracy Comparison Graph
- TensorBoard Visualizations

---

## 📸 Output Screenshots

### 🔹 Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

### 🔹 ROC Curve
![ROC Curve](images/roc_curve.png)

### 🔹 Accuracy Comparison
![Accuracy](images/accuracy_graph.png)

---

## 🚀 Key Features
- Multi-model comparison
- Hyperparameter tuning (GridSearchCV)
- Ensemble learning for improved accuracy
- Clean and modular ML pipeline

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/human-activity-recognition-ml.git