# =========================================
# IMPORT LIBRARIES
# =========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score
)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import os
import datetime
import io

# =========================================
# FIX RANDOMNESS (Reproducibility)
# =========================================
import random
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# =========================================
# HELPER: ROC CURVE -> TENSORBOARD IMAGE
# =========================================
def log_multiclass_roc_to_tensorboard(y_true, y_proba, n_classes, writer, tag="ROC/Curve", step=0):
    """
    Logs multi-class ROC (OvR) as an IMAGE to TensorBoard and macro AUC as scalar.
    y_true  : shape (N,) int labels
    y_proba : shape (N, n_classes) probabilities
    """
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    # Plot ROC
    fig = plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} AUC={roc_auc[i]:.2f}")
    plt.plot(all_fpr, mean_tpr, "k--", label=f"Macro Avg AUC={macro_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k:", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Class ROC (OvR)")
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Convert plot to image tensor
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    img = tf.image.decode_png(buf.getvalue(), channels=4)
    img = tf.expand_dims(img, 0)  # (1, H, W, C)

    with writer.as_default():
        tf.summary.image(tag, img, step=step)
        tf.summary.scalar(tag.replace("ROC/", "AUC/"), macro_auc, step=step)

    return macro_auc


# =========================================
# LOAD DATASET
# =========================================
print("\n===== LOADING DATASET =====")
base_path = "/Users/ayushsingh/Downloads/human+activity+recognition+using+smartphones/UCI HAR Dataset/"

X_train = pd.read_csv(base_path + "train/X_train.txt", sep=r'\s+', header=None)
X_test  = pd.read_csv(base_path + "test/X_test.txt", sep=r'\s+', header=None)
y_train = pd.read_csv(base_path + "train/y_train.txt", header=None)
y_test  = pd.read_csv(base_path + "test/y_test.txt", header=None)

print("Training Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# =========================================
# EDA
# =========================================
print("\n===== EXPLORATORY DATA ANALYSIS =====")
plt.figure(figsize=(6,4))
sns.countplot(x=y_train.values.ravel())
plt.title("Training Labels Distribution")
plt.xlabel("Activity Class")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(12,8))
sample_corr = pd.DataFrame(X_train).sample(n=500, random_state=42).corr()
sns.heatmap(sample_corr, cmap='coolwarm', cbar=True)
plt.title("Correlation Heatmap of Features (sampled 500 rows)")
plt.show()

# =========================================
# SCALING & ENCODING
# =========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train.values.ravel())
y_test_encoded  = encoder.transform(y_test.values.ravel())

n_classes = len(np.unique(y_test_encoded))

# =========================================
# PCA + SVM
# =========================================
print("\n===== PCA + SVM =====")
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

svm_pca = SVC(kernel='rbf', C=10, gamma=0.001)
svm_pca.fit(X_train_pca, y_train_encoded)
y_pred_pca_svm = svm_pca.predict(X_test_pca)
pca_svm_acc = svm_pca.score(X_test_pca, y_test_encoded)

print("PCA + SVM Accuracy:", pca_svm_acc)
print(classification_report(y_test_encoded, y_pred_pca_svm))
sns.heatmap(confusion_matrix(y_test_encoded, y_pred_pca_svm), annot=True, fmt='d', cmap='Blues')
plt.title("PCA+SVM Confusion Matrix")
plt.show()

# =========================================
# LOGISTIC REGRESSION
# =========================================
print("\n===== LOGISTIC REGRESSION =====")
lr = LogisticRegression(C=10, max_iter=5000, n_jobs=-1)
lr.fit(X_train_scaled, y_train_encoded)
y_pred_lr = lr.predict(X_test_scaled)
lr_acc = lr.score(X_test_scaled, y_test_encoded)

print("Logistic Accuracy:", lr_acc)
print(classification_report(y_test_encoded, y_pred_lr))
sns.heatmap(confusion_matrix(y_test_encoded, y_pred_lr), annot=True, fmt='d', cmap='Greens')
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# =========================================
# ULTRA TUNED SVM
# =========================================
print("\n===== ULTRA TUNED SVM =====")
param_grid = {'C':[10,50,100],'gamma':[0.001,0.0005,0.0001],'kernel':['rbf']}
grid_svm = GridSearchCV(SVC(probability=True), param_grid, cv=5, n_jobs=-1, verbose=1)
grid_svm.fit(X_train_scaled, y_train_encoded)

best_svm = grid_svm.best_estimator_
y_pred_svm = best_svm.predict(X_test_scaled)
svm_acc = best_svm.score(X_test_scaled, y_test_encoded)

print("Best Parameters:", grid_svm.best_params_)
print("Ultra Tuned SVM Accuracy:", svm_acc)
print(classification_report(y_test_encoded, y_pred_svm))
sns.heatmap(confusion_matrix(y_test_encoded, y_pred_svm), annot=True, fmt='d', cmap='Oranges')
plt.title("Ultra Tuned SVM Confusion Matrix")
plt.show()

# =========================================
# OPTIMIZED XGBOOST
# =========================================
print("\n===== OPTIMIZED XGBOOST =====")
xgb = XGBClassifier(
    n_estimators=1000, max_depth=4, learning_rate=0.03,
    subsample=0.9, colsample_bytree=0.9, gamma=0,
    reg_lambda=1, objective='multi:softprob',
    eval_metric='mlogloss', tree_method='hist', random_state=42
)
xgb.fit(X_train_scaled, y_train_encoded)
y_pred_xgb = xgb.predict(X_test_scaled)
xgb_acc = xgb.score(X_test_scaled, y_test_encoded)

print("Optimized XGBoost Accuracy:", xgb_acc)
print(classification_report(y_test_encoded, y_pred_xgb))
sns.heatmap(confusion_matrix(y_test_encoded, y_pred_xgb), annot=True, fmt='d', cmap='Reds')
plt.title("XGBoost Confusion Matrix")
plt.show()

# =========================================
# IMPROVED DEEP ANN WITH TENSORBOARD
# =========================================
print("\n===== IMPROVED DEEP ANN =====")

log_dir = "logs/fit"
os.makedirs(log_dir, exist_ok=True)

run_log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb = TensorBoard(log_dir=run_log_dir, histogram_freq=1)

deep_model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y_train_encoded)), activation='softmax')
])

deep_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

deep_model.fit(
    X_train_scaled,
    y_train_encoded,
    epochs=80,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, tensorboard_cb],
    verbose=0
)

_, deep_ann_acc = deep_model.evaluate(X_test_scaled, y_test_encoded, verbose=0)
deep_ann_proba = deep_model.predict(X_test_scaled, verbose=0)
y_pred_deep = np.argmax(deep_ann_proba, axis=1)

print("Deep ANN Accuracy:", deep_ann_acc)
print(classification_report(y_test_encoded, y_pred_deep))
sns.heatmap(confusion_matrix(y_test_encoded, y_pred_deep), annot=True, fmt='d', cmap='Purples')
plt.title("Deep ANN Confusion Matrix")
plt.show()

print(f"To view TensorBoard, run: tensorboard --logdir {log_dir}")

# =========================================
# SOFT VOTING ENSEMBLE
# =========================================
print("\n===== SOFT VOTING ENSEMBLE =====")
voting_model = VotingClassifier(
    estimators=[('lr', lr), ('svm', best_svm), ('xgb', xgb)],
    voting='soft', n_jobs=-1
)
voting_model.fit(X_train_scaled, y_train_encoded)
y_pred_voting = voting_model.predict(X_test_scaled)
voting_acc = voting_model.score(X_test_scaled, y_test_encoded)
voting_proba = voting_model.predict_proba(X_test_scaled)

print("Soft Voting Accuracy:", voting_acc)
print(classification_report(y_test_encoded, y_pred_voting))
sns.heatmap(confusion_matrix(y_test_encoded, y_pred_voting), annot=True, fmt='d', cmap='cool')
plt.title("Soft Voting Confusion Matrix")
plt.show()

# =========================================
# FAST STACKING CLASSIFIER
# =========================================
print("\n===== FAST STACKING CLASSIFIER =====")
fast_stack = StackingClassifier(
    estimators=[('lr', lr), ('xgb', xgb)],
    final_estimator=RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    cv=3, n_jobs=-1, passthrough=True
)
fast_stack.fit(X_train_scaled, y_train_encoded)
y_pred_stack = fast_stack.predict(X_test_scaled)
stack_acc = fast_stack.score(X_test_scaled, y_test_encoded)

print("Fast Stacking Accuracy:", stack_acc)
print(classification_report(y_test_encoded, y_pred_stack))
sns.heatmap(confusion_matrix(y_test_encoded, y_pred_stack), annot=True, fmt='d', cmap='magma')
plt.title("Fast Stacking Confusion Matrix")
plt.show()

# =========================================
# TENSORBOARD LOGGING FOR ALL ML MODELS
# =========================================
log_dir_all = "logs/full_project/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir_all, exist_ok=True)
writer = tf.summary.create_file_writer(log_dir_all)

accuracies = [
    pca_svm_acc,
    lr_acc,
    svm_acc,
    xgb_acc,
    deep_ann_acc,
    voting_acc,
    stack_acc
]

with writer.as_default():
    for i, acc in enumerate(accuracies):
        tf.summary.scalar("Comparison/Accuracy", acc, step=i)

# Logistic CV curve
lr_cv = cross_val_score(
    LogisticRegression(C=10, max_iter=5000),
    X_train_scaled,
    y_train_encoded,
    cv=5,
    n_jobs=1
)
with writer.as_default():
    for i, score in enumerate(lr_cv):
        tf.summary.scalar("ML_Models/Logistic_CV", score, step=i)

# PCA + SVM CV curve
svm_pca_cv = cross_val_score(
    SVC(kernel='rbf', C=10, gamma=0.001),
    X_train_pca,
    y_train_encoded,
    cv=5,
    n_jobs=1
)
with writer.as_default():
    for i, score in enumerate(svm_pca_cv):
        tf.summary.scalar("ML_Models/PCA_SVM_CV", score, step=i)

# Ultra Tuned SVM CV curve
svm_cv = cross_val_score(
    best_svm,
    X_train_scaled,
    y_train_encoded,
    cv=5,
    n_jobs=1
)
with writer.as_default():
    for i, score in enumerate(svm_cv):
        tf.summary.scalar("ML_Models/Ultra_SVM_CV", score, step=i)

# XGBoost CV curve
xgb_cv = cross_val_score(
    xgb,
    X_train_scaled,
    y_train_encoded,
    cv=5,
    n_jobs=1
)
with writer.as_default():
    for i, score in enumerate(xgb_cv):
        tf.summary.scalar("ML_Models/XGBoost_CV", score, step=i)

# =========================================
# ✅ ROC CURVE LOGGING TO TENSORBOARD (IMAGE)
#    1) Soft Voting ROC
#    2) Deep ANN ROC
# =========================================
soft_auc = log_multiclass_roc_to_tensorboard(
    y_test_encoded, voting_proba, n_classes, writer, tag="ROC/SoftVoting", step=0
)
ann_auc = log_multiclass_roc_to_tensorboard(
    y_test_encoded, deep_ann_proba, n_classes, writer, tag="ROC/DeepANN", step=0
)

writer.flush()
writer.close()

print("\nTo view FULL PROJECT TensorBoard run:")
print("tensorboard --logdir logs/full_project")
print("SoftVoting Macro AUC:", soft_auc)
print("DeepANN   Macro AUC:", ann_auc)

# =========================================
# FINAL MODEL COMPARISON GRAPH
# =========================================
models = ["PCA + SVM","Logistic","Ultra SVM","XGBoost","Deep ANN","Soft Voting","Stacking"]
accuracies = [pca_svm_acc, lr_acc, svm_acc, xgb_acc, deep_ann_acc, voting_acc, stack_acc]

f1_scores = [
    f1_score(y_test_encoded, y_pred_pca_svm, average='weighted'),
    f1_score(y_test_encoded, y_pred_lr, average='weighted'),
    f1_score(y_test_encoded, y_pred_svm, average='weighted'),
    f1_score(y_test_encoded, y_pred_xgb, average='weighted'),
    f1_score(y_test_encoded, y_pred_deep, average='weighted'),
    f1_score(y_test_encoded, y_pred_voting, average='weighted'),
    f1_score(y_test_encoded, y_pred_stack, average='weighted')
]

precisions = [
    precision_score(y_test_encoded, y_pred_pca_svm, average='weighted'),
    precision_score(y_test_encoded, y_pred_lr, average='weighted'),
    precision_score(y_test_encoded, y_pred_svm, average='weighted'),
    precision_score(y_test_encoded, y_pred_xgb, average='weighted'),
    precision_score(y_test_encoded, y_pred_deep, average='weighted'),
    precision_score(y_test_encoded, y_pred_voting, average='weighted'),
    precision_score(y_test_encoded, y_pred_stack, average='weighted')
]

recalls = [
    recall_score(y_test_encoded, y_pred_pca_svm, average='weighted'),
    recall_score(y_test_encoded, y_pred_lr, average='weighted'),
    recall_score(y_test_encoded, y_pred_svm, average='weighted'),
    recall_score(y_test_encoded, y_pred_xgb, average='weighted'),
    recall_score(y_test_encoded, y_pred_deep, average='weighted'),
    recall_score(y_test_encoded, y_pred_voting, average='weighted'),
    recall_score(y_test_encoded, y_pred_stack, average='weighted')
]

x = np.arange(len(models))
width = 0.2
plt.figure(figsize=(14,7))
plt.bar(x - 1.5*width, accuracies, width, label='Accuracy')
plt.bar(x - 0.5*width, f1_scores, width, label='F1 Score')
plt.bar(x + 0.5*width, precisions, width, label='Precision')
plt.bar(x + 1.5*width, recalls, width, label='Recall')

plt.xticks(x, models, rotation=25)
plt.ylabel("Score")
plt.ylim(0,1.05)
plt.title("Performance Comparison of All Models - UCI HAR")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
