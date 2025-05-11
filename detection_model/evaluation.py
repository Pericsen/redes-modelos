import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import dill as pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import warnings
warnings.filterwarnings('ignore')


# --- Cargar modelo ---
with open("complaint_classifier_model.pkl", "rb") as f:
    modelo = pickle.load(f)

# --- Cargar datos ---
df = pd.read_csv("./data/instagram/reclamo_clasificado/comments_ig_clasificados.csv")  # Asegurate que tiene 'comment_text' y 'label'
df.dropna(subset=["comment_text"], inplace=True)  # Eliminar filas sin texto
df.drop_duplicates(subset=["comment_text"], inplace=True)  # Eliminar duplicados
df.reset_index(drop=True, inplace=True)  # Reiniciar índice

# --- Features y etiquetas ---
X = df["comment_text"]
y_true = df["es_reclamo"]

# --- Predicciones ---
y_pred = modelo.predict(X)
y_proba = modelo.predict_proba(X)[:, 1]  # Probabilidad de clase 1

# --- Classification report y matriz de confusión ---
print("=== Classification Report ===")
print(classification_report(y_true, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_true, y_pred))

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# --- Precision-Recall Curve ---
precision, recall, _ = precision_recall_curve(y_true, y_proba)

plt.figure()
plt.plot(recall, precision, label="PR Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.show()
