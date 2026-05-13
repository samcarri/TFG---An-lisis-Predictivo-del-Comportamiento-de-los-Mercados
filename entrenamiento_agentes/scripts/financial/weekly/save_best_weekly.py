"""
save_best_weekly.py

Entrena y guarda el mejor modelo weekly (Random Forest) en trained_models/
Incluye análisis detallado de métricas por clase.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    classification_report, roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/dataset_nvda_financial.csv"
MODEL_PATH = "trained_models/rf_weekly_model.joblib"
