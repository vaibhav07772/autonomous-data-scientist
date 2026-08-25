import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from xgboost import XGBRegressor, XGBClassifier
import joblib
import io
import sys
import contextlib

class CodeExecutor:
    def __init__(self):
        self.namespace = {
            'pd': pd, 'np': np, 'plt': plt, 'sns': sns,
            'RandomForestRegressor': RandomForestRegressor,
            'RandomForestClassifier': RandomForestClassifier,
            'LinearRegression': LinearRegression,
            'LogisticRegression': LogisticRegression,
            'XGBRegressor': XGBRegressor,
            'XGBClassifier': XGBClassifier,
            'train_test_split': train_test_split,
            'r2_score': r2_score,
            'mean_squared_error': mean_squared_error,
            'mean_absolute_error': mean_absolute_error,
            'accuracy_score': accuracy_score,
            'f1_score': f1_score,
            'joblib': joblib
        }

    def execute(self, code: str, df: pd.DataFrame):
        self.namespace['df'] = df
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            try:
                exec(code, self.namespace)
                output = f.getvalue()
                return {
                    "success": True,
                    "output": output,
                    "variables": {k: str(v) for k, v in self.namespace.items() if not k.startswith('_')}
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "output": f.getvalue()
                }