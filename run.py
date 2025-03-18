import pandas as pd
import numpy as np
from src.data.unified_dataloader import load_dataset
from src.evaluation.five_fold_cross_validation import evaluate_model
from xgboost import XGBClassifier

# for i in ['census_income_kdd', 'default_credit']:
i = 'adult'
_, processed = load_dataset(i)
if i == 'census_income_kdd':
    sampled_data = processed.sample(frac=0.1, random_state=25)
    df = sampled_data.copy()
    X = df.drop('class', axis=1)
    y = df['class']
elif i == 'default_credit':
    df = processed.copy()
    X = df.drop('default_payment_next_month', axis=1)
    y = df['default_payment_next_month']
else:
    df = processed.copy()
    X = df.drop('income', axis=1)
    y = df['income']
print(f'run experiment on {i} dataset')
model = XGBClassifier()
evaluate_model(model=model, X_train=X, y_train=y, num_folds=5, dataset_name=i, fairshap_base='DR', matching_method='NN',thresh=0.05)