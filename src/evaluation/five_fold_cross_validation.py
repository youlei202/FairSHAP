from sklearn.model_selection import KFold
import numpy as np
from src.experiments.experiment import Experiment
from src.experiments.experiment_eo import Experiment as ExperimentEO
from src.experiments.experiment_new_eo import Experiment as ExperimentNewEO
import pandas as pd



def evaluate_model(model, X_train:pd.DataFrame, y_train:pd.Series, num_folds, dataset_name, fairshap_base='DR',matching_method='OT', thresh=0.05):  
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=1)  # 5-fold cross-validation
    scores = []  # Store evaluation metrics (e.g., accuracy) for each fold
    i = 1

    for train_index, val_index in kf.split(X_train):
        print("-------------------------------------")
        print(f"-------------{i}th fold----------------")
        print("-------------------------------------")
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        
        # Train the model
        model.fit(X_train_fold, y_train_fold)
        
        # evaluate the model
        if fairshap_base == 'DR' or fairshap_base == 'DP':
            experiment = Experiment(model=model, X_train=X_train_fold, y_train=y_train_fold, X_test=X_val_fold, y_test=y_val_fold, dataset_name=dataset_name, fairshap_base=fairshap_base, matching_method=matching_method)
            experiment.run(ith_fold=i, threshold=thresh)
        elif fairshap_base == 'EO':
            experiment = ExperimentNewEO(model=model, X_train=X_train_fold, y_train=y_train_fold, X_test=X_val_fold, y_test=y_val_fold, dataset_name=dataset_name, fairshap_base=fairshap_base)
            experiment.run(ith_fold=i)
        i += 1


    pass
    # Calculate the average of the evaluation metrics over all folds
    mean_score = np.mean(scores)
    print(f"Average score across {num_folds}-Fold Cross Validation: {mean_score:.4f}")
    return mean_score



if __name__ == '__main__':
    from data.unified_dataloader import load_dataset

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score

    a, processed_german_credit = load_dataset('german_credit')
    '''German Credit dataset'''
    df = processed_german_credit.copy()
    X = df.drop('Risk', axis=1)
    y = df['Risk']

    model = XGBClassifier()  # XGboost can be replaced by RandomForestClassifier() tc.
    
    evaluate_model(model=model, X_train=X, y_train=y, num_folds=5, dataset_name='german_credit')



