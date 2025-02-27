from sklearn.model_selection import KFold
import numpy as np
from src.experiments.experiment import Experiment
from src.experiments.experiment_eo import Experiment as ExperimentEO
import pandas as pd
# 假设 model 是你的模型

def evaluate_model(model, X_train:pd.DataFrame, y_train:pd.Series, num_folds, dataset_name,fairshap_base='DR'):  
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=1)  # 5-fold 交叉验证
    scores = []  # 存储每次验证的评估指标（如 accuracy）
    i = 1

    for train_index, val_index in kf.split(X_train):
        print("-------------------------------------")
        print(f"-------------{i}th fold----------------")
        print("-------------------------------------")
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        
        # 训练模型
        model.fit(X_train_fold, y_train_fold)
        
        # 评估模型（这里假设用准确率评估）
        if fairshap_base == 'DR' or fairshap_base == 'DP':
            experiment = Experiment(model=model, X_train=X_train_fold, y_train=y_train_fold, X_test=X_val_fold, y_test=y_val_fold, dataset_name=dataset_name, fairshap_base=fairshap_base)
            experiment.run(ith_fold=i)
        elif fairshap_base == 'EO':
            experiment = ExperimentEO(model=model, X_train=X_train_fold, y_train=y_train_fold, X_test=X_val_fold, y_test=y_val_fold, dataset_name=dataset_name, fairshap_base=fairshap_base)
            experiment.run(ith_fold=i)
        i += 1


    pass
    # # 计算 5 次评估指标的平均值
    # mean_score = np.mean(scores)
    # print(f"5-Fold 交叉验证的平均得分: {mean_score:.4f}")
    # return mean_score



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

    model = XGBClassifier()  # 可以替换为 RandomForestClassifier() 等其他模型
    
    evaluate_model(model=model, X_train=X, y_train=y, num_folds=5, dataset_name='german_credit')



