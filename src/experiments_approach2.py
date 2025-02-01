import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from typing import Tuple, List, Dict
from xgboost import XGBClassifier
import pdb
from sklearn.metrics import accuracy_score
from src.matching.ot_matcher import OptimalTransportPolicy
from src.matching.nn_matcher import NearestNeighborDataMatcher
from src.attribution import FairnessExplainer
from src.composition.data_composer import DataComposer
from src.attribution.oracle_metric import perturb_numpy_ver

EPSILON = 1e-20

class ExperimentNew:
    """ 
    The core part of how experiments work.
    
    ```
    Args:
        X_train_majority: pd.DataFrame,
        y_train_majority: pd.Series,
        X_train_minority: pd.DataFrame,
        y_train_minority: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        dataset_name: str
        base: string (default='majority'/'minority') 
                    if base='majority' then we will select some values from the minority group's data to replace values in the majority group's data.       
                    if base='minority' then we will select some values from the majority group's data to replace values in the minority group's data.
    
    """
    def __init__(self,
            model,
            X_train_majority: pd.DataFrame,
            y_train_majority: pd.Series,
            X_train_minority: pd.DataFrame,
            y_train_minority: pd.Series,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            dataset_name: str,
            change_group: str = 'majority'
            ):
        self.model = model
        self.X_train_majority = X_train_majority
        self.y_train_majority = y_train_majority
        self.X_train_minority = X_train_minority
        self.y_train_minority = y_train_minority

        self.X_test = X_test
        self.y_test = y_test
        self.dataset_name = dataset_name
        self.change_group = change_group
        if self.dataset_name == 'german_credit':
            self.sen_att_name = ['sex']
            self.original_DR = 0.05330166965723038
            self.gap = 1
        elif self.dataset_name == 'adult':
            self.sen_att_name = ['sex']
            self.original_DR = 0.02362159453332424
            self.gap = 10
        elif self.dataset_name == 'compas':
            pass

    def run_experiment(self):
        """
        Run the experiment.

        ```
        1. 从majority parity(此处男性)中随机选择30%, 50%, 70%的比例，作为将被替换的数据集X_train_replace_majority,剩余部分为X_train_rest_majority
        2. 将X_train_minority与X_train_replace_majority进行匹配
        3. 使用fairshap,把X_train_minority作为baseline dataset，找到X_train_replace_majority中需要替换的数据，假设总共需要替换n个数据点
        4. (1,n,20)根据这些,分别计算替换(1,n)中不同个数的结果,把需要替换的数据替换到X_train_replace_majority中,得到X_train_replace_majority_new
        5. 把X_train_replace_majority_new和X_train_rest_majority,还有X_train_minority合并,得到新的X_train_new，然后重新训练，得到新的模型model_new，计算新的DR值

        """
        # 1. 从majority parity(此处男性)中随机选择30%, 50%, 70%的比例，作为将被替换的数据集X_train_replace_majority,剩余部分为X_train_rest_majority
        proportion = 1
        X_train_replace_majority = self.X_train_majority.sample(frac=proportion, random_state=20)
        X_train_rest_majority = self.X_train_majority.drop(X_train_replace_majority.index)

        y_train_replace_majority = self.y_train_majority.loc[X_train_replace_majority.index]
        y_train_rest_majority = self.y_train_majority.drop(X_train_replace_majority.index)
        
        # 2. 初始化fairness_explainer
        sen_att_name = self.sen_att_name
        sen_att = [self.X_test.columns.get_loc(name) for name in sen_att_name]
        priv_val = [1]
        unpriv_dict = [list(set(self.X_test.values[:, sa])) for sa in sen_att]
        for sa_list, pv in zip(unpriv_dict, priv_val):
            sa_list.remove(pv)


        fairness_explainer_original = FairnessExplainer(
                model=self.model, 
                sen_att=sen_att, 
                priv_val=priv_val, 
                unpriv_dict=unpriv_dict
                )

        # 3. 将X_train_minority与X_train_replace_majority进行匹配
        if self.change_group == 'minority':
            matching = NearestNeighborDataMatcher(X_labeled=self.X_train_minority, X_unlabeled=X_train_replace_majority).match(n_neighbors=1)
            # 4. 使用fairshap,把X_train_minority作为baseline dataset，找到X_train_replace_majority中需要替换的数据，假设总共需要替换n个数据点
            fairness_shapley_value = fairness_explainer_original.shap_values(
                                        X = self.X_train_minority.values,
                                        X_baseline = X_train_replace_majority.values,
                                        matching=matching,
                                        sample_size=500,
                                        shap_sample_size="auto",
                                    )
            X_baseline = X_train_replace_majority.values
            X_change = self.X_train_minority.copy()
        elif self.change_group == 'majority':
            matching = NearestNeighborDataMatcher(X_labeled=X_train_replace_majority, X_unlabeled=self.X_train_minority).match(n_neighbors=1)
            # 4. 使用fairshap,把X_train_replace_majority作为baseline dataset，找到X_train_minority中需要替换的数据，假设总共需要替换n个数据点
            fairness_shapley_value = fairness_explainer_original.shap_values(
                                        X = X_train_replace_majority.values,
                                        X_baseline = self.X_train_minority.values,
                                        matching=matching,
                                        sample_size=500,
                                        shap_sample_size="auto",
                                    )  
            X_baseline = self.X_train_minority.values
            X_change = X_train_replace_majority.copy()
        # 5. 计算varphi, q          
        varphi = fix_negative_probabilities(fairness_shapley_value)
        non_zero_count =np.count_nonzero(varphi)
        print(f'总共可以替换的点数:{non_zero_count}')
        print(f'但是我们只替换其中的三分之一:{non_zero_count//3}')

        q = DataComposer(
                        x_counterfactual=X_baseline, 
                        joint_prob=matching, 
                        method="max").calculate_q()    # q是与X_train_replace_majority匹配的X_train_minority中的数据

        # 6. 用新数据重新训练，并且评估
        values_range = np.arange(1, non_zero_count//3, self.gap)
        after_values = []
        fairness_accuracy_pairs = []
        for action_number in values_range:
            # Step 1: 将 varphi 的值和位置展开为一维
            flat_varphi = [(value, row, col) for row, row_vals in enumerate(varphi)
                        for col, value in enumerate(row_vals)]
            # Step 2: 按值降序排序
            flat_varphi_sorted = sorted(flat_varphi, key=lambda x: x[0], reverse=True)
            # Step 3: 挑出前 action_number 个数的位置
            top_positions = flat_varphi_sorted[:action_number]

            # Step 4: 替换 X 中前三列的值为 S 中对应位置的值
            for value, row_idx, col_idx in top_positions:
                X_change.iloc[row_idx, col_idx] = q[row_idx, col_idx]        
            # Step 5: 合并数据集
            if self.change_group == 'minority':
                X_Train_New = pd.concat([X_train_replace_majority, X_train_rest_majority, X_change], axis=0)
                y_train_new = pd.concat([y_train_replace_majority, y_train_rest_majority, self.y_train_minority], axis=0)
            elif self.change_group == 'majority':
                X_Train_New = pd.concat([X_change, X_train_rest_majority, self.X_train_minority], axis=0)
                y_train_new = pd.concat([y_train_replace_majority, y_train_rest_majority, self.y_train_minority], axis=0)

            # Step 6: Train and evaluate model
            x = X_Train_New
            y = y_train_new
    
            model_new = XGBClassifier()
            model_new.fit(x, y)
            after = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, model_new)
            after_values.append(after)
            
            if after < self.original_DR:
                y_new_pred = model_new.predict(self.X_test)
                accuracy_new = accuracy_score(self.y_test, y_new_pred)
                fairness_accuracy_pairs.append((after, accuracy_new, action_number))  # Store both values as a tuple

        viz1(values_range, after_values, self.original_DR, proportion)
        viz2(fairness_accuracy_pairs, proportion)


        return after_values, fairness_accuracy_pairs
    
    

def fairness_value_function(sen_att, priv_val, unpriv_dict, X, model):
    X_disturbed = perturb_numpy_ver(
        X=X,
        sen_att=sen_att,
        priv_val=priv_val,
        unpriv_dict=unpriv_dict,
        ratio=1.0,
    )
    fx = model.predict_proba(X)[:, 1]
    fx_q = model.predict_proba(X_disturbed)[:, 1]
    return np.mean(np.abs(fx - fx_q))

def fix_negative_probabilities(varphi):
    """
    Fix the probability distribution by:
    1. Setting negative values to 0.
    2. Normalizing the probabilities to sum to 1.
    """
    # Step 1: Set negative values to 0
    varphi = np.maximum(varphi, 0)
    
    # Step 2: Normalize to make the sum equal to 1
    total_prob = varphi.sum()
    if total_prob == 0:
        raise ValueError("All probabilities are zero after fixing negative values.")
    
    varphi = varphi / total_prob
    return varphi

def viz1(values_range, after_values, original_DR, proportion):
    plt.figure(figsize=(10, 6))
    plt.scatter(values_range, after_values, label='New model', marker='x')
    plt.axhline(y=original_DR, color='r', linestyle='--', label='Original DR')
    plt.title(f'Proportion: {proportion}', fontsize=10)
    plt.xlabel('Limited actions')
    plt.ylabel('DR Value')
    plt.legend()
    plt.show()

def viz2(fairness_accuracy_pairs, proportion):
    # 假设 fairness_accuracy_pairs, proportion, num_new_data 已经定义
    fairness_values, accuracy_values, action_numbers = zip(*fairness_accuracy_pairs)
    # 创建颜色映射
    min_action = min(action_numbers)
    max_action = max(action_numbers)
    norm = plt.Normalize(min_action, max_action)
    # 创建从淡黄色到黑色的颜色映射
    cmap = plt.cm.YlOrBr  # YlOrBr colormap: 从淡黄色过渡到深褐色/黑色
    # 创建图像
    plt.figure(figsize=(10, 6))
    # 绘制散点图，使用圆形标记和颜色映射
    scatter = plt.scatter(accuracy_values, fairness_values, 
                        c=action_numbers, 
                        cmap=cmap, 
                        norm=norm,
                        marker='o',  # 使用圆形标记
                        alpha=0.6)

    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Action Number')

    plt.title(f'Proportion: {proportion}', fontsize=10)
    plt.xlabel('Accuracy')
    plt.ylabel('Fairness Value')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 显示图像
    plt.show()