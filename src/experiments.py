import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from typing import Tuple, List, Dict
from xgboost import XGBClassifier
import pdb
import logging
from sklearn.metrics import accuracy_score
from src.matching.ot_matcher import OptimalTransportPolicy
from src.matching.nn_matcher import NearestNeighborDataMatcher
from src.attribution import FairnessExplainer
from src.composition.data_composer import DataComposer
from src.attribution.oracle_metric import perturb_numpy_ver

EPSILON = 1e-20

class Experiment:
    """ 
    The core part of how experiments work.
    X_train_majority_label1 match with X_train_minority_label1
    X_train_majority_label0 match with X_train_minority_label0
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
            original_Xtest_DR,
            original_Xtest_acc,

            ):
        self.model = model
        self.X_train_majority = X_train_majority
        self.y_train_majority = y_train_majority
        self.X_train_minority = X_train_minority
        self.y_train_minority = y_train_minority
        self.X_test = X_test
        self.y_test = y_test
        self.dataset_name = dataset_name

        self.original_Xtest_DR = original_Xtest_DR
        self.original_Xtest_acc = original_Xtest_acc
        if self.dataset_name == 'german_credit':
            self.sen_att_name = ['sex']
            self.gap = 1
        elif self.dataset_name == 'adult':
            self.sen_att_name = ['sex']
            self.gap = 1
        elif self.dataset_name == 'compas':
            self.sen_att_name = ['sex']
            self.gap = 1
        elif self.dataset_name == 'compas4race':
            self.sen_att_name = ['race']
            self.gap = 1
        elif self.dataset_name == 'census_income':
            self.sen_att_name = ['sex']
            self.gap = 1
        elif self.dataset_name == 'default_credit':
            self.sen_att_name = ['sex']
            self.gap = 1
        else :
            raise ValueError('The dataset name is not supported')

    def run_experiment(self):
        """
        Run the experiment.

        ```
        1. 把X_train_majority和X_train_minority分别分成X_train_majority_label1, X_train_majority_label0, X_train_minority_label1, X_train_minority_label0
        2. 初始化FairnessExplainer
        """

        # 1. 把X_train_majority和X_train_minority分别分成X_train_majority_label1, X_train_majority_label0, X_train_minority_label1, X_train_minority_label0
        y_train_majority_label1 = self.y_train_majority[self.y_train_majority == 1]
        y_train_majority_label0 = self.y_train_majority[self.y_train_majority == 0]
        y_train_minority_label1 = self.y_train_minority[self.y_train_minority == 1]
        y_train_minority_label0 = self.y_train_minority[self.y_train_minority == 0]

        X_train_majority_label0 = self.X_train_majority.loc[y_train_majority_label0.index]
        X_train_majority_label1 = self.X_train_majority.loc[y_train_majority_label1.index]
        X_train_minority_label0 = self.X_train_minority.loc[y_train_minority_label0.index]
        X_train_minority_label1 = self.X_train_minority.loc[y_train_minority_label1.index]


        print('2. 初始化FairnessExplainer')
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
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

        print('--------接下来先对minority group进行修改--------')
        print('3(a). 将X_train_minority_label0与X_train_majority_label0进行匹配')
        matching_minority_label0 = NearestNeighborDataMatcher(X_labeled=X_train_minority_label0, X_unlabeled=X_train_majority_label0).match(n_neighbors=1)
        print('3(b). 将X_train_minority_label1与X_train_majority_label1进行匹配')
        matching_minority_label1 = NearestNeighborDataMatcher(X_labeled=X_train_minority_label1, X_unlabeled=X_train_majority_label1).match(n_neighbors=1)
        print('4(a). 使用fairshap, 从 X_train_majority_label0中找到合适的值替换X_train_minority_label0中的数据')
        fairness_shapley_minority_value_label0 = fairness_explainer_original.shap_values(
                                    X = X_train_minority_label0.values,
                                    X_baseline = X_train_majority_label0.values,
                                    matching=matching_minority_label0,
                                    sample_size=1000,
                                    shap_sample_size="auto",
                                )
        X_change_minority_label0 = X_train_minority_label0.copy()
        X_base_minority_label0 = X_train_majority_label0
        print('4(b). 使用fairshap, 从 X_train_majority_label1中找到合适的值替换X_train_minority_label1中的数据')
        fairness_shapley_minority_value_label1 = fairness_explainer_original.shap_values(
                                    X = X_train_minority_label1.values,
                                    X_baseline = X_train_majority_label1.values,
                                    matching=matching_minority_label1,
                                    sample_size=1000,
                                    shap_sample_size="auto",
                                )
        
        X_change_minority_label1 = X_train_minority_label1.copy()
        X_base_minority_label1 = X_train_majority_label1


        print('5. 计算出varphi和q')
        # 筛选出shapley value大于0.1的值，其他值设为0，然后归一化
        varphi_minority_label0 = fix_negative_probabilities_select_larger(fairness_shapley_minority_value_label0)
        varphi_minority_label1 = fix_negative_probabilities_select_larger(fairness_shapley_minority_value_label1)
        varphi_minority = np.vstack((varphi_minority_label0, varphi_minority_label1))
        non_zero_count_minority =np.count_nonzero(varphi_minority_label0) + np.count_nonzero(varphi_minority_label1)
        print(f"在X_train_minority中shapely value中大于0.1的值的个数有: {non_zero_count_minority}")
        q_minority_label0 = DataComposer(
                        x_counterfactual=X_base_minority_label0.values, 
                        joint_prob=matching_minority_label0, 
                        method="max").calculate_q() 
        q_minority_label1 = DataComposer(
                        x_counterfactual=X_base_minority_label1.values, 
                        joint_prob=matching_minority_label1, 
                        method="max").calculate_q()
        q_minority = np.vstack((q_minority_label0, q_minority_label1))




        print('--------接下来对majority group进行修改--------')
        print('3(a). 将X_train_majority_label0与X_train_minority_label0进行匹配')
        matching_majority_label0 = NearestNeighborDataMatcher(X_labeled=X_train_majority_label0, X_unlabeled=X_train_minority_label0).match(n_neighbors=1)
        print('3(b). 将X_train_majority_label1与X_train_minority_label1进行匹配')
        matching_majority_label1 = NearestNeighborDataMatcher(X_labeled=X_train_majority_label1, X_unlabeled=X_train_minority_label1).match(n_neighbors=1)

        print('4(a). 使用fairshap, 从 X_train_minority_label0中找到合适的值替换X_train_majority_label0中的数据')
        fairness_shapley_majority_value_label0 = fairness_explainer_original.shap_values(
                                    X = X_train_majority_label0.values,
                                    X_baseline = X_train_minority_label0.values,
                                    matching=matching_majority_label0,
                                    sample_size=1000,
                                    shap_sample_size="auto",
                                )
        X_change_majority_label0 = X_train_majority_label0.copy()
        X_base_majority_label0 = X_train_minority_label0
        print('4(b). 使用fairshap, 从 X_train_minority_label1中找到合适的值替换X_train_majority_label1中的数据')
        fairness_shapley_majority_value_label1 = fairness_explainer_original.shap_values(
                                    X = X_train_majority_label1.values,
                                    X_baseline = X_train_minority_label1.values,
                                    matching=matching_majority_label1,
                                    sample_size=1000,
                                    shap_sample_size="auto",
                                )  
        X_change_majority_label1 = X_train_majority_label1.copy()
        X_base_majority_label1 = X_train_minority_label1
            
        print('5. 计算出varphi和q')
        # 筛选出shapley value大于0.1的值，其他值设为0，然后归一化
        varphi_majority_label0 = fix_negative_probabilities_select_larger(fairness_shapley_majority_value_label0)
        varphi_majority_label1 = fix_negative_probabilities_select_larger(fairness_shapley_majority_value_label1)
        varphi_majority = np.vstack((varphi_majority_label0, varphi_majority_label1))
        non_zero_count_majority =np.count_nonzero(varphi_majority_label0) + np.count_nonzero(varphi_majority_label1)
        # viz_varphi(varphi=fairness_shapley_value_label0)
        # viz_varphi(varphi=fairness_shapley_value_label1)
        print(f"在X_train_majority中shapely value中大于0.1的值的个数有: {non_zero_count_majority}")
        q_majority_label0 = DataComposer(
                        x_counterfactual=X_base_majority_label0.values, 
                        joint_prob=matching_majority_label0, 
                        method="max").calculate_q() 
        q_majority_label1 = DataComposer(
                        x_counterfactual=X_base_majority_label1.values, 
                        joint_prob=matching_majority_label1, 
                        method="max").calculate_q()
        q_majority = np.vstack((q_majority_label0, q_majority_label1))

        varphi = np.vstack((varphi_minority, varphi_majority))
        q = np.vstack((q_minority,q_majority))   # q_minority_label0 + q_minority_label1 + q_majority_label0 + q_majority_label1
        X_change = pd.concat([X_change_minority_label0, X_change_minority_label1, X_change_majority_label0, X_change_majority_label1], axis=0)
        non_zero_count = non_zero_count_majority + non_zero_count_minority

        print(f'6. 开始整理minority部分的修改和majority部分的修改并且合并新数据,共修改{non_zero_count}个数据点, 使用new training set训练新模型')
        values_range = np.arange(1, non_zero_count, self.gap)
        after_values_on_test_set = []
        after_values_on_train_set = []
        fairness_accuracy_pairs = []
        changed_value_info = []   # 存储修改的值的信息(column, before_value, after_value)
        for action_number in values_range:
            # Step 1: 将 varphi 的值和位置展开为一维
            flat_varphi = [(value, row, col) for row, row_vals in enumerate(varphi)
                        for col, value in enumerate(row_vals)]
            # Step 2: 按值降序排序
            flat_varphi_sorted = sorted(flat_varphi, key=lambda x: x[0], reverse=True)
            # Step 3: 挑出前 action_number 个数的位置
            top_positions = flat_varphi_sorted[:action_number]
            
            # # 有change_group的时候记录结果的
            # if action_number == non_zero_count-1:  
            #     for value, row_idx, col_idx in top_positions:
            #         before_value = self.X_train_majority.iloc[row_idx, col_idx]
            #         after_value = q[row_idx, col_idx]
            #         changed_value_info.append((col_idx, before_value, after_value))  # 存储修改的值的信息

            # Step 4: 替换 X 中前三列的值为 S 中对应位置的值
            for value, row_idx, col_idx in top_positions:
                X_change.iloc[row_idx, col_idx] = q[row_idx, col_idx]

            x = X_change
            y = pd.concat([y_train_minority_label0, y_train_minority_label1, y_train_majority_label0, y_train_majority_label1], axis=0)

            # Step 6: Train and evaluate model            
            model_new = XGBClassifier()
            model_new.fit(x, y)
            after = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, model_new)
            after_values_on_test_set.append(after)
            # after_value = fairness_value_function(sen_att, priv_val, unpriv_dict, x.values, model_new)
            # after_values_on_train_set.append(after_value)

            
            if after < self.original_Xtest_DR:
                y_new_pred = model_new.predict(self.X_test)
                accuracy_new = accuracy_score(self.y_test, y_new_pred)
                fairness_accuracy_pairs.append((after, accuracy_new, action_number))  # Store both values as a tuple
        # print(f'changed_value_info.shape: {len(changed_value_info)}')
        #修改不同位置后训练的new_model在相应修改后的training set上的DR值
        # viz1(values_range, after_values_on_train_set, self.original_Xtrain_DR, title='new_model\'s DR on training set')
        #修改不同位置后训练的new_model在test set上的DR值
        viz1(values_range, after_values_on_test_set, self.original_Xtest_DR, title='new_model\'s DR on test set')
        viz2(fairness_accuracy_pairs, self.original_Xtest_acc, title='Accuracy vs. DR')

        changed_value_info_df = pd.DataFrame(changed_value_info, columns=['Column', 'Before Value', 'After Value'])
        changed_value_info_df.to_csv('changed_value_info.csv', index=True)


        return after_values_on_test_set, fairness_accuracy_pairs
    
    

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


def fix_negative_probabilities_select_larger(varphi):
    """
    Fix the probability distribution by:
    1. Filtering out values less than or equal to 0.1.
    2. Normalizing the probabilities to sum to 1.
    """

    varphi = np.where(varphi > 0.1, varphi, 0)
    # varphi = np.where(varphi < - 0.1, varphi, 0)
    # varphi = np.abs(varphi)

    total_prob = varphi.sum()
    if total_prob == 0:
        raise ValueError("All probabilities are zero after filtering values <= -0.1.")
    varphi = varphi / total_prob
    return varphi


def viz_varphi(varphi):
    '''
    绘制varphi的分布图
    '''
    varphi_flat = varphi.flatten()
    # 去掉varphi中小于0.001的值
    varphi_flat = varphi_flat[varphi_flat >= 0.001]

    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(varphi_flat, bins=50, edgecolor='black', alpha=0.7)
    # 在柱状图上显示数值
    for count, bin_edge in zip(counts, bins):
        plt.text(bin_edge, count, str(int(count)), ha='center', va='bottom')

    plt.title('Distribution of varphi values')
    plt.xlabel('Value')
    plt.xlim(0.01,0.5)
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()



def viz1(values_range, after_values, original_DR, title):
    '''
    绘制修改后的DR值与修改的action number的散点图
    '''
    plt.figure(figsize=(10, 6))
    plt.scatter(values_range, after_values, label='New model', marker='x')
    plt.axhline(y=original_DR, color='r', linestyle='--', label='Original DR')
    plt.title(title, fontsize=10)
    plt.xlabel('Limited actions')
    plt.ylabel('DR Value')
    plt.legend()
    plt.show()



def viz2(fairness_accuracy_pairs, original_acc, title):
    '''
    绘制DR小于baseline的点数的散点图，横坐标为Accuracy，纵坐标为DR，颜色深浅表示Action Number
    '''
    fairness_values, accuracy_values, action_numbers = zip(*fairness_accuracy_pairs)    
    # create a colormap
    min_action = min(action_numbers)
    max_action = max(action_numbers)
    norm = plt.Normalize(min_action, max_action)
    cmap = plt.cm.YlOrBr  # YlOrBr colormap: 从淡黄色过渡到深褐色/黑色
    
    # create the figure
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(accuracy_values, fairness_values, 
                          c=action_numbers, 
                          cmap=cmap, 
                          norm=norm,
                          marker='o',  # 使用圆形标记
                          alpha=0.6)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Action Number')
    plt.axvline(x=original_acc, color='r', linestyle='--', label='Original Accuracy')
    plt.title(title, fontsize=10)
    plt.xlabel('Accuracy')
    plt.ylabel('Fairness Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()