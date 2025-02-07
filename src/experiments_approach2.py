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
            self.original_Xtrain_DR = 0.044362135231494904
            self.original_Xtest_DR = 0.05330166965723038
            self.original_Xtest_acc = 0.6766666666666666
            
            self.gap = 1
        elif self.dataset_name == 'adult':
            self.sen_att_name = ['sex']
            self.original_Xtrain_DR = 0.024025436490774155
            self.original_Xtest_DR = 0.02362159453332424
            self.original_Xtest_acc = 0.8749104309550619
            self.gap = 1
        elif self.dataset_name == 'compas':
            self.sen_att_name = ['sex']
            self.original_Xtest_DR = 0.09081892669200897
            self.original_Xtest_acc = 0.6609699769053118
            self.gap = 1
        elif self.dataset_name == 'compas4race':
            self.sen_att_name = ['race']
            self.original_Xtrain_DR =0.09234681725502014
            self.original_Xtest_DR = 0.09175542742013931
            self.original_Xtest_acc = 0.6742547425474255

            self.gap = 1
        elif self.dataset_name == 'census_income':
            self.sen_att_name = ['sex']
            self.original_Xtest_DR = 0.08531209826469421
            self.original_Xtest_acc = 0.9366941121640516
            self.gap = 1
        elif self.dataset_name == 'default_credit':
            self.sen_att_name = ['sex']
            self.original_Xtest_DR = 0.01625891402363777
            self.original_Xtest_acc = None
            self.gap = 1
        else :
            raise ValueError('The dataset name is not supported')

    def run_experiment(self):
        """
        Run the experiment.

        ```
        
        2. 将X_train_minority与X_train_replace_majority进行匹配
        3. 使用fairshap,把X_train_minority作为baseline dataset，找到X_train_replace_majority中需要替换的数据，假设总共需要替换n个数据点
        4. (1,n,20)根据这些,分别计算替换(1,n)中不同个数的结果,把需要替换的数据替换到X_train_replace_majority中,得到X_train_replace_majority_new
        5. 把X_train_replace_majority_new和X_train_rest_majority,还有X_train_minority合并,得到新的X_train_new，然后重新训练，得到新的模型model_new，计算新的DR值

        """
        # 1. 从majority parity(此处男性)中随机选择30%, 50%, 70%的比例，作为将被替换的数据集X_train_replace_majority,剩余部分为X_train_rest_majority
        # proportion = 1
        # X_train_replace_majority = self.X_train_majority.sample(frac=proportion, random_state=20)
        # X_train_rest_majority = self.X_train_majority.drop(X_train_replace_majority.index)
        # y_train_replace_majority = self.y_train_majority.loc[X_train_replace_majority.index]
        # y_train_rest_majority = self.y_train_majority.drop(X_train_replace_majority.index)
        
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
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

        if self.change_group == 'minority':
            print('3. 将X_train_minority与X_train_majority进行匹配')
            matching = NearestNeighborDataMatcher(X_labeled=self.X_train_minority, X_unlabeled=self.X_train_majority).match(n_neighbors=1)
            print('4. 使用fairshap, 从X_train_majority中找到合适的值替换X_train_minority中的数据')
            fairness_shapley_value = fairness_explainer_original.shap_values(
                                        X = self.X_train_minority.values,
                                        X_baseline = self.X_train_majority.values,
                                        matching=matching,
                                        sample_size=1000,
                                        shap_sample_size="auto",
                                    )
            X_change = self.X_train_minority.copy()
            X_base = self.X_train_majority

        elif self.change_group == 'majority':
            print('3. 将X_train_majority与X_train_minority进行匹配')
            matching = NearestNeighborDataMatcher(X_labeled=self.X_train_majority, X_unlabeled=self.X_train_minority).match(n_neighbors=1)
            print('4. 使用fairshap, 从X_train_minority中找到合适的值替换X_train_majority中的数据')
            fairness_shapley_value = fairness_explainer_original.shap_values(
                                        X = self.X_train_majority.values,
                                        X_baseline = self.X_train_minority.values,
                                        matching=matching,
                                        sample_size=1000,
                                        shap_sample_size="auto",
                                    )  
            X_change = self.X_train_majority.copy()
            X_base = self.X_train_minority

        print('5. 计算出varphi和q')
        # 筛选出shapley value大于0.1的值，其他值设为0，然后归一化
        varphi = fix_negative_probabilities_select_larger(fairness_shapley_value)
        non_zero_count =np.count_nonzero(varphi)
        viz_varphi(varphi=fairness_shapley_value)
        q = DataComposer(
                        x_counterfactual=X_base.values, 
                        joint_prob=matching, 
                        method="max").calculate_q()    # q是与X_train_replace_majority匹配的X_train_minority中的数据
        print('6. 用新数据重新训练，并且评估')
        values_range = np.arange(1, non_zero_count, self.gap)
        after_values_on_test_set = []
        after_values_on_train_set = []
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
                X_Train_New = pd.concat([X_base, X_change], axis=0)
                y_train_new = pd.concat([self.y_train_majority, self.y_train_minority], axis=0)
            elif self.change_group == 'majority':
                X_Train_New = pd.concat([X_change, X_base], axis=0)
                y_train_new = pd.concat([self.y_train_majority, self.y_train_minority], axis=0)

            # Step 6: Train and evaluate model
            x = X_Train_New
            y = y_train_new
    
            model_new = XGBClassifier()
            model_new.fit(x, y)
            after = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, model_new)
            after_values_on_test_set.append(after)
            after_value = fairness_value_function(sen_att, priv_val, unpriv_dict, X_Train_New.values, model_new)
            after_values_on_train_set.append(after_value)

            
            if after < self.original_Xtest_DR:
                y_new_pred = model_new.predict(self.X_test)
                accuracy_new = accuracy_score(self.y_test, y_new_pred)
                fairness_accuracy_pairs.append((after, accuracy_new, action_number))  # Store both values as a tuple

        #修改不同位置后训练的new_model在相应修改后的training set上的DR值
        viz1(values_range, after_values_on_train_set, self.original_Xtrain_DR, title='new_model\'s DR on training set')
        #修改不同位置后训练的new_model在test set上的DR值
        viz1(values_range, after_values_on_test_set, self.original_Xtest_DR, title='new_model\'s DR on test set')
        viz2(fairness_accuracy_pairs, self.original_Xtest_acc, title='Accuracy vs. DR')


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
    total_prob = varphi.sum()
    if total_prob == 0:
        raise ValueError("All probabilities are zero after filtering values <= 0.1.")
    varphi = varphi / total_prob

    # 计算出非0的个数
    non_zero_count = np.count_nonzero(varphi)
    print(f"shapely value中大于0.1的值的个数有: {non_zero_count}")
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