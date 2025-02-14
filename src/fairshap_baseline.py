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
from fairness_measures import marginalised_np_mat, grp1_DP, grp2_EO, grp3_PQP
EPSILON = 1e-20

class Baseline:
    """ 
    The core part of how fairshap(based on discriminative_risk) works.
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
            original_Xtest_DP,
            original_Xtest_EO,
            original_Xtest_PQP,

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
        self.original_Xtest_DP = original_Xtest_DP
        self.original_Xtest_EO = original_Xtest_EO
        self.original_Xtest_PQP = original_Xtest_PQP

        if self.dataset_name == 'german_credit':
            self.sen_att_name = ['sex']
            self.gap = 1
            self.change_num = 200
        elif self.dataset_name == 'adult':
            self.sen_att_name = ['sex']
            self.gap = 1
            self.change_num = 200
        elif self.dataset_name == 'compas':
            self.sen_att_name = ['sex']
            self.gap = 1
            self.change_num = 200
        elif self.dataset_name == 'compas4race':
            self.sen_att_name = ['race']
            self.gap = 1
            self.change_num = 200
        elif self.dataset_name == 'census_income':
            self.sen_att_name = ['sex']
            self.gap = 1
            self.change_num = 200

        # elif self.dataset_name == 'default_credit':
        #     self.sen_att_name = ['sex']
        #     self.gap = 1
        else :
            raise ValueError('The dataset name is not supported')

    def baseline1(self):
        """
        Run the baseline1: 对总共的X_train_majority和X_train_minority中的所有features随机修改500个点   

        ```
        1. 把X_train_majority和X_train_minority分别分成X_train_majority_label1, X_train_majority_label0, X_train_minority_label1, X_train_minority_label0
        2. X_train_majority_label1和X_train_minority_label1进行匹配
           X_train_majority_label0和X_train_minority_label0进行匹配
           X_train_minority_label1和X_train_majority_label1进行匹配
           X_train_minority_label0和X_train_majority_label0进行匹配
        3. 前4个使用np.vstack组成X_change， 计算出后与之匹配的q，把4个q使用np.vstack组成X_base
        4. 在X_change的所有值中挑500个位置，把这些位置的值替换成X_base里的值
        5. 使用新的数据集训练new_model，并且进行评估
        """
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

        # 1. 把X_train_majority和X_train_minority分别分成X_train_majority_label1, X_train_majority_label0, X_train_minority_label1, X_train_minority_label0
        print('1. 把X_train_majority和X_train_minority分别分成X_train_majority_label1, X_train_majority_label0, X_train_minority_label1, X_train_minority_label0')
        y_train_majority_label1 = self.y_train_majority[self.y_train_majority == 1]
        y_train_majority_label0 = self.y_train_majority[self.y_train_majority == 0]
        y_train_minority_label1 = self.y_train_minority[self.y_train_minority == 1]
        y_train_minority_label0 = self.y_train_minority[self.y_train_minority == 0]

        X_train_majority_label0 = self.X_train_majority.loc[y_train_majority_label0.index]
        X_train_majority_label1 = self.X_train_majority.loc[y_train_majority_label1.index]
        X_train_minority_label0 = self.X_train_minority.loc[y_train_minority_label0.index]
        X_train_minority_label1 = self.X_train_minority.loc[y_train_minority_label1.index]

        # 2. matching
        print('2. matching')
        print('2(a). 将X_train_majority_label1与X_train_minority_label1进行匹配')
        matching_majority_label1 = NearestNeighborDataMatcher(X_labeled=X_train_majority_label1, X_unlabeled=X_train_minority_label1).match(n_neighbors=1)
        print('2(b). 将X_train_majority_label0与X_train_minority_label0进行匹配')
        matching_majority_label0 = NearestNeighborDataMatcher(X_labeled=X_train_majority_label0, X_unlabeled=X_train_minority_label0).match(n_neighbors=1)
        print('2(c). 将X_train_minority_label1与X_train_majority_label1进行匹配')
        matching_minority_label1 = NearestNeighborDataMatcher(X_labeled=X_train_minority_label1, X_unlabeled=X_train_majority_label1).match(n_neighbors=1)
        print('2(d). 将X_train_minority_label0与X_train_majority_label0进行匹配')
        matching_minority_label0 = NearestNeighborDataMatcher(X_labeled=X_train_minority_label0, X_unlabeled=X_train_majority_label0).match(n_neighbors=1)


        # 3. 前4个使用np.vstack组成X_change， 计算出后与之匹配的q，把4个q使用np.vstack组成X_base
        print('3. 前4个使用np.vstack组成X_change， 计算出后与之匹配的q，把4个q使用np.vstack组成X_base')
        X_change = pd.concat([X_train_majority_label1, X_train_majority_label0, X_train_minority_label1, X_train_minority_label0], axis=0)

        print('3(a). 挑出X_train_minority_label1中与\'X_train_majority_label1中每个instance\'最近的instance,')
        q_majority_label1 = DataComposer(
                        x_counterfactual=X_train_minority_label1.values,
                        joint_prob=matching_majority_label1,
                        method="max").calculate_q()

        print('3(b). 挑出X_train_minority_label0中与\'X_train_majority_label0中每个instance\'最近的instance,')
        q_majority_label0 = DataComposer(
                        x_counterfactual=X_train_minority_label0.values,
                        joint_prob=matching_majority_label0,
                        method="max").calculate_q()
        print('3(c). 挑出X_train_majority_label1中与\'X_train_minority_label1中每个instance\'最近的instance,')
        q_minority_label1 = DataComposer(
                        x_counterfactual=X_train_majority_label1.values, 
                        joint_prob=matching_minority_label1, 
                        method="max").calculate_q() 
        print('3(d). 挑出X_train_majority_label0中与\'X_train_minority_label0中每个instance\'最近的instance,')
        q_minority_label0 = DataComposer(
                        x_counterfactual=X_train_majority_label0.values, 
                        joint_prob=matching_minority_label0, 
                        method="max").calculate_q()
        X_base = np.vstack((q_majority_label1, q_majority_label0, q_minority_label1, q_minority_label0))



        # 4. 在X_change的所有值中挑500个位置，把这些位置的值替换成X_base里的值
        print('4. 在X_change的所有值中挑500个位置，把这些位置的值替换成X_base里的值')

        shape = X_change.shape  # 获取 X_change 的形状
        random_varphi = np.zeros(shape)  # 创建与 X_change 形状相同的零矩阵
        # 随机选择 500 个唯一的位置
        indices = np.random.choice(np.prod(shape), self.change_num, replace=False)
        # 在这些位置填充 [0.5, 1.0] 之间的随机值
        random_varphi.flat[indices] = np.random.uniform(0.5, 1.0, self.change_num)

        # 初始化 fairness_measure中需要使用的变量
        sen_att_name = self.sen_att_name
        sen_att = [self.X_test.columns.get_loc(name) for name in sen_att_name]
        priv_val = [1]
        unpriv_dict = [list(set(self.X_test.values[:, sa])) for sa in sen_att]
        for sa_list, pv in zip(unpriv_dict, priv_val):
            sa_list.remove(pv)

        values_range = np.arange(1, self.change_num , self.gap)
        after_values_on_test_set = []
        after_values_on_train_set = []
        after_DP_on_test_set = []
        after_EO_on_test_set = []
        after_PQP_on_test_set = []
        fairness_accuracy_pairs = []
        changed_value_info = []   # 存储修改的值的信息(column, before_value, after_value)
        for action_number in values_range:
            # Step 1: 将 varphi 的值和位置展开为一维
            flat_varphi = [(value, row, col) for row, row_vals in enumerate(random_varphi)
                        for col, value in enumerate(row_vals)]
            # Step 2: 按值降序排序
            flat_varphi_sorted = sorted(flat_varphi, key=lambda x: x[0], reverse=True)
            # Step 3: 挑出前 action_number 个数的位置
            top_positions = flat_varphi_sorted[:action_number]
            
            # Step 4: 替换 X 中前三列的值为 S 中对应位置的值
            for value, row_idx, col_idx in top_positions:
                X_change.iloc[row_idx, col_idx] = X_base[row_idx, col_idx]

            x = X_change
            y = pd.concat([y_train_minority_label0, y_train_minority_label1, y_train_majority_label0, y_train_majority_label1], axis=0)

            # Step 6: Train and evaluate model            
            model_new = XGBClassifier()
            model_new.fit(x, y)
            after = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, model_new)
            after_values_on_test_set.append(after)
            
            # step7: 评估新模型在其他fairnes measures上的表现
            if self.dataset_name != 'compas4race':
                priv_idx = self.X_test['sex'].to_numpy().astype(bool)
            else:
                priv_idx = self.X_test['race'].to_numpy().astype(bool)
            y_hat = model_new.predict(self.X_test)
            y_test = self.y_test
            g1_Cm, g0_Cm = marginalised_np_mat(y_test, y_hat, 1, priv_idx)
            fair_grp1 = grp1_DP(g1_Cm, g0_Cm)[0]
            fair_grp2 = grp2_EO(g1_Cm, g0_Cm)[0]
            fair_grp3 = grp3_PQP(g1_Cm, g0_Cm)[0]
            after_DP_on_test_set.append(fair_grp1)
            after_EO_on_test_set.append(fair_grp2)
            after_PQP_on_test_set.append(fair_grp3)

            if after < self.original_Xtest_DR:
                y_new_pred = model_new.predict(self.X_test)
                accuracy_new = accuracy_score(self.y_test, y_new_pred)
                fairness_accuracy_pairs.append((after, accuracy_new, action_number))  # Store both values as a tuple

        viz1(values_range, after_values_on_test_set, self.original_Xtest_DR, title='new_model\'s DR on test set',color='purple')
        viz1(values_range, after_DP_on_test_set, self.original_Xtest_DP, title='new_model\'s DP on test set', color='g')
        viz1(values_range, after_EO_on_test_set, self.original_Xtest_EO, title='new_model\'s EO on test set', color='b')
        viz1(values_range, after_PQP_on_test_set, self.original_Xtest_PQP, title='new_model\'s PQP on test set',color='y')
        pass

    def baseline2(self):
        """
        Run the baseline2: 对总共的X_train_majority和X_train_minority中的sensitive attribute随机修改500个点   

        ```
        1. 把X_train_majority和X_train_minority分别分成X_train_majority_label1, X_train_majority_label0, X_train_minority_label1, X_train_minority_label0
        2. X_train_majority_label1和X_train_minority_label1进行匹配
           X_train_majority_label0和X_train_minority_label0进行匹配
           X_train_minority_label1和X_train_majority_label1进行匹配
           X_train_minority_label0和X_train_majority_label0进行匹配
        3. 前4个使用np.vstack组成X_change， 计算出后与之匹配的q，把4个q使用np.vstack组成X_base
        4. 在X_change的sensitive attribute列中挑500个位置，把这些位置的值替换成X_base里的值
        5. 使用新的数据集训练new_model，并且进行评估
        """
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

        # 1. 把X_train_majority和X_train_minority分别分成X_train_majority_label1, X_train_majority_label0, X_train_minority_label1, X_train_minority_label0
        print('1. 把X_train_majority和X_train_minority分别分成X_train_majority_label1, X_train_majority_label0, X_train_minority_label1, X_train_minority_label0')
        y_train_majority_label1 = self.y_train_majority[self.y_train_majority == 1]
        y_train_majority_label0 = self.y_train_majority[self.y_train_majority == 0]
        y_train_minority_label1 = self.y_train_minority[self.y_train_minority == 1]
        y_train_minority_label0 = self.y_train_minority[self.y_train_minority == 0]

        X_train_majority_label0 = self.X_train_majority.loc[y_train_majority_label0.index]
        X_train_majority_label1 = self.X_train_majority.loc[y_train_majority_label1.index]
        X_train_minority_label0 = self.X_train_minority.loc[y_train_minority_label0.index]
        X_train_minority_label1 = self.X_train_minority.loc[y_train_minority_label1.index]

        # 2. matching
        print('2. matching')
        print('2(a). 将X_train_majority_label1与X_train_minority_label1进行匹配')
        matching_majority_label1 = NearestNeighborDataMatcher(X_labeled=X_train_majority_label1, X_unlabeled=X_train_minority_label1).match(n_neighbors=1)
        print('2(b). 将X_train_majority_label0与X_train_minority_label0进行匹配')
        matching_majority_label0 = NearestNeighborDataMatcher(X_labeled=X_train_majority_label0, X_unlabeled=X_train_minority_label0).match(n_neighbors=1)
        print('2(c). 将X_train_minority_label1与X_train_majority_label1进行匹配')
        matching_minority_label1 = NearestNeighborDataMatcher(X_labeled=X_train_minority_label1, X_unlabeled=X_train_majority_label1).match(n_neighbors=1)
        print('2(d). 将X_train_minority_label0与X_train_majority_label0进行匹配')
        matching_minority_label0 = NearestNeighborDataMatcher(X_labeled=X_train_minority_label0, X_unlabeled=X_train_majority_label0).match(n_neighbors=1)


        # 3. 前4个使用np.vstack组成X_change， 计算出后与之匹配的q，把4个q使用np.vstack组成X_base
        print('3. 前4个使用np.vstack组成X_change， 计算出后与之匹配的q，把4个q使用np.vstack组成X_base')
        X_change = pd.concat([X_train_majority_label1, X_train_majority_label0, X_train_minority_label1, X_train_minority_label0], axis=0)

        print('3(a). 挑出X_train_minority_label1中与\'X_train_majority_label1中每个instance\'最近的instance,')
        q_majority_label1 = DataComposer(
                        x_counterfactual=X_train_minority_label1.values,
                        joint_prob=matching_majority_label1,
                        method="max").calculate_q()

        print('3(b). 挑出X_train_minority_label0中与\'X_train_majority_label0中每个instance\'最近的instance,')
        q_majority_label0 = DataComposer(
                        x_counterfactual=X_train_minority_label0.values,
                        joint_prob=matching_majority_label0,
                        method="max").calculate_q()
        print('3(c). 挑出X_train_majority_label1中与\'X_train_minority_label1中每个instance\'最近的instance,')
        q_minority_label1 = DataComposer(
                        x_counterfactual=X_train_majority_label1.values, 
                        joint_prob=matching_minority_label1, 
                        method="max").calculate_q() 
        print('3(d). 挑出X_train_majority_label0中与\'X_train_minority_label0中每个instance\'最近的instance,')
        q_minority_label0 = DataComposer(
                        x_counterfactual=X_train_majority_label0.values, 
                        joint_prob=matching_minority_label0, 
                        method="max").calculate_q()
        X_base = np.vstack((q_majority_label1, q_majority_label0, q_minority_label1, q_minority_label0))



        # 4. 在X_change的sensitive attribute列中挑500个位置，把这些位置的值替换成X_base里的值
        print('4. 在X_change的sensitive attribute列中挑500个位置，把这些位置的值替换成X_base里的值')


        # 初始化 fairness_measure中需要使用的变量
        sen_att_name = self.sen_att_name
        sen_att = [self.X_test.columns.get_loc(name) for name in sen_att_name]
        priv_val = [1]
        unpriv_dict = [list(set(self.X_test.values[:, sa])) for sa in sen_att]
        for sa_list, pv in zip(unpriv_dict, priv_val):
            sa_list.remove(pv)

        shape = X_change.shape  
        random_varphi = np.zeros(shape)  # 创建与 X_change 形状相同的零矩阵

        # 获取所有可能的行列索引（仅限于敏感属性列）
        row_indices = np.arange(shape[0])  # 所有行索引
        col_indices = np.array(sen_att)  # 仅限于敏感属性列

        # 生成随机选择的行列索引
        chosen_rows = np.random.choice(row_indices, self.change_num, replace=True)  # 随机选 self.change_num 行
        chosen_cols = np.random.choice(col_indices, self.change_num, replace=True)  # 随机选 self.change_num 列（只从敏感属性列选）

        # 生成 [0.5, 1.0] 之间的随机值
        random_values = np.random.uniform(0.5, 1.0, self.change_num)

        # 填充到 random_varphi 对应的位置
        random_varphi[chosen_rows, chosen_cols] = random_values


        values_range = np.arange(1, self.change_num , self.gap)
        after_values_on_test_set = []
        after_values_on_train_set = []
        after_DP_on_test_set = []
        after_EO_on_test_set = []
        after_PQP_on_test_set = []
        fairness_accuracy_pairs = []
        changed_value_info = []   # 存储修改的值的信息(column, before_value, after_value)
        for action_number in values_range:
            # Step 1: 将 varphi 的值和位置展开为一维
            flat_varphi = [(value, row, col) for row, row_vals in enumerate(random_varphi)
                        for col, value in enumerate(row_vals)]
            # Step 2: 按值降序排序
            flat_varphi_sorted = sorted(flat_varphi, key=lambda x: x[0], reverse=True)
            # Step 3: 挑出前 action_number 个数的位置
            top_positions = flat_varphi_sorted[:action_number]
            
            # Step 4: 替换 X 中前三列的值为 S 中对应位置的值
            for value, row_idx, col_idx in top_positions:
                X_change.iloc[row_idx, col_idx] = X_base[row_idx, col_idx]

            x = X_change
            y = pd.concat([y_train_minority_label0, y_train_minority_label1, y_train_majority_label0, y_train_majority_label1], axis=0)

            # Step 6: Train and evaluate model            
            model_new = XGBClassifier()
            model_new.fit(x, y)
            after = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, model_new)
            after_values_on_test_set.append(after)
            
            # step7: 评估新模型在其他fairnes measures上的表现
            if self.dataset_name != 'compas4race':
                priv_idx = self.X_test['sex'].to_numpy().astype(bool)
            else:
                priv_idx = self.X_test['race'].to_numpy().astype(bool)
            y_hat = model_new.predict(self.X_test)
            y_test = self.y_test
            g1_Cm, g0_Cm = marginalised_np_mat(y_test, y_hat, 1, priv_idx)
            fair_grp1 = grp1_DP(g1_Cm, g0_Cm)[0]
            fair_grp2 = grp2_EO(g1_Cm, g0_Cm)[0]
            fair_grp3 = grp3_PQP(g1_Cm, g0_Cm)[0]
            after_DP_on_test_set.append(fair_grp1)
            after_EO_on_test_set.append(fair_grp2)
            after_PQP_on_test_set.append(fair_grp3)

            if after < self.original_Xtest_DR:
                y_new_pred = model_new.predict(self.X_test)
                accuracy_new = accuracy_score(self.y_test, y_new_pred)
                fairness_accuracy_pairs.append((after, accuracy_new, action_number))  # Store both values as a tuple

        viz1(values_range, after_values_on_test_set, self.original_Xtest_DR, title='new_model\'s DR on test set',color='purple')
        viz1(values_range, after_DP_on_test_set, self.original_Xtest_DP, title='new_model\'s DP on test set', color='g')
        viz1(values_range, after_EO_on_test_set, self.original_Xtest_EO, title='new_model\'s EO on test set', color='b')
        viz1(values_range, after_PQP_on_test_set, self.original_Xtest_PQP, title='new_model\'s PQP on test set',color='y')
        pass




    

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
    # varphi = np.abs(varphi)
    varphi = np.where(varphi > 0.1, varphi, 0)
    # varphi = np.where(varphi < - 0.1, varphi, 0)
    # varphi = np.abs(varphi)

    total_prob = varphi.sum()
    if total_prob == 0:
        # print("All probabilities are zero after filtering values <= 0.1.")
        # return varphi
        raise ValueError("All probabilities are zero after filtering values <= 0.1.")
    varphi = varphi / total_prob
    return varphi


def viz_varphi(varphi):
    '''
    绘制varphi的分布图
    '''
    varphi_flat = varphi.flatten()
    # # 去掉varphi中小于0.001的值
    varphi_flat = varphi_flat[varphi_flat >= -0.05]

    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(varphi_flat, bins=100, edgecolor='black', alpha=0.7)
    # 在柱状图上显示数值
    for count, bin_edge in zip(counts, bins):
        plt.text(bin_edge, count, str(int(count)), ha='center', va='bottom')

    plt.title('Distribution of varphi values')
    plt.xlabel('Value')
    plt.xlim(-0.05,0.2)
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()



def viz1(values_range, after_values, original_DR, title, color='b'):
    '''
    绘制修改后的DR值与修改的action number的散点图
    '''
    plt.figure(figsize=(10, 6))
    plt.scatter(values_range, after_values, label='New model', marker='x', color=color)
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