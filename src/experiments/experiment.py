import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from typing import Tuple, List, Dict
from xgboost import XGBClassifier
import pdb
import logging
import os

from sklearn.metrics import accuracy_score
from src.matching.ot_matcher import OptimalTransportPolicy
from src.matching.nn_matcher import NearestNeighborDataMatcher
from src.attribution import FairnessExplainer
from src.composition.data_composer import DataComposer
from src.attribution.oracle_metric import perturb_numpy_ver
from fairness_measures import marginalised_np_mat, grp1_DP, grp2_EO, grp3_PQP


EPSILON = 1e-20

class Experiment:
    '''
    This class is used to run the core experiment (Use FairSHAP to enhance fairness of a model), 
    and return the results, including (4 fairness measures vs. modification num) and (fairness measures vs. accuracy)

    Arg:
    - X_train: pd.Dataframe, training data
    - y_train: pd.Series, training labels
    - X_test: pd.Dataframe, testing data
    - y_test: pd.Series, testing labels
    - dataset_name: str, the name of the dataset ('german_credit', 'adult', 'compas', 'census_kdd')
    - ith_fold: int, the ith fold of the cross-validation (combined with k-fold cross-validation)

    Return:
    - result_fairness_measures: Dict, the results of the experiment (we will train a new model after each modification, and then evaluate the fairness measures on each new model) )
    - result_accuracy: Dict, the results of the experiment (fairness measures vs. accuracy)
    
    - save_csv_file

    '''
    def __init__(self,
                 model,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 X_test: pd.DataFrame,
                 y_test: pd.Series,
                 dataset_name: str,
                 fairshap_base: str = 'DR',   # 'DR', 'DR+precision', 'DR+recall', 'DR+sufficiency'
                 matching_method: str = 'NN',  # 'NN', 'OT'
                 ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.dataset_name = dataset_name
    
        self.fairshap_base = fairshap_base  # combine FairSHAP with DR
        self.matching_method = matching_method  # 'NN', 'OT'

        if self.dataset_name == 'german_credit':
            self.sensitive_attri = 'sex'
            self.gap = 1
        elif self.dataset_name == 'adult':
            self.sensitive_attri = 'sex'
            self.gap = 1
        elif self.dataset_name == 'compas':
            self.sensitive_attri = 'sex'
            self.gap = 1
        elif self.dataset_name == 'compas4race':
            self.sensitive_attri = 'race'
            self.gap = 1
        elif self.dataset_name == 'census_income':
            self.sensitive_attri = 'sex'
            self.gap = 1
        elif self.dataset_name == 'default_credit':
            self.sensitive_attri = 'sex'
            self.gap = 1
        else :
            raise ValueError('The dataset name is not supported')      


    def run(self, ith_fold: int, threshold: float = 0.1):
        self.ith_fold = ith_fold
        print(f"1. Split the {self.dataset_name} dataset into majority group and minority group according to the number of sensitive attribute, besides split by label 0 and label 1")
        X_train_majority_label0, y_train_majority_label0, X_train_majority_label1, y_train_majority_label1, X_train_minority_label0, y_train_minority_label0, X_train_minority_label1, y_train_minority_label1 = self._split_into_majority_minority_label0_label1()

        print(f'X_train_majority_label0 shape: {X_train_majority_label0.shape}')
        print(f'X_train_majority_label1 shape: {X_train_majority_label1.shape}')
        print(f'X_train_minority_label0 shape: {X_train_minority_label0.shape}')
        print(f'X_train_minority_label1 shape: {X_train_minority_label1.shape}')
        print('2. 初始化FairnessExplainer')
        sen_att_name = [self.sensitive_attri]
        sen_att = [self.X_test.columns.get_loc(name) for name in sen_att_name]
        priv_val = [1]
        unpriv_dict = [list(set(self.X_test.values[:, sa])) for sa in sen_att]
        for sa_list, pv in zip(unpriv_dict, priv_val):
            sa_list.remove(pv)
        fairness_explainer_original = FairnessExplainer(
                model=self.model, 
                sen_att=sen_att, 
                priv_val=priv_val, 
                unpriv_dict=unpriv_dict,
                fairshap_base=self.fairshap_base
                )
        
        print('--------接下来先对minority group进行修改--------')
        print('3(a). 将X_train_minority_label0与X_train_majority_label0进行匹配')
        print('3(b). 将X_train_minority_label1与X_train_majority_label1进行匹配')
        if self.matching_method == 'NN':
            matching_minority_label0 = NearestNeighborDataMatcher(X_labeled=X_train_minority_label0, X_unlabeled=X_train_majority_label0).match(n_neighbors=1)
            matching_minority_label1 = NearestNeighborDataMatcher(X_labeled=X_train_minority_label1, X_unlabeled=X_train_majority_label1).match(n_neighbors=1)
        elif self.matching_method == 'OT':
            matching_minority_label0 = OptimalTransportPolicy(X_labeled=X_train_minority_label0.values, X_unlabeled=X_train_majority_label0.values).match()
            matching_minority_label1 = OptimalTransportPolicy(X_labeled=X_train_minority_label1.values, X_unlabeled=X_train_majority_label1.values).match()
        else:
            raise ValueError('The matching method is not supported')
        print('4(a). 使用FairSHAP, 从 X_train_majority_label0中找到合适的值替换X_train_minority_label0中的数据')
        fairness_shapley_minority_value_label0 = fairness_explainer_original.shap_values(
                                    X = X_train_minority_label0.values,
                                    Y = y_train_minority_label0.values,
                                    X_baseline = X_train_majority_label0.values,
                                    matching=matching_minority_label0,
                                    sample_size=2000,
                                    shap_sample_size="auto",
                                )
        X_change_minority_label0 = X_train_minority_label0.copy()
        X_base_minority_label0 = X_train_majority_label0
        print('4(b). 使用FairSHAP, 从 X_train_majority_label1中找到合适的值替换X_train_minority_label1中的数据')
        fairness_shapley_minority_value_label1 = fairness_explainer_original.shap_values(
                                    X = X_train_minority_label1.values,
                                    Y = y_train_minority_label1.values,
                                    X_baseline = X_train_majority_label1.values,
                                    matching=matching_minority_label1,
                                    sample_size=2000,
                                    shap_sample_size="auto",
                                )
        X_change_minority_label1 = X_train_minority_label1.copy()
        X_base_minority_label1 = X_train_majority_label1
        print('5. 计算出varphi和q')
        fairness_shapley_minority_value = np.vstack((fairness_shapley_minority_value_label0, fairness_shapley_minority_value_label1))
        non_zero_count_minority = np.sum(fairness_shapley_minority_value > threshold)
        print(f"在X_train_minority中shapely value中大于{threshold}的值的个数有: {non_zero_count_minority}")
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
        print('3(b). 将X_train_majority_label1与X_train_minority_label1进行匹配')
        if self.matching_method == 'NN':
            matching_majority_label0 = NearestNeighborDataMatcher(X_labeled=X_train_majority_label0, X_unlabeled=X_train_minority_label0).match(n_neighbors=1)       
            matching_majority_label1 = NearestNeighborDataMatcher(X_labeled=X_train_majority_label1, X_unlabeled=X_train_minority_label1).match(n_neighbors=1)
        elif self.matching_method == 'OT':
            matching_majority_label0 = OptimalTransportPolicy(X_labeled=X_train_majority_label0.values, X_unlabeled=X_train_minority_label0.values).match()
            matching_majority_label1 = OptimalTransportPolicy(X_labeled=X_train_majority_label1.values, X_unlabeled=X_train_minority_label1.values).match()
        else:
            raise ValueError('The matching method is not supported')

        print('4(a). 使用fairshap, 从 X_train_minority_label0中找到合适的值替换X_train_majority_label0中的数据')
        fairness_shapley_majority_value_label0 = fairness_explainer_original.shap_values(
                                    X = X_train_majority_label0.values,
                                    Y = y_train_majority_label0.values,
                                    X_baseline = X_train_minority_label0.values,
                                    matching=matching_majority_label0,
                                    sample_size=2000,
                                    shap_sample_size="auto",
                                )
        X_change_majority_label0 = X_train_majority_label0.copy()
        X_base_majority_label0 = X_train_minority_label0
        print('4(b). 使用fairshap, 从 X_train_minority_label1中找到合适的值替换X_train_majority_label1中的数据')
        fairness_shapley_majority_value_label1 = fairness_explainer_original.shap_values(
                                    X = X_train_majority_label1.values,
                                    Y = y_train_majority_label1.values,
                                    X_baseline = X_train_minority_label1.values,
                                    matching=matching_majority_label1,
                                    sample_size=2000,
                                    shap_sample_size="auto",
                                )  
        X_change_majority_label1 = X_train_majority_label1.copy()
        X_base_majority_label1 = X_train_minority_label1
            
        print('5. 计算出varphi和q')
        # 筛选出shapley value大于0.1的值，其他值设为0，然后归一化
        fairness_shapley_majority_value = np.vstack((fairness_shapley_majority_value_label0, fairness_shapley_majority_value_label1))
        non_zero_count_majority =np.sum(fairness_shapley_majority_value > threshold)

        print(f"在X_train_majority中shapely value中大于{threshold}的值的个数有: {non_zero_count_majority}")
        q_majority_label0 = DataComposer(
                        x_counterfactual=X_base_majority_label0.values, 
                        joint_prob=matching_majority_label0, 
                        method="max").calculate_q() 
        q_majority_label1 = DataComposer(
                        x_counterfactual=X_base_majority_label1.values, 
                        joint_prob=matching_majority_label1, 
                        method="max").calculate_q()
        q_majority = np.vstack((q_majority_label0, q_majority_label1))
        fairness_shapley_value = np.vstack((fairness_shapley_minority_value, fairness_shapley_majority_value))
        # varphi = fix_negative_probabilities_select_larger(fairness_shapley_value)

        if self.fairshap_base == 'DR':
            varphi = np.where(fairness_shapley_value > threshold, fairness_shapley_value, 0)
        elif self.fairshap_base == 'DP' or self.fairshap_base == 'EO' or self.fairshap_base == 'PQP':
           varphi = np.where(fairness_shapley_value > threshold, fairness_shapley_value, 0)
           varphi = np.abs(varphi)
           pass
        q = np.vstack((q_minority,q_majority))   # q_minority_label0 + q_minority_label1 + q_majority_label0 + q_majority_label1
        X_change = pd.concat([X_change_minority_label0, X_change_minority_label1, X_change_majority_label0, X_change_majority_label1], axis=0)
        non_zero_count = non_zero_count_majority + non_zero_count_minority

        print('6. 计算original model在X_test上的accuracy, DR, DP, EO, PP')
        y_pred = self.model.predict(self.X_test)
        original_accuracy = accuracy_score(self.y_test, y_pred)
        original_DR = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, self.model)

        priv_idx = self.X_test[self.sensitive_attri].to_numpy().astype(bool)
        g1_Cm, g0_Cm = marginalised_np_mat(y=self.y_test, y_hat=y_pred, pos_label=1, priv_idx=priv_idx)
        original_DP = grp1_DP(g1_Cm, g0_Cm)[0]
        original_EO = grp2_EO(g1_Cm, g0_Cm)[0]
        original_PQP = grp3_PQP(g1_Cm, g0_Cm)[0]

        # 计算指标
        # original_recall, original_precision, original_sufficiency = calculate_metrics(self.y_test, y_pred, pos=1)

        print(f'7. 开始整理minority部分的修改和majority部分的修改并且合并新数据,共修改{non_zero_count}个数据点, 使用new training set训练新模型')
        values_range = np.arange(1, non_zero_count, self.gap)
        # result_fairness_measures = {'DP':[], 'EO':[], 'PQP':[], 'DR':[]}
        # result_accuracy = {'DP':[], 'EO':[], 'PQP':[], 'DR':[]}

        accuracy_results = []
        DR_results = []
        DP_results = []
        EO_results = []
        PQP_results = []
        # recall_results = []
        # precision_results = []
        # sufficiency_results = []

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

            # Step 6: Train the new model            
            model_new = XGBClassifier()
            model_new.fit(x, y)

            # step7: 评估新模型在DR,DP,EO,PP上的表现
            new_DR = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, model_new)
            y_hat = model_new.predict(self.X_test)
            y_test = self.y_test
            new_accuracy = accuracy_score(self.y_test, y_hat)
            g1_Cm, g0_Cm = marginalised_np_mat(y_test, y_hat, 1, priv_idx)
            new_DP = grp1_DP(g1_Cm, g0_Cm)[0]
            new_EO = grp2_EO(g1_Cm, g0_Cm)[0]
            new_PQP = grp3_PQP(g1_Cm, g0_Cm)[0]
            # new_recall, new_precision, new_sufficiency = calculate_metrics(self.y_test, y_hat, pos=1)

            accuracy_results.append(new_accuracy)
            DR_results.append(new_DR)
            DP_results.append(new_DP)
            EO_results.append(new_EO)
            PQP_results.append(new_PQP)
            # recall_results.append(new_recall)
            # precision_results.append(new_precision)
            # sufficiency_results.append(new_sufficiency)

            # step8: 评估新模型在accuracy上的表现
            # if after < self.original_Xtest_DR:
            #     y_new_pred = model_new.predict(self.X_test)
            #     accuracy_new = accuracy_score(self.y_test, y_new_pred)
            #     fairness_accuracy_pairs.append((after, accuracy_new, action_number))  # Store both values as a tuple
        
        print('8. 保存结果到csv文件')
        df = pd.DataFrame({
            "action_number": values_range,  # 直接使用 values_range
            "new_accuracy": accuracy_results,
            "new_DR": DR_results,
            "new_DP": DP_results,
            "new_EO": EO_results,
            "new_PQP": PQP_results,
            # 'new_recall': recall_results,
            # 'new_precision': precision_results,
            # 'new_sufficiency': sufficiency_results,

        })
        # 在 DataFrame 的第一行添加 original 值
        # df.loc[-1] = ["original", original_DR, original_DP, original_EO, original_PQP, original_recall, original_precision, original_sufficiency]  # 插入到第一行
        df.loc[-1] = ["original", original_accuracy, original_DR, original_DP, original_EO, original_PQP]  # 插入到第一行
        df.index = df.index + 1  # 重新索引
        df = df.sort_index()  # 确保 original 行在最上面

        dataset_folder = os.path.join('saved_results', self.dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)

        # 生成 CSV 文件名
        csv_filename = f"fairSHAP-{self.fairshap_base}_{self.matching_method}_{self.ith_fold}-fold_results.csv"
        csv_filepath = os.path.join(dataset_folder, csv_filename)

        # 保存 CSV
        df.to_csv(csv_filepath, index=False)
        print(f"CSV 文件已保存：{csv_filepath}")




    def _split_into_majority_minority_label0_label1(self):
        '''
        This function is used to divide the dataset into majority group and minority group

        Arg:
        - X: pd.DataFrame, the input data
        - y: pd.Series, the input labels
        - sen_att_name: str, the sensitive attribute name

        Return:
        - X_majority: pd.DataFrame, the majority group data
        - y_majority: pd.Series, the majority group labels
        - X_minority: pd.DataFrame, the minority group data
        - y_minority: pd.Series, the minority group labels
        '''
        group_division = self.X_train[self.sensitive_attri].value_counts()
        '''把X_train分成majority和minority两个部分'''
        if group_division[0] > group_division[1]:  #
            majority = self.X_train[self.sensitive_attri] == 0
            X_train_majority = self.X_train[majority]
            y_train_majority = self.y_train[majority]
            minority = self.X_train[self.sensitive_attri] == 1
            X_train_minority = self.X_train[minority]
            y_train_minority = self.y_train[minority]

        else:
            majority = self.X_train[self.sensitive_attri] == 1
            X_train_majority = self.X_train[majority]
            y_train_majority = self.y_train[majority]
            minority = self.X_train[self.sensitive_attri] == 0
            X_train_minority = self.X_train[minority]
            y_train_minority = self.y_train[minority]

        y_train_majority_label1 = y_train_majority[y_train_majority == 1]
        y_train_majority_label0 = y_train_majority[y_train_majority == 0]
        y_train_minority_label1 = y_train_minority[y_train_minority == 1]
        y_train_minority_label0 = y_train_minority[y_train_minority == 0]

        X_train_majority_label0 = X_train_majority.loc[y_train_majority_label0.index]
        X_train_majority_label1 = X_train_majority.loc[y_train_majority_label1.index]
        X_train_minority_label0 = X_train_minority.loc[y_train_minority_label0.index]
        X_train_minority_label1 = X_train_minority.loc[y_train_minority_label1.index]

        return X_train_majority_label0, y_train_majority_label0, X_train_majority_label1, y_train_majority_label1, X_train_minority_label0, y_train_minority_label0, X_train_minority_label1, y_train_minority_label1 


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

def contingency_tab_bi(y, y_hat, pos=1):
  # For one single classifier
  tp = np.sum((y == pos) & (y_hat == pos))
  fn = np.sum((y == pos) & (y_hat != pos))
  fp = np.sum((y != pos) & (y_hat == pos))
  tn = np.sum((y != pos) & (y_hat != pos))
  return tp, fp, fn, tn


def marginalised_np_mat(y, y_hat, pos_label=1,
                        priv_idx=list()):
  if isinstance(y, list) or isinstance(y_hat, list):
    y, y_hat = np.array(y), np.array(y_hat)

  g1_y = y[priv_idx]
  g0_y = y[~priv_idx]
  g1_hx = y_hat[priv_idx]
  g0_hx = y_hat[~priv_idx]

  g1_Cm = contingency_tab_bi(g1_y, g1_hx, pos_label)
  g0_Cm = contingency_tab_bi(g0_y, g0_hx, pos_label)
  # g1_Cm: for the privileged group
  # g0_Cm: for marginalised group(s)
  return g1_Cm, g0_Cm


def zero_division(dividend, divisor):
  if divisor == 0 and dividend == 0:
    return 0.
  elif divisor == 0:
    return 10.  # return 1.
  return dividend / divisor


def grp1_DP(g1_Cm, g0_Cm):
  g1 = g1_Cm[0] + g1_Cm[1]
  g1 = zero_division(g1, sum(g1_Cm))
  g0 = g0_Cm[0] + g0_Cm[1]
  g0 = zero_division(g0, sum(g0_Cm))
  return abs(g0 - g1), float(g1), float(g0)


def grp2_EO(g1_Cm, g0_Cm):
  g1 = g1_Cm[0] + g1_Cm[2]
  g1 = zero_division(g1_Cm[0], g1)
  g0 = g0_Cm[0] + g0_Cm[2]
  g0 = zero_division(g0_Cm[0], g0)
  return abs(g0 - g1), float(g1), float(g0)


def grp3_PQP(g1_Cm, g0_Cm):
  g1 = g1_Cm[0] + g1_Cm[1]
  g1 = zero_division(g1_Cm[0], g1)
  g0 = g0_Cm[0] + g0_Cm[1]
  g0 = zero_division(g0_Cm[0], g0)
  return abs(g0 - g1), float(g1), float(g0)

# 定义计算指标的函数
def calculate_metrics(y_test, y_pred, pos=1):

    tp, fp, fn, tn = contingency_tab_bi(y_test, y_pred, pos)
    # 召回率
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    
    # 精确率
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    
    # 充分性（根据新定义）
    sufficiency = tn / (tn + fp) if (tn + fp) != 0 else 0
    
    return recall, precision, sufficiency

if __name__ == 'main':

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

    # into 70% training and 30% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 
    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')

    model = XGBClassifier()  # 可以替换为 RandomForestClassifier() 等其他模型
    model.fit(X_train,y_train)

    # 预测和评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # 实例化 Experiment 类  并运行
    experiment = Experiment(model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, dataset_name='german_credit')
    experiment.run(ith_fold=1)