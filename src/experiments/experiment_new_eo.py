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
from fairness_related.fairness_measures import marginalised_np_mat, grp1_DP, grp2_EO, grp3_PQP


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
                 fairshap_base: str = 'EO',   # 'EO'
                 ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.dataset_name = dataset_name
    
        self.fairshap_base = fairshap_base  # combine FairSHAP with DR

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


    def run(self, ith_fold: int):
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
        if self.fairshap_base == 'EO':
            print(f'This is FairSHAP combined with {self.fairshap_base}, so we only consider the label 1')
        else:
           print(f'This is FairSHAP combined with {self.fairshap_base}')


        print('3. 计算original model在X_test上的accuracy, DR, DP, EO, PP, recall, precision, sufficiency')
        y_pred = self.model.predict(self.X_test)
        original_accuracy = accuracy_score(self.y_test, y_pred)
        original_DR = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, self.model)
        priv_idx = self.X_test[self.sensitive_attri].to_numpy().astype(bool)
        g1_Cm, g0_Cm = marginalised_np_mat(y=self.y_test, y_hat=y_pred, pos_label=1, priv_idx=priv_idx)
        original_DP = grp1_DP(g1_Cm, g0_Cm)[0]
        original_EO, eo_g1, eo_g0 = grp2_EO(g1_Cm, g0_Cm)
        original_PQP = grp3_PQP(g1_Cm, g0_Cm)[0]
        original_recall, original_precision, original_sufficiency = calculate_metrics(self.y_test, y_pred, pos=1)

        print('4. 比较eo_g1和eo_g0,然后决定优化的方向')
        print(f'---EO_g1: {eo_g1}, EO_g2: {eo_g0}')
        if eo_g1 > eo_g0:
            print('---EO_g1 > EO_g0, 优化方向为增加EO_g0的TPR: 即使得所有label=1的所有g0的数据点, 尽可能都预测为1') #TPR= TP/(TP+FN)
            print('---因为现在minority group = g0, majority group = g1, 所以我们只对minority group进行修改')
            print('5. 将X_train_minority_label1与X_train_majority_label1进行匹配')
            matching_minority_label1 = NearestNeighborDataMatcher(X_labeled=X_train_minority_label1, X_unlabeled=X_train_majority_label1).match(n_neighbors=1)
            print('6. 使用FairSHAP, 替换X_train_minority_label1中的数据, 从 X_train_majority_label1中寻找到合适的值')
            fairness_shapley_minority_value_label1 = fairness_explainer_original.shap_values(
                                        X = X_train_minority_label1.values,
                                        Y = y_train_minority_label1.values,
                                        X_baseline = X_train_majority_label1.values,
                                        matching=matching_minority_label1,
                                        sample_size=2000,
                                        shap_sample_size="auto",
                                    )
            X_change_label1 = X_train_minority_label1.copy()
            y_change_label1 = y_train_minority_label1.copy()
            X_refer_label1 = X_train_majority_label1
            y_refer_label1 = y_train_majority_label1
            print('7. 计算出varphi和q')
            non_zero_count_minority_label1 = np.sum(fairness_shapley_minority_value_label1 < -0.3 )
            print(f"---在X_train_minority_label1中shapely value中小于-0.1的值的个数有: {non_zero_count_minority_label1}")
            non_zero_count = non_zero_count_minority_label1
            q_label1 = DataComposer( 
                            x_counterfactual=X_refer_label1.values, 
                            joint_prob=matching_minority_label1, 
                            method="max").calculate_q()   # 什么是q?  X_change_label1在X_refer_label1中与之匹配的instances，挑出来组成了q_label1
            varphi = np.where(fairness_shapley_minority_value_label1 < -0.3, fairness_shapley_minority_value_label1, 0)
            varphi = np.abs(varphi)

        elif eo_g1 < eo_g0:
            print('---EO_g1 < EO_g0, 优化方向为增加EO_g1的TPR,即使得所有label=1的所有g1的数据点, 尽可能都预测为1')
            print('---因为现在minority group = g0, majority group = g1, 所以我们只对majority group进行修改')
            print('5. 将X_train_majority_label1与X_train_minority_label1进行匹配')
            matching_majority_label1 = NearestNeighborDataMatcher(X_labeled=X_train_majority_label1, X_unlabeled=X_train_minority_label1).match(n_neighbors=1)
            print('6. 使用FairSHAP, 替换X_train_majority_label1中的数据, 从 X_train_minority_label1中寻找到合适的值')
            fairness_shapley_majority_value_label1 = fairness_explainer_original.shap_values(
                                        X = X_train_majority_label1.values,
                                        Y = y_train_majority_label1.values,
                                        X_baseline = X_train_minority_label1.values,
                                        matching=matching_majority_label1,
                                        sample_size=2000,
                                        shap_sample_size="auto",
                                    )
            X_change_label1 = X_train_majority_label1.copy()
            y_change_label1 = y_train_majority_label1.copy()
            X_refer_label1 = X_train_minority_label1
            y_refer_label1 = y_train_minority_label1
            print('7. 计算出varphi和q')
            non_zero_count_majority_label1 = np.sum(fairness_shapley_majority_value_label1 < -0.3)
            print(f"---在X_train_majority_label1中shapely value中小于-0.1的值的个数有: {non_zero_count_majority_label1}")
            non_zero_count = non_zero_count_majority_label1
            q_label1 = DataComposer( 
                            x_counterfactual=X_change_label1.values, 
                            joint_prob=matching_majority_label1, 
                            method="max").calculate_q()   # 什么是q?  X_change_label1在X_refer_label1中与之匹配的instances，挑出来组成了q_label1
            varphi = np.where(fairness_shapley_majority_value_label1 < -0.3, fairness_shapley_majority_value_label1, 0)
            varphi = np.abs(varphi)
        else:
            print('---EO_g1 == EO_g0, 无需使用FairSHAP进行优化')
            return None



        print(f'7. 开始整理minority部分的修改和majority部分的修改并且合并新数据,共修改{non_zero_count}个数据点, 使用new training set训练新模型')
        values_range = np.arange(1, non_zero_count, self.gap)
        result_fairness_measures = {'DP':[], 'EO':[], 'PQP':[], 'DR':[]}
        result_accuracy = {'DP':[], 'EO':[], 'PQP':[], 'DR':[]}

        DR_results = []
        DP_results = []
        EO_results = []
        EO_G1_results = []
        EO_G0_results = []
        PQP_results = []
        recall_results = []
        precision_results = []
        sufficiency_results = []

        for action_number in values_range:
            # Step 1: 将 varphi 的值和位置展开为一维
            flat_varphi = [(value, row, col) for row, row_vals in enumerate(varphi)
                        for col, value in enumerate(row_vals)]
            # Step 2: 按值降序排序
            flat_varphi_sorted = sorted(flat_varphi, key=lambda x: x[0], reverse=True)
            # Step 3: 挑出前 action_number 个数的位置
            top_positions = flat_varphi_sorted[:action_number]
            
            for value, row_idx, col_idx in top_positions:
                X_change_label1.iloc[row_idx, col_idx] = q_label1[row_idx, col_idx]

            x = pd.concat([X_change_label1, X_refer_label1, X_train_minority_label0, X_train_majority_label0], axis=0)
            y = pd.concat([y_change_label1, y_refer_label1, y_train_minority_label0, y_train_majority_label0], axis=0)

            # Step 6: Train the new model            
            model_new = XGBClassifier()
            model_new.fit(x, y)

            # step7: 评估新模型在DR,DP,EO,PP上的表现
            new_DR = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, model_new)
            y_hat = model_new.predict(self.X_test)
            y_test = self.y_test
            g1_Cm, g0_Cm = marginalised_np_mat(y_test, y_hat, 1, priv_idx)
            new_DP = grp1_DP(g1_Cm, g0_Cm)[0]
            new_EO = grp2_EO(g1_Cm, g0_Cm)[0]
            new_eo_g1 = grp2_EO(g1_Cm, g0_Cm)[1]
            new_eo_g0 = grp2_EO(g1_Cm, g0_Cm)[2]
            new_PQP = grp3_PQP(g1_Cm, g0_Cm)[0]
            new_recall, new_precision, new_sufficiency = calculate_metrics(self.y_test, y_hat, pos=1)
  

            DR_results.append(new_DR)
            DP_results.append(new_DP)
            EO_results.append(new_EO)
            EO_G0_results.append(new_eo_g0)
            EO_G1_results.append(new_eo_g1)
            PQP_results.append(new_PQP)
            recall_results.append(new_recall)
            precision_results.append(new_precision)
            sufficiency_results.append(new_sufficiency)

        
        print('8. 保存结果到csv文件')
        df = pd.DataFrame({
            "action_number": values_range,  # 直接使用 values_range
            "new_DR": DR_results,
            "new_DP": DP_results,
            "new_EO": EO_results,
            'new_eo_g1': EO_G1_results,
            'new_eo_g0': EO_G0_results,
            "new_PQP": PQP_results,
            'new_recall': recall_results,
            'new_precision': precision_results,
            'new_sufficiency': sufficiency_results,

        })
        # 在 DataFrame 的第一行添加 original 值
        df.loc[-1] = ["original", original_DR, original_DP, original_EO, eo_g1, eo_g0, original_PQP, original_recall, original_precision, original_sufficiency]  # 插入到第一行
        df.index = df.index + 1  # 重新索引
        df = df.sort_index()  # 确保 original 行在最上面

        dataset_folder = os.path.join('saved_results', self.dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)

        # 生成 CSV 文件名
        csv_filename = f"fairSHAP-{self.fairshap_base}_{self.ith_fold}-fold_results.csv"
        csv_filepath = os.path.join(dataset_folder, csv_filename)

        # 保存 CSV
        df.to_csv(csv_filepath, index=False)
        print(f"CSV 文件已保存：{csv_filepath}")




    def _split_into_majority_minority_label0_label1(self):
        '''
        This function is used to divide the dataset into sensitive_attribute g0 and g1.                     ----- (dont use 'majority group and minority group' now)

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

        print('Now we assume that: g1 is majority group, g0 is minority group')

        group_division = self.X_train[self.sensitive_attri].value_counts()
        '''把X_train分成majority和minority两个部分'''
        # if group_division[0] > group_division[1]:  #
        #     self.majority_group = 0
        #     self.minority_group = 1
        #     majority = self.X_train[self.sensitive_attri] == 0
        #     X_train_majority = self.X_train[majority]
        #     y_train_majority = self.y_train[majority]
        #     minority = self.X_train[self.sensitive_attri] == 1
        #     X_train_minority = self.X_train[minority]
        #     y_train_minority = self.y_train[minority]

        # else:
        #     self.majority_group = 1
        #     self.minority_group = 0
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