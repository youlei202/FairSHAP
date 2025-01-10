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
# ORIGINAL_DR = 0.05939711630344391

class Experiment:
    """
    The core part of how our experiments are conducted.

    ```
    Args:
        orginal_model: the original model, which is trained by X_train and y_train. 
        X_train: the training data. 
        y_train: the training label.
        X_unlabel: the unlabel data.
        y_unlabel: the unlabel label.

        sex_balance: 新增的unlabel data是否要保持性别平衡。
        Proportion: 从unlabel data中随机抽出的数据量比例。(X_label data在这抽出的部分里面寻找matching的data, 然后进行后续工作)
        replacement: 从unlabel data中抽出的数据, 是否要放回。
        num_new_data(int): 从unlabel data中抽出的数据量! 是X_train的多少倍。
        matcher: 选择matching的方法, 目前有两种, 一种是'nn':nearnest neighbour, 一种是'ot':optimal transport。

    Methods:
        get_result: 实验的主要部分, 进行实验, 并且输出实验结果。
        combination: 这部分是把不同的proportion, 以及不同的num_new_data组合起来, 然后进行实验。
        
    """ 
    def __init__(self,
            orginal_model, 
            X_train, 
            y_train,
            X_test,
            y_test,
            X_unlabel,
            y_unlabel, 
            dataset_name,
            ):
        self.orginal_model = orginal_model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_unlabel = X_unlabel
        self.y_unlabel = y_unlabel
        self.dataset_name = dataset_name
        if self.dataset_name == 'german_credit':
            self.target_name = 'Risk'
            self.original_DR =  0.010580012574791908
            self.gap = 1

        elif self.dataset_name == 'uci':
            self.target_name = 'income'
            self.original_DR = 0.05939711630344391
            self.gap =50
        elif self.dataset_name == 'census_income_kdd':
            self.target_name = 'class'
            self.original_DR = 0.08210
            self.gap = 100
        else:
            raise ValueError('dataset_name should be german_credit or uci')
    def get_result(
            self,
            sex_balance = False, 
            proportion = 0.5,
            replacement = True, 
            num_new_data = 3,
            matcher = 'nn'
            ):
        '''
        这部分是把实验的结果进行输出，并且进行可视化。
        '''
        self.sex_balance = sex_balance
        self.proportion = proportion
        self.replacement = replacement
        self.num_new_data = num_new_data
        self.matcher = matcher

        '''  1. 从X_unlabel中按照比例随机抽出num_new_data组数据  '''
        random_picks = self.random_pick_groups()    # dataframe type

        '''  2. X_label分别与random_picks[0,1,2....,num_new_data-1]进行matching, 找到matching的数据  ''' 
        matchings = self.get_matching(random_picks)   # matchings[0], matchings[1], matchings[2]...



        # Set the sensitive variables, and initilize the fairness_explainer - 这里已经默认是sex, 其中男性是privilege group 
        sen_att_name = ["sex"]
        sen_att = [self.X_train.columns.get_loc(name) for name in sen_att_name]
        priv_val = [1]
        unpriv_dict = [list(set(self.X_train.values[:, sa])) for sa in sen_att]
        for sa_list, pv in zip(unpriv_dict, priv_val):
            sa_list.remove(pv)
        fairness_explainer = FairnessExplainer(
            model=self.orginal_model, 
            sen_att=sen_att, 
            priv_val=priv_val, 
            unpriv_dict=unpriv_dict
            )

        '''  3. 计算每组数据的fairness shapley value  '''
        print(f'开始第3步, 计算每组数据的fairness shapley value')
        fairness_shapley_values = self.get_fairness_shapley_values(random_picks, matchings, fairness_explainer)  # fairness_shapley_values[0], fairness_shapley_values[1], fairness_shapley_values[2]...



        '''  4. 对shapely value不取绝对值, 然后把负值直接变成0, 然后在归一化, 得到新的varphi'''
        print(f'开始第4步, 对shapely value不取绝对值, 然后把负值直接变成0, 然后在归一化, 得到新的varphi')
        varphis = self.get_varphis(fairness_shapley_values, absolute_value=False)

        '''  5. 计算q ---Use joint_prob to find the matching unlabeled data to each labeled data instance  '''
        print(f'开始第5步, 计算q')
        q = self.get_q(random_picks, matchings)

        '''  6. 计算出总共可以修改的actions number, 并且把新的unlabel data整合好, 加入到X_train中, 返回合并后的数据
                重新训练模型, 并且评估性能 
        '''
        print(f'开始第6步, 计算出总共可以修改的actions number, 并且把新的unlabel data整合好, 加入到X_train中, 返回合并后的数据, 重新训练模型, 并且评估性能')
        x_train_with_target = pd.concat([self.X_train, self.y_train], axis=1)
        new_synthetic_data, after_values = self.process_varphis_and_modify_x(varphis, q, x_train_with_target, action_number='test')

        print(f'Proporation: {self.proportion}, new_data_number: {self.num_new_data} 训练结束, DR值已经保存, 可以进行可视化')
        print(f'-----------------------------------------------------------------------------')
        return after_values









    '''
    把里面的一些method的输入参数进行修改,不要使用self.X_unlabel, 而是直接传入X_unlabel。
    
    这样可以在一开始把X_train和X_unlabel分成sex=0,1两个部分，然后把两个部分分别传进去，
    分别构造两个df，再把这两个df合并起来，然后再进行后续的操作。这部分不属于sex balance，
    这部分属于sex separate match，看看有没有效果。
    '''
    def get_sex_separate_nn_result(
            self,
            sex_balance = False, 
            proportion = 0.5,
            replacement = True, 
            num_new_data = 3,
            matcher = 'nn',
            match_method = 'sex_separate'   # 'together','sex_separate','sex_cross'
            ):
        '''
        这部分是把实验的结果进行输出，并且进行可视化。

        其中match_method有三种， 'together','sex_separate','sex_cross'
        '''

        self.sex_balance = sex_balance
        self.proportion = proportion
        self.replacement = replacement
        self.num_new_data = num_new_data
        self.matcher = matcher
        self.match_method = match_method


        if self.match_method == 'together':
            '''  1. 从X_unlabel中按照比例随机抽出num_new_data组数据  '''
            random_picks = self.random_pick_groups(self.X_unlabel)    # dataframe type

            '''  2. X_label分别与random_picks[0,1,2....,num_new_data-1]进行matching, 找到matching的数据  ''' 
            matchings = self.get_matching(random_picks, self.X_train)   # matchings[0], matchings[1], matchings[2]...

            # Set the sensitive variables, and initilize the fairness_explainer - 这里已经默认是sex, 其中男性是privilege group 
            sen_att_name = ["sex"]
            sen_att = [self.X_train.columns.get_loc(name) for name in sen_att_name]
            priv_val = [1]
            unpriv_dict = [list(set(self.X_train.values[:, sa])) for sa in sen_att]
            for sa_list, pv in zip(unpriv_dict, priv_val):
                sa_list.remove(pv)
            fairness_explainer = FairnessExplainer(
                model=self.orginal_model, 
                sen_att=sen_att, 
                priv_val=priv_val, 
                unpriv_dict=unpriv_dict
                )

            '''  3. 计算每组数据的fairness shapley value  '''
            print(f'开始第3步, 计算每组数据的fairness shapley value')
            fairness_shapley_values = self.get_fairness_shapley_values(self.X_train, random_picks, matchings, fairness_explainer)  # fairness_shapley_values[0], fairness_shapley_values[1], fairness_shapley_values[2]...

            '''  4. 对shapely value不取绝对值, 然后把负值直接变成0, 然后在归一化, 得到新的varphi'''
            print(f'开始第4步, 对shapely value不取绝对值, 然后把负值直接变成0, 然后在归一化, 得到新的varphi')
            varphis = self.get_varphis(fairness_shapley_values, absolute_value=False)

            '''  5. 计算q ---Use joint_prob to find the matching unlabeled data to each labeled data instance  '''
            print(f'开始第5步, 计算q')
            q = self.get_q(random_picks, matchings)

            '''  6. 计算出总共可以修改的actions number, 并且把新的unlabel data整合好, 加入到X_train中, 返回合并后的数据
                    重新训练模型, 并且评估性能 '''
            print(f'开始第6步, 计算出总共可以修改的actions number, 并且把新的unlabel data整合好, 加入到X_train中, 返回合并后的数据, 重新训练模型, 并且评估性能')
            x_train_with_target = pd.concat([self.X_train, self.y_train], axis=1)
            new_synthetic_data, after_values, fairness_accuracy_pairs = self.process_varphis_and_modify_x(varphis, q, x_train_with_target, action_number='test')

        elif self.match_method == 'sex_separate': 
            # 将 Sex 列等于 0 的样本分离
            X_train_sex0 = self.X_train[self.X_train["sex"] == 0]
            y_train_sex0 = self.y_train[self.X_train["sex"] == 0]
            # 将 Sex 列等于 1 的样本分离
            X_train_sex1 = self.X_train[self.X_train["sex"] == 1]
            y_train_sex1 = self.y_train[self.X_train["sex"] == 1]

            #将X_unlabel分成两部分
            X_unlabel_sex0 = self.X_unlabel[self.X_unlabel["sex"] == 0]
            X_unlabel_sex1 = self.X_unlabel[self.X_unlabel["sex"] == 1]
            y_unlabel_sex0 = self.y_unlabel[self.X_unlabel["sex"] == 0]
            y_unlabel_sex1 = self.y_unlabel[self.X_unlabel["sex"] == 1]

            '''  1. 从X_unlabel中按照比例随机抽出num_new_data组数据  '''
            random_picks_sex0 = self.random_pick_groups(X_unlabel_sex0)    # dataframe type
            random_picks_sex1 = self.random_pick_groups(X_unlabel_sex1)    # dataframe type

            '''  2. X_label分别与random_picks[0,1,2....,num_new_data-1]进行matching, 找到matching的数据  ''' 
            matchings_sex0 = self.get_matching(random_picks_sex0, X_train_sex0)   # matchings_sex0[0], matchings_sex0[1], matchings_sex0[2]...
            matchings_sex1 = self.get_matching(random_picks_sex1, X_train_sex1)   # matchings[0], matchings[1], matchings[2]...

            # Set the sensitive variables, and initilize the fairness_explainer - 这里已经默认是sex, 其中男性是privilege group 
            sen_att_name = ["sex"]
            sen_att = [self.X_train.columns.get_loc(name) for name in sen_att_name]
            priv_val = [1]
            unpriv_dict = [list(set(self.X_train.values[:, sa])) for sa in sen_att]
            for sa_list, pv in zip(unpriv_dict, priv_val):
                sa_list.remove(pv)
            fairness_explainer = FairnessExplainer(
                model=self.orginal_model, 
                sen_att=sen_att, 
                priv_val=priv_val, 
                unpriv_dict=unpriv_dict
                )

            '''  3. 计算每组数据的fairness shapley value  '''
            print(f'开始第3步, 计算每组数据的fairness shapley value')
            fairness_shapley_values_sex0 = self.get_fairness_shapley_values(X_train_sex0, random_picks_sex0, matchings_sex0, fairness_explainer)  # fairness_shapley_values[0], fairness_shapley_values[1], fairness_shapley_values[2]...
            fairness_shapley_values_sex1 = self.get_fairness_shapley_values(X_train_sex1, random_picks_sex1, matchings_sex1, fairness_explainer)  # fairness_shapley_values[0], fairness_shapley_values[1], fairness_shapley_values[2]...

            '''  4. 对shapely value不取绝对值, 然后把负值直接变成0, 然后在归一化, 得到新的varphi'''
            print(f'开始第4步, 对shapely value不取绝对值, 然后把负值直接变成0, 然后在归一化, 得到新的varphi')
            varphis_sex0 = self.get_varphis(fairness_shapley_values_sex0, absolute_value=False)
            varphis_sex1 = self.get_varphis(fairness_shapley_values_sex1, absolute_value=False)
            '''  5. 计算q ---Use joint_prob to find the matching unlabeled data to each labeled data instance  '''
            print(f'开始第5步, 计算q')
            q_sex0 = self.get_q(random_picks_sex0, matchings_sex0)
            q_sex1 = self.get_q(random_picks_sex1, matchings_sex1)

            '''  6. 计算出总共可以修改的actions number, 并且把新的unlabel data整合好, 加入到X_train中, 返回合并后的数据
                    重新训练模型, 并且评估性能 '''
            print(f'开始第6步, 计算出总共可以修改的actions number, 并且把新的unlabel data整合好, 加入到X_train中, 返回合并后的数据, 重新训练模型, 并且评估性能')
            x_train_with_target_sex0 = pd.concat([X_train_sex0, y_train_sex0], axis=1)
            x_train_with_target_sex1 = pd.concat([X_train_sex1, y_train_sex1], axis=1)


            '''
            下面把sex=0, sex=1数据合成一起
            '''
            for i in range(self.num_new_data):
                if i == 0:
                    x_copy_sex0 = x_train_with_target_sex0.copy()
                    x_copy_sex1 = x_train_with_target_sex1.copy()
                    varphi_combined_sex0 = varphis_sex0[i]
                    varphi_combined_sex1 = varphis_sex1[i]
                    q_combined_sex0 = q_sex0[i]
                    q_combined_sex1 = q_sex1[i]
                else:
                    x_copy_sex0 = pd.concat([x_train_with_target_sex0, x_copy_sex0], axis=0)
                    x_copy_sex1 = pd.concat([x_train_with_target_sex1, x_copy_sex1], axis=0)
                    varphi_combined_sex0 = np.vstack([varphi_combined_sex0, varphis_sex0[i]])
                    varphi_combined_sex1 = np.vstack([varphi_combined_sex1, varphis_sex1[i]])
                    q_combined_sex0 = np.vstack([q_combined_sex0, q_sex0[i]])
                    q_combined_sex1 = np.vstack([q_combined_sex1, q_sex1[i]])
            
            x_copy = pd.concat([x_copy_sex0, x_copy_sex1], axis=0)
            varphi_combined = np.vstack([varphi_combined_sex0, varphi_combined_sex1])
            q_combined = np.vstack([q_combined_sex0, q_combined_sex1])

            non_zero_count = (varphi_combined != 0).sum().sum()
            print(f"Total number of non-zero values across all varphis: {non_zero_count}")

            # self.limited_values_range = 0
            self.limited_values_range = np.arange(1, non_zero_count, self.gap)
            before_values = []
            after_values = []
            fairness_accuracy_pairs = []
            for action_number in self.limited_values_range:


                # Step 1: 将 varphi_combined 的值和位置展开为一维
                flat_varphi = [(value, row, col) for row, row_vals in enumerate(varphi_combined)
                            for col, value in enumerate(row_vals)]

                # Step 2: 按值降序排序
                flat_varphi_sorted = sorted(flat_varphi, key=lambda x: x[0], reverse=True)

                # Step 3: 挑出前 action_number 个数的位置
                top_positions = flat_varphi_sorted[:action_number]

                # Step 4: 替换 X 中前三列的值为 S 中对应位置的值
                for value, row_idx, col_idx in top_positions:
                    x_copy.iloc[row_idx, col_idx] = q_combined[row_idx, col_idx]
                X_Train_New = pd.concat([x_train_with_target_sex0,x_train_with_target_sex1, x_copy], axis=0)
        
                # Step 5: Train and evaluate model
                target_name = self.target_name
                x = X_Train_New.drop(target_name, axis=1)
                y = X_Train_New[target_name]
                
                model_new = XGBClassifier()
                model_new.fit(x, y)
                
                # 计算fairness values
                sen_att_name = ["sex"]
                sen_att = [self.X_train.columns.get_loc(name) for name in sen_att_name]
                priv_val = [1]
                unpriv_dict = [list(set(self.X_train.values[:, sa])) for sa in sen_att]
                for sa_list, pv in zip(unpriv_dict, priv_val):
                    sa_list.remove(pv)

                after = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, model_new)
                after_values.append(after)
                
                if after < self.original_DR:
                    y_new_pred = model_new.predict(self.X_test)
                    accuracy_new = accuracy_score(self.y_test, y_new_pred)
                    fairness_accuracy_pairs.append((after, accuracy_new, action_number))
    


        elif self.match_method == 'sex_cross':
            # 将 Sex 列等于 0 的样本分离
            X_train_sex0 = self.X_train[self.X_train["sex"] == 0]
            y_train_sex0 = self.y_train[self.X_train["sex"] == 0]
            # 将 Sex 列等于 1 的样本分离
            X_train_sex1 = self.X_train[self.X_train["sex"] == 1]
            y_train_sex1 = self.y_train[self.X_train["sex"] == 1]

            #将X_unlabel分成两部分
            X_unlabel_sex0 = self.X_unlabel[self.X_unlabel["sex"] == 0]
            X_unlabel_sex1 = self.X_unlabel[self.X_unlabel["sex"] == 1]
            y_unlabel_sex0 = self.y_unlabel[self.X_unlabel["sex"] == 0]
            y_unlabel_sex1 = self.y_unlabel[self.X_unlabel["sex"] == 1]

            '''  1. 从X_unlabel中按照比例随机抽出num_new_data组数据  '''
            random_picks_sex0 = self.random_pick_groups(X_unlabel_sex0)    # dataframe type
            random_picks_sex1 = self.random_pick_groups(X_unlabel_sex1)    # dataframe type

            '''  2. X_label分别与random_picks[0,1,2....,num_new_data-1]进行matching, 找到matching的数据  ''' 
            matchings_sex0 = self.get_matching(random_picks_sex1, X_train_sex0)   # original sex0 与 x_unlabel sex1匹配
            matchings_sex1 = self.get_matching(random_picks_sex0, X_train_sex1)   # original sex1 与 x_unlabel sex0匹配

            # Set the sensitive variables, and initilize the fairness_explainer - 这里已经默认是sex, 其中男性是privilege group 
            sen_att_name = ["sex"]
            sen_att = [self.X_train.columns.get_loc(name) for name in sen_att_name]
            priv_val = [1]
            unpriv_dict = [list(set(self.X_train.values[:, sa])) for sa in sen_att]
            for sa_list, pv in zip(unpriv_dict, priv_val):
                sa_list.remove(pv)
            fairness_explainer = FairnessExplainer(
                model=self.orginal_model, 
                sen_att=sen_att, 
                priv_val=priv_val, 
                unpriv_dict=unpriv_dict
                )

            '''  3. 计算每组数据的fairness shapley value  '''
            print(f'开始第3步, 计算每组数据的fairness shapley value')
            fairness_shapley_values_sex0 = self.get_fairness_shapley_values(X_train_sex0, random_picks_sex1, matchings_sex0, fairness_explainer)  # fairness_shapley_values[0], fairness_shapley_values[1], fairness_shapley_values[2]...
            fairness_shapley_values_sex1 = self.get_fairness_shapley_values(X_train_sex1, random_picks_sex0, matchings_sex1, fairness_explainer)  # fairness_shapley_values[0], fairness_shapley_values[1], fairness_shapley_values[2]...

            '''  4. 对shapely value不取绝对值, 然后把负值直接变成0, 然后在归一化, 得到新的varphi'''
            print(f'开始第4步, 对shapely value不取绝对值, 然后把负值直接变成0, 然后在归一化, 得到新的varphi')
            varphis_sex0 = self.get_varphis(fairness_shapley_values_sex0, absolute_value=False)
            varphis_sex1 = self.get_varphis(fairness_shapley_values_sex1, absolute_value=False)
            '''  5. 计算q ---Use joint_prob to find the matching unlabeled data to each labeled data instance  '''
            print(f'开始第5步, 计算q')
            q_sex0 = self.get_q(random_picks_sex1, matchings_sex0)  #与sex=0相匹配的，sex=1的unlabel data
            q_sex1 = self.get_q(random_picks_sex0, matchings_sex1)  #与sex=1相匹配的，sex=0的unlabel data

            '''  6. 计算出总共可以修改的actions number, 并且把新的unlabel data整合好, 加入到X_train中, 返回合并后的数据
                    重新训练模型, 并且评估性能 '''
            print(f'开始第6步, 计算出总共可以修改的actions number, 并且把新的unlabel data整合好, 加入到X_train中, 返回合并后的数据, 重新训练模型, 并且评估性能')
            x_train_with_target_sex0 = pd.concat([X_train_sex0, y_train_sex0], axis=1)
            x_train_with_target_sex1 = pd.concat([X_train_sex1, y_train_sex1], axis=1)


            '''
            下面把sex=0, sex=1数据合成一起
            '''
            for i in range(self.num_new_data):
                if i == 0:
                    x_copy_sex0 = x_train_with_target_sex0.copy()
                    x_copy_sex1 = x_train_with_target_sex1.copy()
                    varphi_combined_sex0 = varphis_sex0[i]
                    varphi_combined_sex1 = varphis_sex1[i]
                    q_combined_sex0 = q_sex0[i]
                    q_combined_sex1 = q_sex1[i]
                else:
                    x_copy_sex0 = pd.concat([x_train_with_target_sex0, x_copy_sex0], axis=0)
                    x_copy_sex1 = pd.concat([x_train_with_target_sex1, x_copy_sex1], axis=0)
                    varphi_combined_sex0 = np.vstack([varphi_combined_sex0, varphis_sex0[i]])
                    varphi_combined_sex1 = np.vstack([varphi_combined_sex1, varphis_sex1[i]])
                    q_combined_sex0 = np.vstack([q_combined_sex0, q_sex0[i]])
                    q_combined_sex1 = np.vstack([q_combined_sex1, q_sex1[i]])
            
            x_copy = pd.concat([x_copy_sex0, x_copy_sex1], axis=0)
            varphi_combined = np.vstack([varphi_combined_sex0, varphi_combined_sex1])
            q_combined = np.vstack([q_combined_sex0, q_combined_sex1])

            non_zero_count = (varphi_combined != 0).sum().sum()
            print(f"Total number of non-zero values across all varphis: {non_zero_count}")

            # self.limited_values_range = 0
            self.limited_values_range = np.arange(1, non_zero_count, self.gap)
            after_values = []
            fairness_accuracy_pairs = []
            for action_number in self.limited_values_range:


                # Step 1: 将 varphi_combined 的值和位置展开为一维
                flat_varphi = [(value, row, col) for row, row_vals in enumerate(varphi_combined)
                            for col, value in enumerate(row_vals)]

                # Step 2: 按值降序排序
                flat_varphi_sorted = sorted(flat_varphi, key=lambda x: x[0], reverse=True)

                # Step 3: 挑出前 action_number 个数的位置
                top_positions = flat_varphi_sorted[:action_number]

                # Step 4: 替换 X 中前三列的值为 S 中对应位置的值
                for value, row_idx, col_idx in top_positions:
                    x_copy.iloc[row_idx, col_idx] = q_combined[row_idx, col_idx]
                X_Train_New = pd.concat([x_train_with_target_sex0,x_train_with_target_sex1, x_copy], axis=0)
        
                # Step 5: Train and evaluate model
                target_name = self.target_name
                x = X_Train_New.drop(target_name, axis=1)
                y = X_Train_New[target_name]
                
                model_new = XGBClassifier()
                model_new.fit(x, y)
                
                # 计算fairness values
                sen_att_name = ["sex"]
                sen_att = [self.X_train.columns.get_loc(name) for name in sen_att_name]
                priv_val = [1]
                unpriv_dict = [list(set(self.X_train.values[:, sa])) for sa in sen_att]
                for sa_list, pv in zip(unpriv_dict, priv_val):
                    sa_list.remove(pv)
                after = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, model_new)
                after_values.append(after)
                if after < self.original_DR:
                    y_new_pred = model_new.predict(self.X_test)
                    accuracy_new = accuracy_score(self.y_test, y_new_pred)
                    fairness_accuracy_pairs.append((after, accuracy_new, action_number))  # Store both values as a tuple
                

        else:
            raise ValueError('match_method should be together, sex_separate, sex_cross')
        print(f'Proporation: {self.proportion}, new_data_number: {self.num_new_data} 训练结束, match_method:{self.match_method},DR值已经保存, 可以进行可视化')
        print(f'-----------------------------------------------------------------------------')
        return after_values, fairness_accuracy_pairs














    def visualize(self, after_values):
        # after_values = self.get_result()
        plt.scatter(self.limited_values_range, after_values, label='New model', marker='x')
        plt.axhline(y=self.original_DR, color='r', linestyle='--', label='Original DR')
        plt.xlabel('Limited actions')
        plt.ylabel('DR Value')
        plt.title('Fairness Value vs. Limited Values')
        plt.legend()
        plt.show()


    def combination(self, match_met):
        '''
        这部分是把不同的proportion, 以及不同的num_new_data组合起来, 然后进行实验。
        '''
        # 保存结果
        results = {}

        # 根据数据集名称确定迭代范围和子图布局
        if self.dataset_name == 'census_income_kdd':
            proportions = [0.4, 0.6, 0.8]
            num_new_data_values = [1, 2, 3]
            rows, cols = 3, 3
        else:
            proportions = [0.2, 0.4, 0.6, 0.8]
            num_new_data_values = [1, 2, 3]
            rows, cols = 4, 3

        # 创建子图布局
        fig1, axes1 = plt.subplots(rows, cols, figsize=(10, 12))
        fig1.suptitle("Fairness Value vs. Limited Actions for Different Parameters", fontsize=16)

        fig2, axes2 = plt.subplots(rows, cols, figsize=(10, 12))
        fig2.suptitle("Fairness-Accuracy Trade-off for Different Parameters", fontsize=16)

        # 用于跟踪当前子图的位置
        plot_index = 0

        # 迭代不同的 proportion 和 num_new_data 组合
        for proportion in proportions:
            for num_new_data in num_new_data_values:
                # 获取当前的子图轴
                ax1 = axes1[plot_index // cols, plot_index % cols]
                ax2 = axes2[plot_index // cols, plot_index % cols]

                # 使用 get_result() 获取结果
                after_values, fairness_accuracy_pairs = self.get_sex_separate_nn_result(
                    sex_balance=False,
                    proportion=proportion,
                    replacement=True,
                    num_new_data=num_new_data,
                    matcher='nn',
                    match_method=match_met
                )

                # 保存结果到字典中
                results[(proportion, num_new_data)] = after_values

                # 在第一个图的子图上绘制原始结果
                ax1.scatter(self.limited_values_range, after_values, label='New model', marker='x')
                ax1.axhline(y=self.original_DR, color='r', linestyle='--', label='Original DR')
                ax1.set_title(f'Proportion: {proportion}, Num New Data: {num_new_data}', fontsize=10)
                ax1.set_xlabel('Limited actions')
                ax1.set_ylabel('DR Value')
                ax1.legend()

                # 在第二个图的子图上绘制fairness-accuracy散点图
                if fairness_accuracy_pairs:  # 确保有数据可画
                    fairness_values, accuracy_values, action_numbers = zip(*fairness_accuracy_pairs)

                    # 创建颜色映射
                    min_action = min(action_numbers)
                    max_action = max(action_numbers)
                    norm = plt.Normalize(min_action, max_action)

                    # 创建从淡黄色到黑色的颜色映射
                    cmap = plt.cm.YlOrBr  # YlOrBr colormap: 从淡黄色过渡到深褐色/黑色

                    # 绘制散点图，使用圆形标记和颜色映射
                    scatter = ax2.scatter(accuracy_values, fairness_values, 
                                        c=action_numbers, 
                                        cmap=cmap, 
                                        norm=norm,
                                        marker='o',  # 使用圆形标记
                                        alpha=0.6)

                    # 添加颜色条
                    cbar = plt.colorbar(scatter, ax=ax2)
                    cbar.set_label('Action Number')

                    ax2.set_title(f'Proportion: {proportion}, Num New Data: {num_new_data}', fontsize=10)
                    ax2.set_xlabel('Accuracy')
                    ax2.set_ylabel('Fairness Value')
                    ax2.grid(True, linestyle='--', alpha=0.7)

                # 更新子图索引
                plot_index += 1

        # 自动调整两个图的子图布局
        fig1.tight_layout(rect=[0, 0, 1, 0.95])  # rect 调整避免标题重叠
        fig2.tight_layout(rect=[0, 0, 1, 0.95])

        # 显示图表
        plt.show()
        
        return results






























    def random_pick_groups(self, X_unlabel):

        if self.replacement == True:
            # calculate the sample size
            sample_size = int(len(X_unlabel) * self.proportion)
            # random pick
            group_num = self.num_new_data
            random_picks = []
            for i in range(group_num):
                random_pick = X_unlabel.sample(n=sample_size, random_state=25+i)
                random_picks.append(random_pick)
        else:
            raise ValueError('replacement should be True')
        return tuple(random_picks)



    def get_matching(self, random_pick, X_train):
        matchings = []
        for i in range(self.num_new_data):
            if self.matcher == 'nn':
                matching = NearestNeighborDataMatcher(X_labeled=X_train, X_unlabeled=random_pick[i]).match(n_neighbors=1)
                matchings.append(matching)

            elif self.matcher == 'ot':
                matching = OptimalTransportPolicy(X_labeled=X_train, X_unlabeled=random_pick[i]).match(n_neighbors=1)
                matchings.append(matching)
            else:
                raise ValueError('matcher should be nn or ot')
        return tuple(matchings)

    def get_fairness_shapley_values(self, X_train, random_picks, matchings, fairness_explainer):
        fairness_shapley_values = []
        breakpoint()
        for i in range(self.num_new_data):
            fairness_shapley_value = fairness_explainer.shap_values(
                                        X = X_train.values,
                                        X_baseline = random_picks[i].values,
                                        matching=matchings[i],
                                        sample_size=500,
                                        shap_sample_size="auto",
                                    )
            fairness_shapley_values.append(fairness_shapley_value)
        return tuple(fairness_shapley_values)

    def get_varphis(self, fairness_shapley_values, absolute_value=False):
        varphis = []
        if absolute_value == False:
            for i in range(self.num_new_data):
                varphi = fix_negative_probabilities(fairness_shapley_values[i])
                varphis.append(varphi)
        else:
            for i in range(self.num_new_data):
                varphi = convert_matrix_to_policy(fairness_shapley_values[i])
                varphis.append(varphi)
        return tuple(varphis)

    def get_q(self, random_picks, matchings):
        qs = []
        for i in range(self.num_new_data):
            q = DataComposer(
                x_counterfactual=random_picks[i].values, 
                joint_prob=matchings[i], 
                method="max").calculate_q() 
            qs.append(q)
        return tuple(qs)



    def process_varphis_and_modify_x(self,
                                    varphis: Tuple, 
                                    q: Tuple, 
                                    x_train_with_target: pd.DataFrame,
                                    action_number: int=True) -> pd.DataFrame:
        """
        Process num varphis and modify copies of x based on top action_number non-zero values.
        
        Args:
            varphis: Tuple of arrays representing modifications to x
            x: Original array to be modified
            qs: Tuple of reference arrays for replacement values
            num: Number of varphis to process
            action_number: Number of top positions to modify
        
        Returns:
            np.ndarray: Concatenated array of all modified x versions
        """

        for i in range(self.num_new_data):
            if i == 0:
                x_copy = x_train_with_target.copy()
                varphi_combined = varphis[i]
                q_combined = q[i]
            else:
                x_copy = pd.concat([x_train_with_target, x_copy], axis=0)
                varphi_combined = np.vstack([varphi_combined, varphis[i]])
                q_combined = np.vstack([q_combined, q[i]])
                
        non_zero_count = (varphi_combined != 0).sum().sum()
        print(f"Total number of non-zero values across all varphis: {non_zero_count}")

        # self.limited_values_range = 0
        self.limited_values_range = np.arange(1, non_zero_count, self.gap)
        before_values = []
        after_values = []
        fairness_accuracy_pairs = []
        for action_number in self.limited_values_range:


            # Step 1: 将 varphi_combined 的值和位置展开为一维
            flat_varphi = [(value, row, col) for row, row_vals in enumerate(varphi_combined)
                        for col, value in enumerate(row_vals)]

            # Step 2: 按值降序排序
            flat_varphi_sorted = sorted(flat_varphi, key=lambda x: x[0], reverse=True)

            # Step 3: 挑出前 action_number 个数的位置
            top_positions = flat_varphi_sorted[:action_number]

            # Step 4: 替换 X 中前三列的值为 S 中对应位置的值
            for value, row_idx, col_idx in top_positions:
                x_copy.iloc[row_idx, col_idx] = q_combined[row_idx, col_idx]
            X_Train_New = pd.concat([x_train_with_target, x_copy], axis=0)
      
            # Step 5: Train and evaluate model
            target_name = self.target_name
            x = X_Train_New.drop(target_name, axis=1)
            y = X_Train_New[target_name]
            
            model_new = XGBClassifier()
            model_new.fit(x, y)
            
            # 计算fairness values
            sen_att_name = ["sex"]
            sen_att = [self.X_train.columns.get_loc(name) for name in sen_att_name]
            priv_val = [1]
            unpriv_dict = [list(set(self.X_train.values[:, sa])) for sa in sen_att]
            for sa_list, pv in zip(unpriv_dict, priv_val):
                sa_list.remove(pv)

            after = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, model_new)
            after_values.append(after)
            if after < self.original_DR:
                y_new_pred = model_new.predict(self.X_test)
                accuracy_new = accuracy_score(self.y_test, y_new_pred)
                fairness_accuracy_pairs.append((after, accuracy_new, action_number))

        return X_Train_New, after_values, fairness_accuracy_pairs

        ''' 
        这里varphi_combined是一个dataframe

       # Step 1: 获取 varphi_combined 的所有值及其位置
        flat_varphi = varphi_combined.unstack().reset_index()
        flat_varphi.columns = ['column', 'row', 'value']

        # Step 2: 按值降序排序
        flat_varphi_sorted = flat_varphi.sort_values(by='value', ascending=False)

        # Step 3: 挑出前 action_number 个数的位置
        top_positions = flat_varphi_sorted.head(action_number)

        # Step 4: 替换 X 中前三列的值为 S 中对应位置的值
        for _, row in top_positions.iterrows():
            col_name = row['column']  # 列名（A, B, C）
            row_idx = row['row']  # 行索引
            if col_name in ['A', 'B', 'C']:  # 只操作前三列
                X.at[row_idx, col_name] = S.at[row_idx, col_name]
        '''



        #     # Step 2: Sort all non-zero values and take top action_number positions
        #     all_positions.sort(key=lambda x: x[0], reverse=True)
        #     selected_positions = all_positions[:action_number]
            
        #     # Step 3: Create copies of x and modify them
        #     x_copies = []
        #     modifications = {i: [] for i in range(num)}  # Dictionary to store modifications for each copy
            
        #     # Group selected positions by their varphis index
        #     for _, pos, varphis_idx in selected_positions:
        #         modifications[varphis_idx].append(pos)
            
        #     # Create and modify num copies of x
        #     for i in range(num):
        #         x_copy = x_train_with_target.copy()
        #         for pos in modifications[i]:
        #             x_copy[pos] = q[i][pos]  # Use corresponding q array
        #         x_copies.append(x_copy)

        #     # Step 4: Concatenate all modified versions
        #     x_copies = np.vstack(x_copies)
        #     x_copies = pd.DataFrame(x_copies, columns=x_train_with_target.columns)

        #     X_Train_New = np.vstack(x_train_with_target, x_copies)
        #     X_Train_New = pd.concat([X_Train_New, x_copies], axis=0)

        #     # Step 5: Train and evaluate model        
        #     target_name = self.target_name
        #     x = X_Train_New.drop(target_name, axis=1)
        #     y = X_Train_New[target_name]
            
        #     model_new = XGBClassifier()
        #     model_new.fit(x, y)
            
        #     # 计算fairness values
        #     sen_att_name = ["sex"]
        #     sen_att = [self.X_train.columns.get_loc(name) for name in sen_att_name]
        #     priv_val = [1]
        #     unpriv_dict = [list(set(self.X_train.values[:, sa])) for sa in sen_att]
        #     for sa_list, pv in zip(unpriv_dict, priv_val):
        #         sa_list.remove(pv)

        #     after = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, model_new)
        #     after_values.append(after)

        # return X_Train_New, after_values





def convert_matrix_to_policy(matrix):
    P = np.abs(matrix) / np.abs(matrix).sum()
    P += EPSILON
    P /= P.sum()
    return P

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

''' 计算DR value的函数'''
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

def balance_sex_distribution(df, sex_column='sex'):
    """
    平衡数据集中性别分布，使得男性和女性样本数量相等
    
    Parameters:
    -----------
    df : pd.DataFrame
        需要平衡的数据集
    sex_column : str
        性别列的名称
        
    Returns:
    --------
    pd.DataFrame
        平衡后的数据集
    """
    # 分别获取性别为0和1的数据
    sex_0 = df[df[sex_column] == 0]
    sex_1 = df[df[sex_column] == 1]
    
    # 获取较小的样本数
    min_count = min(len(sex_0), len(sex_1))
    
    # 如果某一类样本数量更多，随机抽样减少到较小的数量
    if len(sex_0) > min_count:
        sex_0 = sex_0.sample(n=min_count, random_state=42)
    if len(sex_1) > min_count:
        sex_1 = sex_1.sample(n=min_count, random_state=42)
    # 合并平衡后的数据
    balanced_df = pd.concat([sex_0, sex_1])
    # 随机打乱数据顺序
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df