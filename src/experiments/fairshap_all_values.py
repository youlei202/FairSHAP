import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from fairlearn.preprocessing import CorrelationRemover
from sklearn.model_selection import KFold
from src.matching.ot_matcher import OptimalTransportPolicy
from src.matching.nn_matcher import NearestNeighborDataMatcher
from src.attribution import FairnessExplainer
from src.data.unified_dataloader import load_dataset
from src.attribution.oracle_metric import perturb_numpy_ver

import os
import numpy as np
from scipy.stats import wasserstein_distance
from src.attribution import FairnessExplainer
from src.composition.data_composer import DataComposer
from src.attribution.oracle_metric import perturb_numpy_ver
from sklearn.metrics.pairwise import rbf_kernel

class FairSHAP:
    def __init__(self, threshold=0.05, matching_method='NN', fairshap_base='DR'):
        # self.dataset_name = dataset_name
        self.threshold = threshold
        self.matching_method = matching_method
        self.fairshap_base = fairshap_base
    def run_and_save_results(self):
        '''
        Run the chosn SOTA method on all datasets and save the results in a csv file
        '''
        for i in ['german_credit', 'compas', 'compas4race', 'adult', 'default_credit', 'census_income_kdd']:
            self.dataset_name = i
            print("-------------------------------------")
            print(f"---!! on {self.dataset_name} dataset !!---")
            print("-------------------------------------")
            if self.dataset_name == 'german_credit':
                self.sen_att_name = 'sex'
                self.target_name = 'Risk'
                self.priv = 0
            elif self.dataset_name == 'adult':
                self.sen_att_name = 'sex'
                self.target_name = 'income'
                self.priv = 1
            elif self.dataset_name == 'compas':
                self.sen_att_name = 'sex'
                self.target_name = 'two_year_recid'
                self.priv =1
            elif self.dataset_name == 'compas4race':
                self.sen_att_name = 'race'
                self.target_name = 'two_year_recid'
                self.priv = 1
            elif self.dataset_name == 'census_income_kdd':
                self.sen_att_name = 'sex'
                self.target_name = 'class'
                self.priv = 0
            elif self.dataset_name == 'default_credit':
                self.sen_att_name = 'sex'
                self.target_name = 'default_payment_next_month'
                self.priv = 0
            _, processed_data = load_dataset(self.dataset_name)
            if self.dataset_name == 'census_income_kdd':
                processed_data = processed_data.sample(frac=0.1, random_state=25) 
            kf = KFold(n_splits=5, shuffle=True, random_state=1)  # 5-fold cross-validation

            original_accuracy = []
            original_dr = []
            original_dp = []
            original_eo = []
            original_pqp = []

            processed_accuracy = []
            processed_dr = []
            processed_dp = []
            processed_eo = []
            processed_pqp = []
            modification_num = []
            wasserstein_distance = []
            mmd_distance = []
            
            for j, (train_index, val_index) in enumerate(kf.split(processed_data)):
                print("-------------------------------------")
                print(f"-------------{j}th fold----------------")
                print("-------------------------------------")
                train_data, val_data = processed_data.iloc[train_index], processed_data.iloc[val_index]
                X_train, y_train = train_data.drop(self.target_name, axis=1), train_data[self.target_name]
                X_val, y_val = val_data.drop(self.target_name, axis=1), val_data[self.target_name]
                model = XGBClassifier()
                model.fit(X_train, y_train)
                print(f"1. Split the {self.dataset_name} dataset into majority group and minority group according to the number of sensitive attribute, besides split by label 0 and label 1")
                X_train_majority_label0, y_train_majority_label0, X_train_majority_label1, y_train_majority_label1, X_train_minority_label0, y_train_minority_label0, X_train_minority_label1, y_train_minority_label1 = self._split_into_majority_minority_label0_label1(X_train=X_train, y_train=y_train)

                print(f'X_train_majority_label0 shape: {X_train_majority_label0.shape}')
                print(f'X_train_majority_label1 shape: {X_train_majority_label1.shape}')
                print(f'X_train_minority_label0 shape: {X_train_minority_label0.shape}')
                print(f'X_train_minority_label1 shape: {X_train_minority_label1.shape}')
                print('2. 初始化FairnessExplainer')
                sen_att_name = [self.sen_att_name]
                sen_att = [X_val.columns.get_loc(name) for name in sen_att_name]
                priv_val = [1]
                unpriv_dict = [list(set(X_val.values[:, sa])) for sa in sen_att]
                for sa_list, pv in zip(unpriv_dict, priv_val):
                    sa_list.remove(pv)
                fairness_explainer_original = FairnessExplainer(
                        model=model, 
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
                non_zero_count_minority = np.sum(fairness_shapley_minority_value > self.threshold)
                print(f"在X_train_minority中shapely value中大于{self.threshold}的值的个数有: {non_zero_count_minority}")
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
                non_zero_count_majority =np.sum(fairness_shapley_majority_value > self.threshold)

                print(f"在X_train_majority中shapely value中大于{self.threshold}的值的个数有: {non_zero_count_majority}")
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
                    varphi = np.where(fairness_shapley_value > self.threshold, fairness_shapley_value, 0)
                elif self.fairshap_base == 'DP' or self.fairshap_base == 'EO' or self.fairshap_base == 'PQP':
                    varphi = np.where(fairness_shapley_value > self.threshold, fairness_shapley_value, 0)
                    varphi = np.abs(varphi)
                    pass
                q = np.vstack((q_minority,q_majority))   # q_minority_label0 + q_minority_label1 + q_majority_label0 + q_majority_label1
                X_change = pd.concat([X_change_minority_label0, X_change_minority_label1, X_change_majority_label0, X_change_majority_label1], axis=0)
                x = X_change.copy()
                y = pd.concat([y_train_minority_label0, y_train_minority_label1, y_train_majority_label0, y_train_majority_label1], axis=0)
                non_zero_count = non_zero_count_majority + non_zero_count_minority

                # print('6. 计算original model在X_test上的accuracy, DR, DP, EO, PP')
                # y_pred = self.model.predict(self.X_test)
                # original_accuracy = accuracy_score(self.y_test, y_pred)
                # original_DR = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, self.model)

                # priv_idx = self.X_test[self.sensitive_attri].to_numpy().astype(bool)
                # g1_Cm, g0_Cm = marginalised_np_mat(y=self.y_test, y_hat=y_pred, pos_label=1, priv_idx=priv_idx)
                # original_DP = grp1_DP(g1_Cm, g0_Cm)[0]
                # original_EO = grp2_EO(g1_Cm, g0_Cm)[0]
                # original_PQP = grp3_PQP(g1_Cm, g0_Cm)[0]

                # 计算指标
                # original_recall, original_precision, original_sufficiency = calculate_metrics(self.y_test, y_pred, pos=1)
                print(f'7. 开始整理minority部分的修改和majority部分的修改并且合并新数据,共修改{non_zero_count}个数据点, 使用new training set训练新模型')

                # Step 1: 将 varphi 的值和位置展开为一维
                flat_varphi = [(value, row, col) for row, row_vals in enumerate(varphi)
                            for col, value in enumerate(row_vals)]
                # Step 2: 按值降序排序
                flat_varphi_sorted = sorted(flat_varphi, key=lambda x: x[0], reverse=True)
                # Step 3: 挑出前 action_number 个数的位置
                top_positions = flat_varphi_sorted[:non_zero_count-1]
                print(f"top_positions: {non_zero_count}")
                # Step 4: 替换 X 中前三列的值为 S 中对应位置的值
                for value, row_idx, col_idx in top_positions:
                    X_change.iloc[row_idx, col_idx] = q[row_idx, col_idx]
  
                wasserstein_scores = compute_wasserstein_fidelity(X_change.values, x.values)

                # 比较这两个变量有多少不同
                diff_count = np.sum(X_change.values != x.values)
                print(f"diff_count: {diff_count}")

                # Step 6: Train the new model            
                model_new = XGBClassifier()
                model_new.fit(X_change, y)

                # accuracy, dr, dp, eo, pqp = self._run_evaluation_np(model, X_val_repair, y_val)
                accuracy, dr, dp, eo, pqp = self._run_evaluation_pd(model_new, X_val, y_val)
                mmd_score = compute_mmd_distance(X_change.values, x.values, gamma=1.0 / X_train.shape[1])
                processed_accuracy.append(accuracy)
                processed_dr.append(dr)
                processed_dp.append(dp)
                processed_eo.append(eo)
                processed_pqp.append(pqp)
                modification_num.append(diff_count)
                wasserstein_distance.append(wasserstein_scores)
                mmd_distance.append(mmd_score)

            save_dir = "saved_results/sota_results"
            os.makedirs(save_dir, exist_ok=True)  # 自动创建目录（如果不存在）
            # Save PROCESSED results of current dataset in a csv file
            processed_stats = {
                'processed_accuracy': f"{np.mean(processed_accuracy):.4f} ± {np.std(processed_accuracy):.4f}",
                'processed_dr': f"{np.mean(processed_dr):.4f} ± {np.std(processed_dr):.4f}",
                'processed_dp': f"{np.mean(processed_dp):.4f} ± {np.std(processed_dp):.4f}",
                'processed_eo': f"{np.mean(processed_eo):.4f} ± {np.std(processed_eo):.4f}",
                'processed_pqp': f"{np.mean(processed_pqp):.4f} ± {np.std(processed_pqp):.4f}",
                'modification_num': f"{np.mean(modification_num):.4f} ± {np.std(modification_num):.4f}",
                'wasserstein_distance': f"{np.mean(wasserstein_distance):.4f} ± {np.std(wasserstein_distance):.4f}",
                'mmd_distance': f"{np.mean(mmd_distance):.4f} ± {np.std(mmd_distance):.4f}",
            }
            processed_results = pd.DataFrame(processed_stats, index=[self.dataset_name]).T

            processed_csv_file = os.path.join(save_dir, f"FairSHAP_{self.matching_method}_{self.threshold}_results.csv")
            if os.path.exists(processed_csv_file):
                existing_df = pd.read_csv(processed_csv_file, index_col=0)
                existing_df[self.dataset_name] = processed_results[self.dataset_name]
                existing_df.to_csv(processed_csv_file)
            else:
                processed_results.to_csv(processed_csv_file)
            print(f"✅ {self.dataset_name} 处理后结果已保存到 {processed_csv_file}")







    def _run_evaluation_pd(self, model, X_val:pd.DataFrame, y_val:pd.Series):
        sen_att_name = [self.sen_att_name]
        sen_att = [X_val.columns.get_loc(name) for name in sen_att_name]
        priv_val = [self.priv]
        unpriv_dict = [list(set(X_val.values[:, sa])) for sa in sen_att]
        for sa_list, pv in zip(unpriv_dict, priv_val):
            sa_list.remove(pv)
        fairness_explainer_original = FairnessExplainer(
                model=model, 
                sen_att=sen_att, 
                priv_val=priv_val, 
                unpriv_dict=unpriv_dict,
                fairshap_base='DR'
                )
        # calculate fairness value on val data(test data)
        y_pred = model.predict(X_val)
        original_accuracy = accuracy_score(y_val, y_pred)
        original_DR = fairness_value_function(sen_att, priv_val, unpriv_dict, X_val.values, model)

        priv_idx = X_val[self.sen_att_name].to_numpy().astype(bool)
        g1_Cm, g0_Cm = marginalised_np_mat(y=y_val, y_hat=y_pred, pos_label=1, priv_idx=priv_idx)
        original_DP = grp1_DP(g1_Cm, g0_Cm)[0]
        original_EO = grp2_EO(g1_Cm, g0_Cm)[0]
        original_PQP = grp3_PQP(g1_Cm, g0_Cm)[0]
        return original_accuracy, original_DR, original_DP, original_EO, original_PQP
    
    def _run_evaluation_np(self, model, X_val: np.ndarray, y_val: np.ndarray):
        """
        Fairness evaluation function adapted for NumPy arrays.

        Args:
            model: The trained model used for prediction.
            X_val (np.ndarray): The validation feature set (NumPy array).
            y_val (np.ndarray): The validation labels (NumPy array).

        Returns:
            original_accuracy (float): Accuracy of the model on validation data.
            original_DR (float): Fairness metric computed using fairness_value_function.
            original_DP (float): Disparity in positive prediction rates.
            original_EO (float): Disparity in true positive rates.
            original_PQP (float): Disparity in predictive parity.
        """
        sen_att_name = [self.sen_att_name]  # Sensitive attribute name (list format)
        sen_att = self.sen_att  # Sensitive attribute index (assumed to be column index)
        sen_att_index = self.sen_att_index  # Sensitive attribute index (assumed to be column index)
        priv_val = [1]  # Privileged group value (assumed to be 1)

        # Extract unprivileged group values from the sensitive attribute column
        unpriv_dict = [list(set(X_val[:, sa])) for sa in sen_att]  # Get unique values per sensitive attribute
        for sa_list, pv in zip(unpriv_dict, priv_val):
            if pv in sa_list:
                sa_list.remove(pv)  # Remove the privileged value, keeping only unprivileged ones

        # Initialize the fairness explainer
        fairness_explainer_original = FairnessExplainer(
            model=model, 
            sen_att=sen_att, 
            priv_val=priv_val, 
            unpriv_dict=unpriv_dict,
            fairshap_base='DR'
        )

        # Compute model predictions
        y_pred = model.predict(X_val)
        original_accuracy = accuracy_score(y_val, y_pred)  # Compute accuracy

        # Compute fairness metrics
        original_DR = fairness_value_function(sen_att, priv_val, unpriv_dict, X_val, model)

        # Get boolean index for privileged group (True = privileged, False = unprivileged)
        priv_idx = X_val[:, sen_att_index].astype(bool)  # Extract the sensitive attribute column

        # Compute group-based fairness metrics
        g1_Cm, g0_Cm = marginalised_np_mat(y=y_val, y_hat=y_pred, pos_label=1, priv_idx=priv_idx)
        original_DP = grp1_DP(g1_Cm, g0_Cm)[0]  # Disparity in positive prediction rates
        original_EO = grp2_EO(g1_Cm, g0_Cm)[0]  # Disparity in true positive rates
        original_PQP = grp3_PQP(g1_Cm, g0_Cm)[0]  # Predictive parity difference

        return original_accuracy, original_DR, original_DP, original_EO, original_PQP

    def _split_into_majority_minority_label0_label1(self, X_train:pd.DataFrame, y_train:pd.Series):
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
        group_division = X_train[self.sen_att_name].value_counts()
        '''把X_train分成majority和minority两个部分'''
        if group_division[0] > group_division[1]:  #
            majority = X_train[self.sen_att_name] == 0
            X_train_majority = X_train[majority]
            y_train_majority = y_train[majority]
            minority = X_train[self.sen_att_name] == 1
            X_train_minority = X_train[minority]
            y_train_minority = y_train[minority]

        else:
            majority = X_train[self.sen_att_name] == 1
            X_train_majority = X_train[majority]
            y_train_majority = y_train[majority]
            minority = X_train[self.sen_att_name] == 0
            X_train_minority = X_train[minority]
            y_train_minority = y_train[minority]

        y_train_majority_label1 = y_train_majority[y_train_majority == 1]
        y_train_majority_label0 = y_train_majority[y_train_majority == 0]
        y_train_minority_label1 = y_train_minority[y_train_minority == 1]
        y_train_minority_label0 = y_train_minority[y_train_minority == 0]

        X_train_majority_label0 = X_train_majority.loc[y_train_majority_label0.index]
        X_train_majority_label1 = X_train_majority.loc[y_train_majority_label1.index]
        X_train_minority_label0 = X_train_minority.loc[y_train_minority_label0.index]
        X_train_minority_label1 = X_train_minority.loc[y_train_minority_label1.index]

        return X_train_majority_label0, y_train_majority_label0, X_train_majority_label1, y_train_majority_label1, X_train_minority_label0, y_train_minority_label0, X_train_minority_label1, y_train_minority_label1 




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



def compute_wasserstein_fidelity(original_data: np.ndarray, augmented_data: np.ndarray):
    assert original_data.shape == augmented_data.shape, "数据形状必须匹配"
    
    num_features = original_data.shape[1]
    wasserstein_scores = [
        wasserstein_distance(original_data[:, i], augmented_data[:, i]) 
        for i in range(num_features)
    ]
    
    return np.array(wasserstein_scores)


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

def compute_mmd_distance(X1, X2, gamma=1.0):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two datasets using the RBF (Gaussian) kernel.

    Args:
        X1: numpy array of shape (n1, d) - samples from distribution P
        X2: numpy array of shape (n2, d) - samples from distribution Q
        gamma: float - parameter for the RBF kernel (1 / (2 * sigma^2))

    Returns:
        mmd_score: float - the estimated MMD distance between X1 and X2
    """
    n1 = X1.shape[0]
    n2 = X2.shape[0]

    # Compute RBF kernel matrices
    K_xx = rbf_kernel(X1, X1, gamma=gamma)  # Kernel within X1
    K_yy = rbf_kernel(X2, X2, gamma=gamma)  # Kernel within X2
    K_xy = rbf_kernel(X1, X2, gamma=gamma)  # Kernel between X1 and X2

    # Compute MMD^2 using the unbiased estimator (excluding diagonal elements)
    mmd_xx = (np.sum(K_xx) - np.trace(K_xx)) / (n1 * (n1 - 1))
    mmd_yy = (np.sum(K_yy) - np.trace(K_yy)) / (n2 * (n2 - 1))
    mmd_xy = np.sum(K_xy) / (n1 * n2)

    # Final MMD score
    mmd_score = mmd_xx + mmd_yy - 2 * mmd_xy
    return np.sqrt(mmd_score)  # Return the square root for interpretability