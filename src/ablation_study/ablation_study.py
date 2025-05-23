import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from fairlearn.preprocessing import CorrelationRemover
from sklearn.model_selection import KFold
import aif360

from src.matching.ot_matcher import OptimalTransportPolicy
from src.matching.nn_matcher import NearestNeighborDataMatcher
from src.composition.data_composer import DataComposer
from src.attribution import FairnessExplainer
from src.data.unified_dataloader import load_dataset
from src.attribution.oracle_metric import perturb_numpy_ver
import os
from scipy.stats import wasserstein_distance
import random

class AblationStudy:
    def __init__(self, model):
        self.model = model


    def run_ablation_study_1(self):
        """
        Random Background Data Modification: Randomly selecting the corresponding number of values from the background data to modify the original dataset.

        During Ablation Study 1, we found that many data points were replaced with identical values. To mitigate this issue, we set a relatively high self.change_num.
        However, in the final implementation, we added a conditional check to skip replacement if the new value is the same as the original.
        """

        for i in ['german_credit', 'compas', 'compas4race', 'adult', 'default_credit', 'census_income_kdd']:
            self.dataset_name = i
            print("-------------------------------------")
            print(f"---!! on {self.dataset_name} dataset !!---")
            print("-------------------------------------")
            if self.dataset_name == 'german_credit':
                self.sen_att_name = 'sex'
                self.target_name = 'Risk'
                self.change_num = 3000
                self.priv = 0
            elif self.dataset_name == 'adult':
                self.sen_att_name = 'sex'
                self.target_name = 'income'
                self.change_num = 90000
                self.priv = 1
            elif self.dataset_name == 'compas':
                self.sen_att_name = 'sex'
                self.target_name = 'two_year_recid'
                self.change_num = 8000
                self.priv = 1
            elif self.dataset_name == 'compas4race':
                self.sen_att_name = 'race'
                self.target_name = 'two_year_recid'
                self.change_num = 8000
                self.priv = 1
            elif self.dataset_name == 'census_income_kdd':
                self.sen_att_name = 'sex'
                self.target_name = 'class'
                self.change_num = 200000
                self.priv = 0
            elif self.dataset_name == 'default_credit':
                self.sen_att_name = 'sex'
                self.target_name = 'default_payment_next_month'
                self.change_num = 5000
                self.priv = 0

            _, processed_data = load_dataset(self.dataset_name)

            if self.dataset_name == 'census_income_kdd':
                processed_data = processed_data.sample(frac=0.1, random_state=25)

            kf = KFold(n_splits=5, shuffle=True, random_state=1)  # 5-fold cross-validation
            processed_accuracy = []
            processed_dr = []
            processed_dp = []
            processed_eo = []
            processed_pqp = []
            modification_num = []
            wasserstein_distance = []

            for j, (train_index, val_index) in enumerate(kf.split(processed_data)):
                print("-------------------------------------")
                print(f"-------------{j}th fold----------------")
                print("-------------------------------------")

                train_data, val_data = processed_data.iloc[train_index], processed_data.iloc[val_index]
                X_train, y_train = train_data.drop(self.target_name, axis=1), train_data[self.target_name]
                X_val, y_val = val_data.drop(self.target_name, axis=1), val_data[self.target_name]

                model = XGBClassifier()
                model.fit(X_train, y_train)

                # Split into majority and minority groups based on label
                X_train_majority_label0, y_train_majority_label0, X_train_majority_label1, y_train_majority_label1, X_train_minority_label0, y_train_minority_label0, X_train_minority_label1, y_train_minority_label1 = self._split_into_majority_minority_label0_label1(X_train=X_train, y_train=y_train)

                # Perform nearest neighbor matching for each group
                matching_majority_label1 = NearestNeighborDataMatcher(X_labeled=X_train_majority_label1, X_unlabeled=X_train_minority_label1).match(n_neighbors=1)
                matching_majority_label0 = NearestNeighborDataMatcher(X_labeled=X_train_majority_label0, X_unlabeled=X_train_minority_label0).match(n_neighbors=1)
                matching_minority_label1 = NearestNeighborDataMatcher(X_labeled=X_train_minority_label1, X_unlabeled=X_train_majority_label1).match(n_neighbors=1)
                matching_minority_label0 = NearestNeighborDataMatcher(X_labeled=X_train_minority_label0, X_unlabeled=X_train_majority_label0).match(n_neighbors=1)

                # Concatenate all groups into one DataFrame
                X_change = pd.concat([X_train_majority_label1, X_train_majority_label0, X_train_minority_label1, X_train_minority_label0], axis=0)
                x = X_change.copy()
                y = pd.concat([y_train_majority_label1, y_train_majority_label0, y_train_minority_label1, y_train_minority_label0], axis=0)

                # Generate counterfactual data using DataComposer
                q_majority_label1 = DataComposer(
                    x_counterfactual=X_train_minority_label1.values,
                    joint_prob=matching_majority_label1,
                    method="max").calculate_q()

                q_majority_label0 = DataComposer(
                    x_counterfactual=X_train_minority_label0.values,
                    joint_prob=matching_majority_label0,
                    method="max").calculate_q()

                q_minority_label1 = DataComposer(
                    x_counterfactual=X_train_majority_label1.values,
                    joint_prob=matching_minority_label1,
                    method="max").calculate_q()

                q_minority_label0 = DataComposer(
                    x_counterfactual=X_train_majority_label0.values,
                    joint_prob=matching_minority_label0,
                    method="max").calculate_q()

                X_base = np.vstack((q_majority_label1, q_majority_label0, q_minority_label1, q_minority_label0))

                # 4. Select self.change_num positions in X_change and replace them with values from X_base
                print(f'Selecting {self.change_num} positions in X_change and replacing them with values from X_base')

                num_rows, num_cols = X_change.shape
                random_row_indices = np.random.choice(num_rows, self.change_num, replace=True)
                random_col_indices = np.random.choice(num_cols, self.change_num, replace=True)

                print(f'X_change shape: {X_change.shape}')
                d = np.sum(X_change.values != X_base)
                print(f'There are {d} different values between X_change and X_base')
                n = 0

                # Replace only when the new value is different from the original
                for row, col in zip(random_row_indices, random_col_indices):
                    if X_change.iloc[row, col] == X_base[row, col]:
                        n += 1
                    else:
                        X_change.iloc[row, col] = X_base[row, col]

                diff_count = np.sum(X_change.values != x.values)
                print(f'A total of {self.change_num} values are intended to be replaced')
                print(f'{n} values are the same as original and will not be replaced')
                print(f'Actually replaced {diff_count} values')

                # Set up fairness evaluation parameters
                sen_att_name = [self.sen_att_name]
                sen_att = [X_val.columns.get_loc(name) for name in sen_att_name]
                priv_val = [1]
                unpriv_dict = [list(set(X_val.values[:, sa])) for sa in sen_att]
                for sa_list, pv in zip(unpriv_dict, priv_val):
                    sa_list.remove(pv)

                # Train and evaluate the modified model
                model_new = XGBClassifier()
                model_new.fit(X_change, y)

                accuracy, dr, dp, eo, pqp = self._run_evaluation_pd(model_new, X_val, y_val)
                wasserstein_scores = compute_wasserstein_fidelity(X_change.values, x.values)

                processed_accuracy.append(accuracy)
                processed_dr.append(dr)
                processed_dp.append(dp)
                processed_eo.append(eo)
                processed_pqp.append(pqp)
                modification_num.append(diff_count)
                wasserstein_distance.append(wasserstein_scores)

            save_dir = "saved_results/ablation_study"
            os.makedirs(save_dir, exist_ok=True)

            # Save processed results for the current dataset
            processed_stats = {
                'processed_accuracy': f"{np.mean(processed_accuracy):.4f} ± {np.std(processed_accuracy):.4f}",
                'processed_dr': f"{np.mean(processed_dr):.4f} ± {np.std(processed_dr):.4f}",
                'processed_dp': f"{np.mean(processed_dp):.4f} ± {np.std(processed_dp):.4f}",
                'processed_eo': f"{np.mean(processed_eo):.4f} ± {np.std(processed_eo):.4f}",
                'processed_pqp': f"{np.mean(processed_pqp):.4f} ± {np.std(processed_pqp):.4f}",
                'modification_num': f"{np.mean(modification_num):.4f} ± {np.std(modification_num):.4f}",
                'wasserstein_distance': f"{np.mean(wasserstein_distance):.4f} ± {np.std(wasserstein_distance):.4f}",
            }

            processed_results = pd.DataFrame(processed_stats, index=[self.dataset_name]).T

            processed_csv_file = os.path.join(save_dir, "ablation_study_1.csv")
            if os.path.exists(processed_csv_file):
                existing_df = pd.read_csv(processed_csv_file, index_col=0)
                existing_df[self.dataset_name] = processed_results[self.dataset_name]
                existing_df.to_csv(processed_csv_file)
            else:
                processed_results.to_csv(processed_csv_file)

            print(f"✅ Results for {self.dataset_name} have been saved to {processed_csv_file}")

    def run_ablation_study_2(self):
        '''
        Sensitive Attribute Modification: Randomly selecting the corresponding number of values of sensitive attributes from the background data to modify the original dataset.
        '''
        for i in ['german_credit', 'compas', 'compas4race', 'adult', 'default_credit', 'census_income_kdd']:
            self.dataset_name = i
            print("-------------------------------------")
            print(f"---!! on {self.dataset_name} dataset !!---")
            print("-------------------------------------")
            if self.dataset_name == 'german_credit':
                self.sen_att_name = 'sex'
                self.target_name = 'Risk'
                self.change_num = 323
                self.priv = 0
            elif self.dataset_name == 'adult':
                self.sen_att_name = 'sex'
                self.target_name = 'income'
                self.change_num = 3361
                self.priv = 1
            elif self.dataset_name == 'compas':
                self.sen_att_name = 'sex'
                self.target_name = 'two_year_recid'
                self.change_num = 1175
            elif self.dataset_name == 'compas4race':
                self.sen_att_name = 'race'
                self.target_name = 'two_year_recid'
                self.change_num = 907
                self.priv = 1
            elif self.dataset_name == 'census_income_kdd':
                self.sen_att_name = 'sex'
                self.target_name = 'class'
                self.change_num = 3236
                self.priv = 0
            elif self.dataset_name == 'default_credit':
                self.sen_att_name = 'sex'
                self.target_name = 'default_payment_next_month'
                self.change_num = 888
                self.priv = 0
            _, processed_data = load_dataset(self.dataset_name)
            if self.dataset_name == 'census_income_kdd':
                processed_data = processed_data.sample(frac=0.1, random_state=25) 
            kf = KFold(n_splits=5, shuffle=True, random_state=1)  # 5-fold cross-validation
            processed_accuracy = []
            processed_dr = []
            processed_dp = []
            processed_eo = []
            processed_pqp = []
            modification_num = []
            wasserstein_distance = []
            for j, (train_index, val_index) in enumerate(kf.split(processed_data)):
                print("-------------------------------------")
                print(f"-------------{j}th fold----------------")
                print("-------------------------------------")
                train_data, val_data = processed_data.iloc[train_index], processed_data.iloc[val_index]
                X_train, y_train = train_data.drop(self.target_name, axis=1), train_data[self.target_name]
                X_val, y_val = val_data.drop(self.target_name, axis=1), val_data[self.target_name]
                model = XGBClassifier()
                model.fit(X_train, y_train)
                X_train_majority_label0, y_train_majority_label0, X_train_majority_label1, y_train_majority_label1, X_train_minority_label0, y_train_minority_label0, X_train_minority_label1, y_train_minority_label1 = self._split_into_majority_minority_label0_label1(X_train=X_train, y_train=y_train)
                matching_majority_label1 = NearestNeighborDataMatcher(X_labeled=X_train_majority_label1, X_unlabeled=X_train_minority_label1).match(n_neighbors=1)
                matching_majority_label0 = NearestNeighborDataMatcher(X_labeled=X_train_majority_label0, X_unlabeled=X_train_minority_label0).match(n_neighbors=1)
                matching_minority_label1 = NearestNeighborDataMatcher(X_labeled=X_train_minority_label1, X_unlabeled=X_train_majority_label1).match(n_neighbors=1)
                matching_minority_label0 = NearestNeighborDataMatcher(X_labeled=X_train_minority_label0, X_unlabeled=X_train_majority_label0).match(n_neighbors=1)
                X_change = pd.concat([X_train_majority_label1, X_train_majority_label0, X_train_minority_label1, X_train_minority_label0], axis=0)
                x = X_change.copy()
                # y = pd.concat([y_train_minority_label0, y_train_minority_label1, y_train_majority_label0, y_train_majority_label1], axis=0)
                y = pd.concat([y_train_majority_label1, y_train_majority_label0, y_train_minority_label1, y_train_minority_label0], axis=0)
                q_majority_label1 = DataComposer(
                                x_counterfactual=X_train_minority_label1.values,
                                joint_prob=matching_majority_label1,
                                method="max").calculate_q()
                q_majority_label0 = DataComposer(
                                x_counterfactual=X_train_minority_label0.values,
                                joint_prob=matching_majority_label0,
                                method="max").calculate_q()
                q_minority_label1 = DataComposer(
                                x_counterfactual=X_train_majority_label1.values, 
                                joint_prob=matching_minority_label1, 
                                method="max").calculate_q() 
                q_minority_label0 = DataComposer(
                                x_counterfactual=X_train_majority_label0.values, 
                                joint_prob=matching_minority_label0, 
                                method="max").calculate_q()
                X_base = np.vstack((q_majority_label1, q_majority_label0, q_minority_label1, q_minority_label0))
                # 4. In all values of X_change, select change_num positions and replace them with values from X_base
                print(f'Selecting {self.change_num} positions in the sensitive attribute of X_change and flipping the value from 1 to 0 or vice versa')
                # Ensure that self.sen_att_name is in X_change
                if self.sen_att_name not in X_change.columns:
                    raise ValueError(f"The sensitive attribute {self.sen_att_name} is not present in X_change")
                # Get indices of the sensitive attribute column
                sensitive_col_indices = X_change.index  # Get real indices of X_change
                # Generate random indices
                random_indices = np.random.choice(sensitive_col_indices, self.change_num, replace=False)  # Selected row indices
                # Flip values 1 ↔ 0
                X_change.loc[random_indices, self.sen_att_name] = 1 - X_change.loc[random_indices, self.sen_att_name]
                # Count how many values have been modified
                diff_count = np.sum(X_change[self.sen_att_name].values != x[self.sen_att_name].values)
                # Print modification info
                print(f"Successfully modified {diff_count} values of {self.sen_att_name}")
                # Initialize variables needed in fairness_measure
                sen_att_name = [self.sen_att_name]
                sen_att = [X_val.columns.get_loc(name) for name in sen_att_name]
                priv_val = [1]
                unpriv_dict = [list(set(X_val.values[:, sa])) for sa in sen_att]
                for sa_list, pv in zip(unpriv_dict, priv_val):
                    sa_list.remove(pv)
                # Step 6: Train and evaluate model            
                model_new = XGBClassifier()
                model_new.fit(X_change, y)
                # accuracy, dr, dp, eo, pqp = self._run_evaluation_np(model, X_val_repair, y_val)
                accuracy, dr, dp, eo, pqp = self._run_evaluation_pd(model_new, X_val, y_val)
                wasserstein_scores = compute_wasserstein_fidelity(X_change.values, x.values)
                processed_accuracy.append(accuracy)
                processed_dr.append(dr)
                processed_dp.append(dp)
                processed_eo.append(eo)
                processed_pqp.append(pqp)
                modification_num.append(diff_count)
                wasserstein_distance.append(wasserstein_scores)
            save_dir = "saved_results/ablation_study"
            os.makedirs(save_dir, exist_ok=True)  # Automatically create directory if it doesn't exist
            # Save PROCESSED results of current dataset in a csv file
            processed_stats = {
                'processed_accuracy': f"{np.mean(processed_accuracy):.4f} ± {np.std(processed_accuracy):.4f}",
                'processed_dr': f"{np.mean(processed_dr):.4f} ± {np.std(processed_dr):.4f}",
                'processed_dp': f"{np.mean(processed_dp):.4f} ± {np.std(processed_dp):.4f}",
                'processed_eo': f"{np.mean(processed_eo):.4f} ± {np.std(processed_eo):.4f}",
                'processed_pqp': f"{np.mean(processed_pqp):.4f} ± {np.std(processed_pqp):.4f}",
                'modification_num': f"{np.mean(modification_num):.4f} ± {np.std(modification_num):.4f}",
                'wasserstein_distance': f"{np.mean(wasserstein_distance):.4f} ± {np.std(wasserstein_distance):.4f}",
            }
            processed_results = pd.DataFrame(processed_stats, index=[self.dataset_name]).T
            processed_csv_file = os.path.join(save_dir, "ablation_study_2.csv")  # Storage path
            if os.path.exists(processed_csv_file):
                existing_df = pd.read_csv(processed_csv_file, index_col=0)
                existing_df[self.dataset_name] = processed_results[self.dataset_name]
                existing_df.to_csv(processed_csv_file)
            else:
                processed_results.to_csv(processed_csv_file)
            print(f"✅ Results for {self.dataset_name} have been saved to {processed_csv_file}")

    def run_ablation_study_3(self):
        '''
        Random Background Data Modification: Randomly selecting the corresponding number of values from the background data to modify the original dataset.
        Because during ablation study 1, we found that many points were replaced with the same values, we set self.change_num higher. However, in the code, we added a conditional statement to ensure replacement only occurs if the new value differs from the original.
        '''
        for i in ['german_credit', 'compas', 'compas4race', 'adult', 'default_credit', 'census_income_kdd']:
            self.dataset_name = i
            print("-------------------------------------")
            print(f"---!! on {self.dataset_name} dataset !!---")
            print("-------------------------------------")
            if self.dataset_name == 'german_credit':
                self.sen_att_name = 'sex'
                self.target_name = 'Risk'
                self.change_num = 1000
                self.priv = 0
            elif self.dataset_name == 'adult':
                self.sen_att_name = 'sex'
                self.target_name = 'income'
                self.change_num = 30000
                self.priv = 1
            elif self.dataset_name == 'compas':
                self.sen_att_name = 'sex'
                self.target_name = 'two_year_recid'
                self.change_num = 4000
                self.priv = 1
            elif self.dataset_name == 'compas4race':
                self.sen_att_name = 'race'
                self.target_name = 'two_year_recid'
                self.change_num = 3000
                self.priv = 1
            elif self.dataset_name == 'census_income_kdd':
                self.sen_att_name = 'sex'
                self.target_name = 'class'
                self.change_num = 50000
                self.priv = 0
            elif self.dataset_name == 'default_credit':
                self.sen_att_name = 'sex'
                self.target_name = 'default_payment_next_month'
                self.change_num = 2000
                self.priv = 0
            _, processed_data = load_dataset(self.dataset_name)
            if self.dataset_name == 'census_income_kdd':
                processed_data = processed_data.sample(frac=0.1, random_state=25) 
            kf = KFold(n_splits=5, shuffle=True, random_state=1)  # 5-fold cross-validation
            processed_accuracy = []
            processed_dr = []
            processed_dp = []
            processed_eo = []
            processed_pqp = []
            modification_num = []
            wasserstein_distance = []
            for j, (train_index, val_index) in enumerate(kf.split(processed_data)):
                print("-------------------------------------")
                print(f"-------------{j}th fold----------------")
                print("-------------------------------------")
                train_data, val_data = processed_data.iloc[train_index], processed_data.iloc[val_index]
                X_train, y_train = train_data.drop(self.target_name, axis=1), train_data[self.target_name]
                X_val, y_val = val_data.drop(self.target_name, axis=1), val_data[self.target_name]
                model = XGBClassifier()
                model.fit(X_train, y_train)
                x = X_train.copy()
                X_change = X_train.copy()
                rows, cols = x.shape
                # Determine the number of points to modify
                change_num = self.change_num
                # Create a list of all possible points (row, col)
                all_points = [(i, j) for i in range(rows) for j in range(cols)]
                # Randomly select points to modify, ensuring they are unique
                selected_points = random.sample(all_points, change_num)
                # Modify each selected point
                for row, col in selected_points:
                    # Get all row indices except the current one
                    other_rows = [i for i in range(rows) if i != row]
                    # Randomly select a row different from the current one
                    random_row = random.choice(other_rows)
                    # Replace the value at the selected point with a value from the same column but different row
                    X_change.iloc[row, col] = x.iloc[random_row, col]
                diff_count = np.sum(X_change.values != x.values)
                print(f'Expected to replace {self.change_num} values')
                print(f'Actually replaced {diff_count} values')
                # Initialize variables needed in fairness_measure
                sen_att_name = [self.sen_att_name]
                sen_att = [X_val.columns.get_loc(name) for name in sen_att_name]
                priv_val = [1]
                unpriv_dict = [list(set(X_val.values[:, sa])) for sa in sen_att]
                for sa_list, pv in zip(unpriv_dict, priv_val):
                    sa_list.remove(pv)
                # Step 6: Train and evaluate model            
                model_new = XGBClassifier()
                model_new.fit(X_change, y_train)
                # accuracy, dr, dp, eo, pqp = self._run_evaluation_np(model, X_val_repair, y_val)
                accuracy, dr, dp, eo, pqp = self._run_evaluation_pd(model_new, X_val, y_val)
                wasserstein_scores = compute_wasserstein_fidelity(X_change.values, x.values)
                processed_accuracy.append(accuracy)
                processed_dr.append(dr)
                processed_dp.append(dp)
                processed_eo.append(eo)
                processed_pqp.append(pqp)
                modification_num.append(diff_count)
                wasserstein_distance.append(wasserstein_scores)

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
            '''split X_train into two groups -- majority and minority'''
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
    assert original_data.shape == augmented_data.shape, "must have the same shape"
    
    num_features = original_data.shape[1]
    wasserstein_scores = [
        wasserstein_distance(original_data[:, i], augmented_data[:, i]) 
        for i in range(num_features)
    ]
    
    return np.array(wasserstein_scores)

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


def calculate_metrics(y_test, y_pred, pos=1):
    tp, fp, fn, tn = contingency_tab_bi(y_test, y_pred, pos)
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    sufficiency = tn / (tn + fp) if (tn + fp) != 0 else 0
    return recall, precision, sufficiency
