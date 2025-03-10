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
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import OptimPreproc

from src.attribution import FairnessExplainer
from src.data.unified_dataloader import load_dataset
from src.attribution.oracle_metric import perturb_numpy_ver
from fairness_measures import marginalised_np_mat, grp1_DP, grp2_EO, grp3_PQP
import os


class BenchMarkPreprocessingMethods:
    def __init__(self,
                # dataset_name: str,  # 'german_Credit', 'compas', 'compas4race', 'adult', 'default_credit', 'census_income_kdd'
                sota_method: str,  # 'disparate_impact', 'correlation_removal', 'reweighing', 'fairUS', 'optimized_preprocessing'
    ):
        # self.dataset_name = dataset_name
        self.sota_method = sota_method
        # if self.dataset_name == 'german_credit':
        #     self.sen_att_name = ['sex']
        #     self.gap = 1
        # elif self.dataset_name == 'adult':
        #     self.sen_att_name = ['sex']
        #     self.gap = 1
        # elif self.dataset_name == 'compas':
        #     self.sen_att_name = ['sex']
        #     self.gap = 1
        # elif self.dataset_name == 'compas4race':
        #     self.sen_att_name = ['race']
        #     self.gap = 1
        # elif self.dataset_name == 'census_income':
        #     self.sen_att_name = ['sex']
        #     self.gap = 1
        # elif self.dataset_name == 'default_credit':
        #     self.sen_att_name = ['sex']
        #     self.gap = 1
        # else :
        #     raise ValueError('The dataset name is not supported')
    def run_and_save_results(self, save_origin=True):
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

            for j, (train_index, val_index) in enumerate(kf.split(processed_data)):
                print("-------------------------------------")
                print(f"-------------{j}th fold----------------")
                print("-------------------------------------")
                train_data, val_data = processed_data.iloc[train_index], processed_data.iloc[val_index]
                X_train, y_train = train_data.drop(self.target_name, axis=1), train_data[self.target_name]
                X_val, y_val = val_data.drop(self.target_name, axis=1), val_data[self.target_name]
                # Baseline model

                self.sen_att = [X_train.columns.get_loc(name) for name in [self.sen_att_name]]
                self.sen_att_index = X_train.columns.get_loc(self.sen_att_name)
                if save_origin:
                    model = XGBClassifier()
                    model.fit(X_train, y_train)
                    accuracy, dr, dp, eo, pqp = self._run_evaluation_pd(model, X_val, y_val)
                    original_accuracy.append(accuracy)
                    original_dr.append(dr)
                    original_dp.append(dp)
                    original_eo.append(eo)
                    original_pqp.append(pqp)

                # Mitigate bias
                if self.sota_method == 'disparate_impact':
                    X_train_repair, X_val_repair = self._disparate_impact_remover(repair_level=1.0, sensitive_attribute=self.sen_att_name, X_train=X_train, X_val=X_val)
                    model = XGBClassifier()
                    model.fit(X_train_repair, y_train)
                    
                    accuracy, dr, dp, eo, pqp = self._run_evaluation_np(model, X_val_repair, y_val)
                    processed_accuracy.append(accuracy)
                    processed_dr.append(dr)
                    processed_dp.append(dp)
                    processed_eo.append(eo)
                    processed_pqp.append(pqp)

                elif self.sota_method == 'correlation_removal':
                    X_train_repair = self._correlation_removal(X_train, remove_ratio=1)
                    X_val_repair = self._correlation_removal(X_val, remove_ratio=1)
                    model = XGBClassifier()
                    model.fit(X_train_repair, y_train)
                    accuracy, dr, dp, eo, pqp = self._run_evaluation_np(model, X_val_repair, y_val)
                    processed_accuracy.append(accuracy)
                    processed_dr.append(dr)
                    processed_dp.append(dp)
                    processed_eo.append(eo)
                    processed_pqp.append(pqp)
                    
                elif self.sota_method == 'reweighing':
                    model = XGBClassifier()
                    instance_weights = self._reweighing(X_train, y_train)
                    model.fit(X_train, y_train, sample_weight=instance_weights)
                    accuracy, dr, dp, eo, pqp = self._run_evaluation_pd(model, X_val, y_val)
                    processed_accuracy.append(accuracy)
                    processed_dr.append(dr)
                    processed_dp.append(dp)
                    processed_eo.append(eo)
                    processed_pqp.append(pqp)
                elif self.sota_method == 'optimized_preprocessing':
                    self.optimized_preprocessing(X_train, y_train, X_val, y_val)                 

                elif self.sota_method == 'fairUS':
                    self.fairUS(X_train, y_train, X_val, y_val)

                else:
                    raise ValueError('The SOTA method is not supported')

        
            save_dir = "saved_results/sota_results"
            os.makedirs(save_dir, exist_ok=True)  # 自动创建目录（如果不存在）
            
            if save_origin:
                # Save ORIGINAL results of current dataset in a csv file  
                original_stats = {
                    'original_accuracy': f"{np.mean(original_accuracy):.4f} ± {np.std(original_accuracy):.4f}",
                    'original_dr': f"{np.mean(original_dr):.4f} ± {np.std(original_dr):.4f}",
                    'original_dp': f"{np.mean(original_dp):.4f} ± {np.std(original_dp):.4f}",
                    'original_eo': f"{np.mean(original_eo):.4f} ± {np.std(original_eo):.4f}",
                    'original_pqp': f"{np.mean(original_pqp):.4f} ± {np.std(original_pqp):.4f}",
                }
                original_results = pd.DataFrame(original_stats, index=[self.dataset_name]).T

                csv_file = os.path.join(save_dir, "original_results.csv")  # 存储路径
                if os.path.exists(csv_file):
                    existing_df = pd.read_csv(csv_file, index_col=0)
                    existing_df[self.dataset_name] = original_results[self.dataset_name]
                    existing_df.to_csv(csv_file)
                else:
                    original_results.to_csv(csv_file)
                print(f"✅ {self.dataset_name} 原始结果已保存到 {csv_file}")

            # Save PROCESSED results of current dataset in a csv file
            processed_stats = {
                'processed_accuracy': f"{np.mean(processed_accuracy):.4f} ± {np.std(processed_accuracy):.4f}",
                'processed_dr': f"{np.mean(processed_dr):.4f} ± {np.std(processed_dr):.4f}",
                'processed_dp': f"{np.mean(processed_dp):.4f} ± {np.std(processed_dp):.4f}",
                'processed_eo': f"{np.mean(processed_eo):.4f} ± {np.std(processed_eo):.4f}",
                'processed_pqp': f"{np.mean(processed_pqp):.4f} ± {np.std(processed_pqp):.4f}",
            }
            processed_results = pd.DataFrame(processed_stats, index=[self.dataset_name]).T

            processed_csv_file = os.path.join(save_dir, f"{self.sota_method}_results.csv")  # 存储路径
            if os.path.exists(processed_csv_file):
                existing_df = pd.read_csv(processed_csv_file, index_col=0)
                existing_df[self.dataset_name] = processed_results[self.dataset_name]
                existing_df.to_csv(processed_csv_file)
            else:
                processed_results.to_csv(processed_csv_file)
            print(f"✅ {self.dataset_name} 处理后结果已保存到 {processed_csv_file}")




    def _disparate_impact_remover(self, repair_level, sensitive_attribute, X_train:pd.DataFrame, X_val:pd.DataFrame):
        DisparateImpactRemover(repair_level=1.0, sensitive_attribute=self.sen_att_name)
        from BlackBoxAuditing.repairers.GeneralRepairer import Repairer
        self.Repairer = Repairer
        features_train = X_train.values.tolist()
        features_val = X_val.values.tolist()
        index = X_train.columns.tolist().index(self.sen_att_name)
        repairer = self.Repairer(features_train, index, repair_level, False)

        repaired_features_train = repairer.repair(features_train)
        repaired_features_val = repairer.repair(features_val)
        repaired_features_train = np.array(repaired_features_train, dtype=np.float64)
        repaired_features_val = np.array(repaired_features_val, dtype=np.float64)

        repaired_features_train[:, index] = X_train.values[:, index]
        repaired_features_val[:, index] = X_val.values[:, index]
        return repaired_features_train, repaired_features_val
    
    def _correlation_removal(self, X:pd.DataFrame, remove_ratio=1):
        cr = CorrelationRemover(sensitive_feature_ids=[self.sen_att_name], alpha=remove_ratio)
        cr.fit(X)
        # CorrelationRemover(sensitive_feature_ids=sensitive_feature)
        X_transform_without_sen_att = cr.transform(X)
        sensitive_column = X['sex'].copy()
        X_transform_with_sensitive = np.insert(X_transform_without_sen_att, self.sen_att_index, sensitive_column, axis=1)
        return X_transform_with_sensitive

    def _reweighing(self, X_train, y_train):
        # self.w_p_fav = 1.
        # self.w_p_unfav = 1.
        # self.w_up_fav = 1.
        # self.w_up_unfav = 1.
        df = X_train.copy()
        df[self.target_name] = y_train

        priv_cond = X_train[self.sen_att_name] == self.priv
        unpriv_cond = X_train[self.sen_att_name] != self.priv
        fav_cond = y_train == 1
        unfav_cond = y_train == 0

        n = len(X_train)
        n_p = np.sum(priv_cond)
        n_up = np.sum(unpriv_cond)
        n_fav = np.sum(fav_cond)
        n_unfav = np.sum(unfav_cond)
        
        # 计算 (group, label) 组合的样本数
        n_p_fav = np.sum(priv_cond & fav_cond)
        n_p_unfav = np.sum(priv_cond & unfav_cond)
        n_up_fav = np.sum(unpriv_cond & fav_cond)
        n_up_unfav = np.sum(unpriv_cond & unfav_cond)

        # 计算 reweighing 权重
        w_p_fav = (n_fav * n_p) / (n * n_p_fav) if n_p_fav > 0 else 1.
        w_p_unfav = (n_unfav * n_p) / (n * n_p_unfav) if n_p_unfav > 0 else 1.
        w_up_fav = (n_fav * n_up) / (n * n_up_fav) if n_up_fav > 0 else 1.
        w_up_unfav = (n_unfav * n_up) / (n * n_up_unfav) if n_up_unfav > 0 else 1.

        instance_weights = np.ones(len(df))
        instance_weights[priv_cond & fav_cond] *= w_p_fav
        instance_weights[priv_cond & unfav_cond] *= w_p_unfav
        instance_weights[unpriv_cond & fav_cond] *= w_up_fav
        instance_weights[unpriv_cond & unfav_cond] *= w_up_unfav

        return instance_weights





    def _run_evaluation_pd(self, model, X_val:pd.DataFrame, y_val:pd.Series):
        sen_att_name = [self.sen_att_name]
        sen_att = self.sen_att
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
