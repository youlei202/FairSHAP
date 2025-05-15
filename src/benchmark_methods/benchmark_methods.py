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
from fairness_related.fairness_measures import marginalised_np_mat, grp1_DP, grp2_EO, grp3_PQP
import os
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel

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
        Run the chosen SOTA method on all datasets and save the results in a CSV file.
        '''
        for i in ['german_credit', 'compas', 'compas4race', 'adult', 'default_credit', 'census_income_kdd']:
        # for i in ['recruit']:
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
                self.priv = 1
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
            elif self.dataset_name == 'recruit':
                self.sen_att_name = 'sex'
                self.target_name = 'decision'
                self.priv = 1

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
                    X_train_repair, X_val_repair = self._disparate_impact_remover(repair_level=1, X_train=X_train, X_val=X_val)
                    wasserstein_scores = compute_wasserstein_fidelity(X_train.values, X_train_repair)
                    mmd_score = compute_mmd_distance(X_train.values, X_train_repair, gamma=1.0 / X_train.shape[1])
                    model = XGBClassifier()
                    model.fit(X_train_repair, y_train)
                    accuracy, dr, dp, eo, pqp = self._run_evaluation_np(model, X_val_repair, y_val)
                    diff_count = np.sum(X_train.values != X_train_repair)
                    processed_accuracy.append(accuracy)
                    processed_dr.append(dr)
                    processed_dp.append(dp)
                    processed_eo.append(eo)
                    processed_pqp.append(pqp)
                    modification_num.append(diff_count)
                    wasserstein_distance.append(wasserstein_scores)
                    mmd_distance.append(mmd_score)
                    print(f"diff_count: {diff_count}")

                elif self.sota_method == 'correlation_removal':
                    X_train_repair, X_val_repair = self._correlation_removal(X_train, X_val, repair_level=1) 
                    mmd_score = compute_mmd_distance(X_train.values, X_train_repair, gamma=1.0 / X_train.shape[1])
                    wasserstein_scores = compute_wasserstein_fidelity(X_train.values, X_train_repair)
                    model = XGBClassifier()
                    model.fit(X_train_repair, y_train)
                    accuracy, dr, dp, eo, pqp = self._run_evaluation_np(model, X_val_repair, y_val)
                    diff_count = np.sum(X_train.values != X_train_repair)
                    processed_accuracy.append(accuracy)
                    processed_dr.append(dr)
                    processed_dp.append(dp)
                    processed_eo.append(eo)
                    processed_pqp.append(pqp)
                    modification_num.append(diff_count)
                    wasserstein_distance.append(wasserstein_scores)
                    mmd_distance.append(mmd_score)
                    print(f"diff_count: {diff_count}")

                elif self.sota_method == 'reweighing':
                    model = XGBClassifier()
                    instance_weights = self._reweighing(X_train, y_train)
                    model.fit(X_train, y_train, sample_weight=instance_weights)
                    accuracy, dr, dp, eo, pqp = self._run_evaluation_pd(model, X_val, y_val)
                    diff_count = 0
                    processed_accuracy.append(accuracy)
                    processed_dr.append(dr)
                    processed_dp.append(dp)
                    processed_eo.append(eo)
                    processed_pqp.append(pqp)
                    modification_num.append(diff_count)
                    print(f"diff_count: {diff_count}")

                elif self.sota_method == 'optimized_preprocessing':
                    self.optimized_preprocessing(X_train, y_train, X_val, y_val)                 
                else:
                    raise ValueError('The SOTA method is not supported')

            save_dir = "saved_results/sota_results"
            os.makedirs(save_dir, exist_ok=True)  # Automatically create directory if it doesn't exist

            if save_origin:
                # Save ORIGINAL results of current dataset in a CSV file  
                original_stats = {
                    'original_accuracy': f"{np.mean(original_accuracy):.4f} ± {np.std(original_accuracy):.4f}",
                    'original_dr': f"{np.mean(original_dr):.4f} ± {np.std(original_dr):.4f}",
                    'original_dp': f"{np.mean(original_dp):.4f} ± {np.std(original_dp):.4f}",
                    'original_eo': f"{np.mean(original_eo):.4f} ± {np.std(original_eo):.4f}",
                    'original_pqp': f"{np.mean(original_pqp):.4f} ± {np.std(original_pqp):.4f}",
                }
                original_results = pd.DataFrame(original_stats, index=[self.dataset_name]).T
                csv_file = os.path.join(save_dir, "original_results.csv")  # Storage path
                if os.path.exists(csv_file):
                    existing_df = pd.read_csv(csv_file, index_col=0)
                    existing_df[self.dataset_name] = original_results[self.dataset_name]
                    existing_df.to_csv(csv_file)
                else:
                    original_results.to_csv(csv_file)
                print(f"✅ Original results for {self.dataset_name} have been saved to {csv_file}")

            # Save PROCESSED results of current dataset in a CSV file
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
            processed_csv_file = os.path.join(save_dir, f"{self.sota_method}_results.csv")  # Storage path
            if os.path.exists(processed_csv_file):
                existing_df = pd.read_csv(processed_csv_file, index_col=0)
                existing_df[self.dataset_name] = processed_results[self.dataset_name]
                existing_df.to_csv(processed_csv_file)
            else:
                processed_results.to_csv(processed_csv_file)
            print(f"✅ Processed results for {self.dataset_name} have been saved to {processed_csv_file}")

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