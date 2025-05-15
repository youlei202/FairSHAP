import numpy as np
import shap
from copy import deepcopy
from src.attribution.oracle_metric import perturb_numpy_ver, ell_fair_x

EPSILON = 1e-20


class WeightedExplainer:
    """
    This class provides explanations for model predictions using SHAP values,
    weighted according to a given probability distribution.
    """

    def __init__(self, model, sen_att, priv_val, unpriv_dict, fairshap_base):
        """
        Initializes the WeightedExplainer.

        :param model: A machine learning model that supports the predict_proba method.
                      This model is used to predict probabilities which are necessary
                      for SHAP value computation.
        """
        self.model = model
        self.sen_att = sen_att
        self.priv_val = priv_val
        self.unpriv_dict = unpriv_dict
        self.fairshap_base = fairshap_base  # 'DR', 'DP', 'EO', 'PQP'

    def explain_instance(
        self, x, y, X_baseline, weights, sample_size=500, shap_sample_size="auto"
    ):
        """
        Generates SHAP values for a single instance using a weighted sample of baseline data.

        :param x: The instance to explain. This should be a single data point.
        :param y: The true label of the instance.
        :param X_baseline: A dataset used as a reference or background distribution.
        :param weights: A numpy array of weights corresponding to the probabilities
                        of choosing each instance in X_baseline.
        :param num_samples: The number of samples to draw from X_baseline to create
                            the background dataset for the SHAP explainer.
        :return: An array of SHAP values for the instance.
        """
        self.y = y
        # Normalize weights to ensure they sum to 1
        weights = weights + EPSILON
        weights = weights / (weights.sum())

        # Generate samples weighted by joint probabilities
        indice = np.random.choice(
            X_baseline.shape[0], p=weights, replace=True, size=sample_size
        )
        indice = np.unique(indice)
        sampled_X_baseline = X_baseline[indice]

        # Use the sampled_X_baseline as the background data for this specific explanation
        explainer_temp = shap.KernelExplainer(
            self._fairness_value_function, sampled_X_baseline
        )
        shap_values = explainer_temp.shap_values(x, nsamples=shap_sample_size)

        return shap_values

    def _fairness_value_function(self, X):
        if self.fairshap_base == "DR":
            X_disturbed = perturb_numpy_ver(
                X=X,
                sen_att=self.sen_att,
                priv_val=self.priv_val,
                unpriv_dict=self.unpriv_dict,
                ratio=1.0,
            )
            fx = self.model.predict_proba(X)[:, 1]
            fx_q = self.model.predict_proba(X_disturbed)[:, 1]
            # TODO: There might be problems when the classification problem is not binary
            # result = np.abs(fx - fx_q)
            # print(f'result.shape:{result.shape}')
            outputs = np.abs(fx - fx_q)
            return outputs
        # elif self.fairshap_base == "DP":
        #     X_disturbed = perturb_numpy_ver(
        #         X=X,
        #         sen_att=self.sen_att,
        #         priv_val=self.priv_val,
        #         unpriv_dict=self.unpriv_dict,
        #         ratio=1.0,
        #     )
        #     fx = self.model.predict_proba(X)[:, 1]
        #     fx_q = self.model.predict_proba(X_disturbed)[:, 1]
        #     threshold = 0.5

        #     relu_fx = relu(fx-threshold)
        #     relu_fx_q = relu(fx_q-threshold)

        #     output_dp = relu_fx * relu_fx_q
        #     return output_dp
        # elif self.fairshap_base == "EO":
        #     X_disturbed = perturb_numpy_ver(
        #         X=X,
        #         sen_att=self.sen_att,
        #         priv_val=self.priv_val,
        #         unpriv_dict=self.unpriv_dict,
        #         ratio=1.0,
        #     )
        #     fx = self.model.predict_proba(X)[:, 1]
        #     fx_q = self.model.predict_proba(X_disturbed)[:, 1]
        #     threshold = 0.6

        #     relu_fx = relu(fx-threshold)
        #     relu_fx_q = relu(fx_q-threshold)

        #     output_dp = relu_fx * relu_fx_q
        #     return output_dp
        else:
            raise ValueError("Fairness base not supported")



class FairnessExplainer:
    """
    This class provides SHAP explanations for model predictions across multiple instances,
    using joint probability distributions to weight the baseline data for each instance.
    """

    def __init__(self, model, sen_att, priv_val, unpriv_dict, fairshap_base):
        """
        Initializes the FairnessExplainer.

        :param model: A machine learning model that supports the predict_proba method.
                      This model is used to compute SHAP values using weighted baseline data.
        """
        self.model = model
        self.sen_att = sen_att
        self.priv_val = priv_val
        self.unpriv_dict = unpriv_dict
        self.fairshap_base = fairshap_base  # 'DR', 'DP', 'EO', 'PQP'
        self.weighted_explainer = WeightedExplainer(
            model, sen_att, priv_val, unpriv_dict, fairshap_base
        )

    def _compute_expected_value(self, X_baseline, sample_size):
        """
        Computes the expected value over the baseline dataset.

        :param X_baseline: The baseline dataset.
        :param sample_size: The number of samples to draw from X_baseline.
        :return: The expected value.
        """
        sample_indices = np.random.choice(
            X_baseline.shape[0], size=sample_size, replace=True
        )
        sampled_X_baseline = X_baseline[sample_indices]
        model_output = self.weighted_explainer._fairness_value_function(
            sampled_X_baseline
        )
        return np.mean(model_output)

    def shap_values(
        self,
        X: np.ndarray,
        Y: np.ndarray, # true label of X
        X_baseline: np.ndarray,
        matching,
        sample_size=1000,
        shap_sample_size="auto",
    ):
        """
        Computes SHAP values for multiple instances using a set of joint probability weights.

        :param X: An array of instances to explain. Each instance is a separate data point.
        :param X_baseline: A dataset used as a reference or background distribution.
        :param matching: A matrix of joint probabilities, where each row corresponds to the
                            probabilities for an instance in X, used to weight X_baseline.
        :param num_samples: The number of samples to draw from X_baseline for each instance in X.
        :return: A numpy array of SHAP values for each instance in X.
        """
        self.Y = Y
        # Compute the expected value
        # self.expected_value = self._compute_expected_value(X_baseline, sample_size)

        return np.array(
            [
                self.weighted_explainer.explain_instance(
                    x,
                    y,
                    X_baseline,
                    weights,
                    sample_size=sample_size,
                    shap_sample_size=shap_sample_size,
                )
                for x, y, weights in zip(X, Y, matching)
            ]
        )



def contingency_tab_bi(y, y_hat, pos=1):
    # For one single classifier
    tp = np.sum((y == pos) & (y_hat == pos))
    fn = np.sum((y == pos) & (y_hat != pos))
    fp = np.sum((y != pos) & (y_hat == pos))
    tn = np.sum((y != pos) & (y_hat != pos))
    return tp, fp, fn, tn


def marginalised_np_mat(y, y_hat, pos_label=1, priv_idx=list()):
    '''
    Given the true labels y and predicted labels y_hat for the full dataset,
    and a boolean vector priv_idx (indicating which samples belong to the privileged group),
    this function splits the data into two groups: the **privileged group** and the **marginalized group**.
    It then computes and returns the corresponding confusion matrices for each group.

    Parameters:
        y (np.ndarray or list): True labels.
        y_hat (np.ndarray or list): Predicted labels.
        pos_label (int or str): Label considered as positive (default is 1).
        priv_idx (np.ndarray or list): Boolean array where True indicates that the sample belongs to the privileged group.
    
    Returns:
        tuple: (confusion matrix for privileged group, confusion matrix for marginalized group)
    '''
    if isinstance(y, list) or isinstance(y_hat, list):
        y, y_hat = np.array(y), np.array(y_hat)
    # Extract true and predicted labels for privileged group
    g1_y = y[priv_idx]
    g1_hx = y_hat[priv_idx]
    # Extract true and predicted labels for marginalized group(s)
    g0_y = y[~priv_idx]
    g0_hx = y_hat[~priv_idx]
    # Compute confusion matrices
    g1_Cm = contingency_tab_bi(g1_y, g1_hx, pos_label)  # Confusion matrix for privileged group
    g0_Cm = contingency_tab_bi(g0_y, g0_hx, pos_label)  # Confusion matrix for marginalized group(s)
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

def relu(x):
    return np.maximum(0, x)