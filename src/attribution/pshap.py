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
            return np.abs(fx - fx_q)
        
        elif self.fairshap_base == "DP":
            outputs = []
            n_samples = X.shape[0]
            # 对每个样本分别计算
            for i in range(n_samples):
                # 取出第 i 个样本 (保持二维)
                xi = X[i:i+1, :]     
                # 对应的敏感属性布尔索引
                priv_idx_i = xi[:, self.sen_att[0]]
                priv_idx_i = np.array(priv_idx_i, dtype=bool)       
                # 模型预测
                y_hat_i = self.model.predict(xi)
                # 构造真实标签（假设 self.y 为单个标签值）
                y_i = np.array([self.y])
                # 计算该样本的混合矩阵统计
                g1_Cm, g0_Cm = marginalised_np_mat(y_i, y_hat_i, 1, priv_idx_i)
                # 根据混合矩阵统计计算 g1 和 g0
                g1 = g1_Cm[0] + g1_Cm[1]
                g1 = zero_division(g1, sum(g1_Cm))
                g0 = g0_Cm[0] + g0_Cm[1]
                g0 = zero_division(g0, sum(g0_Cm))
                # 计算该样本的 DP 公平性值
                sample_output = np.abs(g0 - g1)
                outputs.append(sample_output)
            outputs = np.array(outputs).reshape(-1, 1)
            # print(f'outputs.shape:{outputs.shape}')
            return outputs
        
        elif self.fairshap_base == "EO":
            outputs = []
            n_samples = X.shape[0]
            # 对每个样本分别计算
            for i in range(n_samples):
                # 取出第 i 个样本 (保持二维)
                xi = X[i:i+1, :]     
                # 对应的敏感属性布尔索引
                priv_idx_i = xi[:, self.sen_att[0]]
                priv_idx_i = np.array(priv_idx_i, dtype=bool)       
                # 模型预测
                y_hat_i = self.model.predict(xi)
                # 构造真实标签（假设 self.y 为单个标签值）
                y_i = np.array([self.y])
                # 计算该样本的混合矩阵统计
                g1_Cm, g0_Cm = marginalised_np_mat(y_i, y_hat_i, 1, priv_idx_i)
                # 根据混合矩阵统计计算 g1 和 g0
                g1 = g1_Cm[0] + g1_Cm[2]
                g1 = zero_division(g1_Cm[0], g1)
                g0 = g0_Cm[0] + g0_Cm[2]
                g0 = zero_division(g0_Cm[0], g0)
                # 计算该样本的 DP 公平性值
                sample_output = np.abs(g0 - g1)
                outputs.append(sample_output)
            outputs = np.array(outputs).reshape(-1, 1)
            # print(f'outputs.shape:{outputs.shape}')
            return outputs
        
        elif self.fairshap_base == "PQP":
            outputs = []
            n_samples = X.shape[0]
            # 对每个样本分别计算
            for i in range(n_samples):
                # 取出第 i 个样本 (保持二维)
                xi = X[i:i+1, :]     
                # 对应的敏感属性布尔索引
                priv_idx_i = xi[:, self.sen_att[0]]
                priv_idx_i = np.array(priv_idx_i, dtype=bool)       
                # 模型预测
                y_hat_i = self.model.predict(xi)
                # 构造真实标签（假设 self.y 为单个标签值）
                y_i = np.array([self.y])
                # 计算该样本的混合矩阵统计
                g1_Cm, g0_Cm = marginalised_np_mat(y_i, y_hat_i, 1, priv_idx_i)
                # 根据混合矩阵统计计算 g1 和 g0
                g1 = g1_Cm[0] + g1_Cm[1]
                g1 = zero_division(g1_Cm[0], g1)
                g0 = g0_Cm[0] + g0_Cm[1]
                g0 = zero_division(g0_Cm[0], g0)
                # 计算该样本的 DP 公平性值
                sample_output = np.abs(g0 - g1)
                outputs.append(sample_output)
            outputs = np.array(outputs).reshape(-1, 1)
            # print(f'outputs.shape:{outputs.shape}')
            return outputs

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
    给定完整数据的真实标签 y 和预测标签 y_hat，以及一个布尔向量 priv_idx（表明哪些样本属于特权组），
    将整体数据拆分成**特权组(privileged group)和非特权组(或边缘化组)**两部分，
    然后分别计算'特权组'和'非特权组'，并返回对应的混淆矩阵。

    Param:
        priv_idx: 一个布尔数组，若某个索引位置为 True，则这个样本归为特权组，否则归为边缘化组。
    '''
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