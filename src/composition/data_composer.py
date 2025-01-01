import numpy as np
import string

class DataComposer:
    def __init__(self, x_counterfactual:np, joint_prob:np, method:string):
        self.x_counterfactual = x_counterfactual
        self.joint_probs = joint_prob
        self.method = method

    def calculate_q(self):
        q = A_values(W=self.joint_probs, R=self.x_counterfactual, method=self.method)
        return q


def A_values(W, R, method):
    N, M = W.shape
    _, P = R.shape
    Q = np.zeros((N, P))

    if method == "avg":
        for i in range(N):
            weights = W[i, :]
            # Normalize weights to ensure they sum to 1
            normalized_weights = weights / np.sum(weights)
            # Reshape to match R's rows for broadcasting
            normalized_weights = normalized_weights.reshape(-1, 1)
            # Compute the weighted sum
            Q[i, :] = np.sum(normalized_weights * R, axis=0)
    elif method == "max":
        for i in range(N):
            max_weight_index = np.argmax(W[i, :])
            Q[i, :] = R[max_weight_index, :]
    else:
        raise NotImplementedError
    return Q









# '''
# Here we define the class DataComposer. After we return the q, we also return the unlabel data which deletes the rows of q
# '''

# import numpy as np
# import string

# class DataComposer:
#     def __init__(self, x_counterfactual: np, joint_prob: np, method: string):
#         self.x_counterfactual = x_counterfactual
#         self.joint_probs = joint_prob
#         self.method = method

#     def calculate_q(self):
#         q, unlabel_delete_q = A_values(W=self.joint_probs, R=self.x_counterfactual, method=self.method)
#         return q, unlabel_delete_q


# def A_values(W, R, method):
#     N, M = W.shape
#     _, P = R.shape
#     Q = np.zeros((N, P))

#     if method == "avg":
#         for i in range(N):
#             weights = W[i, :]
#             # Normalize weights to ensure they sum to 1
#             normalized_weights = weights / np.sum(weights)
#             # Reshape to match R's rows for broadcasting
#             normalized_weights = normalized_weights.reshape(-1, 1)
#             # Compute the weighted sum
#             Q[i, :] = np.sum(normalized_weights * R, axis=0)
#     elif method == "max":
#         for i in range(N):
#             max_weight_index = np.argmax(W[i, :])
#             Q[i, :] = R[max_weight_index, :]
#     else:
#         raise NotImplementedError

#     # Create unlabel_delete_q by removing rows that are in Q
#     unlabel_delete_q = R[~np.isin(np.arange(R.shape[0]), np.arange(Q.shape[0])), :]
    
#     return Q, unlabel_delete_q



# def A_values(W, R, method):
#     N, M = W.shape
#     _, P = R.shape
    
#     # 初始化十个矩阵，存储每行前十个最大权重对应的 R 值
#     Q1 = np.zeros((N, P))
#     Q2 = np.zeros((N, P))
#     Q3 = np.zeros((N, P))
#     Q4 = np.zeros((N, P))
#     Q5 = np.zeros((N, P))
#     Q6 = np.zeros((N, P))
#     Q7 = np.zeros((N, P))
#     Q8 = np.zeros((N, P))
#     Q9 = np.zeros((N, P))
#     Q10 = np.zeros((N, P))

#     for i in range(N):
#         # 获取第 i 行的权重
#         weights = W[i, :]

#         # 找到权重前十个最大值的索引
#         top_ten_indices = np.argpartition(weights, -10)[-10:]
        
#         # 对这十个索引按照权重从大到小排序
#         top_ten_indices = top_ten_indices[np.argsort(-weights[top_ten_indices])]
        
#         # 将对应的 R 值分别存入 Q1 到 Q10
#         Q1[i, :] = R[top_ten_indices[0], :]
#         Q2[i, :] = R[top_ten_indices[1], :]
#         Q3[i, :] = R[top_ten_indices[2], :]
#         Q4[i, :] = R[top_ten_indices[3], :]
#         Q5[i, :] = R[top_ten_indices[4], :]
#         Q6[i, :] = R[top_ten_indices[5], :]
#         Q7[i, :] = R[top_ten_indices[6], :]
#         Q8[i, :] = R[top_ten_indices[7], :]
#         Q9[i, :] = R[top_ten_indices[8], :]
#         Q10[i, :] = R[top_ten_indices[9], :]

#     return Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10

