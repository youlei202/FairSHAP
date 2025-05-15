# coding: utf-8
import numpy as np

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


if __name__ == "__main__":

  # g1_y=0
  # g1_hx=1  #predict
  # g0_y=0   #当只有一个instance的时候，我们可以假设只有g1,且g0为空 
  # g0_hx=0
  # g1_Cm = contingency_tab_bi(y=g1_y, y_hat=g1_hx, pos=1)
  # print(g1_Cm)

  # g1 = g1_Cm[0] + g1_Cm[1]
  # g1 = zero_division(g1, sum(g1_Cm))
  # DP = abs(g1 - 0)
  # print(f'DP:{DP}')

  # g1 = g1_Cm[0] + g1_Cm[2]
  # g1 = zero_division(g1_Cm[0], g1)
  # EO = abs(g1 - 0)
  # print(f'EO:{EO}')

  # g1 = g1_Cm[0] + g1_Cm[1]
  # g1 = zero_division(g1_Cm[0], g1)
  # PQP = abs(g1 - 0)
  # print(f'PQP:{PQP}')


  
  # 要测试的四种 (g1_y, g1_hx) 组合
  pairs = [(1,1), (1,0), (0,1), (0,0)]

  for i, (g1_y, g1_hx) in enumerate(pairs, 1):
      # 这里假设 g1_y, g1_hx 各只有一个样本，可用 array([值]) 包起来
      arr_y = np.array([g1_y])
      arr_hat = np.array([g1_hx])
      
      # 计算混淆矩阵 (tp, fp, fn, tn)
      g1_Cm = contingency_tab_bi(y=arr_y, y_hat=arr_hat, pos=1)
      
      # 假设 g0_y, g0_hx 恒等于 0，对应“g0为空”或只存一个负例
      # 但本例只演示 g1_Cm 的计算，g0_Cm 只作参考时为 (0,0,0,0) 或类似
      g0_Cm = (0,0,0,0)
      
      print(f"=== 测试第 {i} 组: (g1_y={g1_y}, g1_hx={g1_hx}) ===")
      print("g1_Cm =", g1_Cm)

      # -- 计算 Demographic Parity (DP) --
      #   DP = | P(f=1|g1) - P(f=1|g0) |
      #   这里 g0_Cm=0，所以 P(f=1|g0)=0
      numerator = g1_Cm[0] + g1_Cm[1]  # tp + fp
      denominator = sum(g1_Cm)        # 总样本 = tp+fp+fn+tn
      g1_rate = zero_division(numerator, denominator)
      DP = abs(g1_rate - 0)
      
      # -- 计算 Equality of Opportunity (EO) --
      #   EO = | TPR(g1) - TPR(g0) |
      #   这里 g0_Cm=0，所以 TPR(g0)=0
      tpr_denominator = g1_Cm[0] + g1_Cm[2]  # tp + fn = 真实为正的总数
      g1_tpr = zero_division(g1_Cm[0], tpr_denominator)
      EO = abs(g1_tpr - 0)
      
      # -- 计算 Predictive Parity (PQP) --
      #   PQP = | Precision(g1) - Precision(g0) |
      #   这里 g0_Cm=0，所以 Precision(g0)=0
      prec_denominator = g1_Cm[0] + g1_Cm[1]  # tp + fp = 预测为正的总数
      g1_prec = zero_division(g1_Cm[0], prec_denominator)
      PQP = abs(g1_prec - 0)

      # 打印结果
      print(f"DP:  {DP:.4f}")
      print(f"EO:  {EO:.4f}")
      print(f"PQP: {PQP:.4f}")
      print("------------------------------------")  
  pass