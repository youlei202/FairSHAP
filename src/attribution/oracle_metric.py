# coding: utf-8
#
# FUNCTION:
#
#   Three commonly-used group fairness measure
#     1. DP, demographic parity
#     2. EO, equality of opportunity
#     3. PQP, predictive quality parity
#
#   Discriminative risk (DR)
#

import numpy as np
import numba
import time
import pandas as pd


# =====================================
# Preliminaries


DTY_INT = "int"
DTY_FLT = "float"
CONST_ZERO = 1e-13


def check_zero(tmp, diff=CONST_ZERO):
    return tmp if tmp != 0.0 else diff


def fantasy_timer(func):
    def wrapper(*args, **kw):
        since = time.time()
        ans = func(*args, **kw)
        time_elapsed = time.time() - since
        return ans, time_elapsed

    return wrapper


# =====================================
# Group fairness measures


# -------------------------------------
# prelim

"""
marginalised groups
|      | h(xneg,gzero)=1 | h(xneg,gzero)=0 |
| y= 1 |    TP_{gzero}   |    FN_{gzero}   |
| y= 0 |    FP_{gzero}   |    TN_{gzero}   |
privileged group
|      | h(xneg,gones)=1 | h(xneg,gones)=0 |
| y= 1 |    TP_{gones}   |    FN_{gones}   |
| y= 0 |    FP_{gones}   |    TN_{gones}   |

instance (xneg,xpos) --> (xneg,xqtb)
        xpos might be `gzero` or `gones`

C_{ij}
|     | hx=0 | hx=1 | ... | hx=? |
| y=0 | C_00 | C_01 | ... | C_0* |
| y=1 | C_10 | C_11 |     | C_1* |
| ... | ...  | ...  |     | ...  |
| y=? | C_*0 | C_*1 | ... | C_*? |
"""
# y, hx: list of scalars (as elements)


def marginalised_contingency(y, hx, vY, dY):
    assert len(y) == len(hx), "Shapes do not match."
    Cij = np.zeros(shape=(dY, dY), dtype=DTY_INT)
    for i in range(dY):
        for j in range(dY):
            tmp = np.logical_and(np.equal(y, vY[i]), np.equal(hx, vY[j]))
            # tmp = np.equal(y, vY[i]) & np.equal(hx, vY[j])
            Cij[i, j] = np.sum(tmp)
    return Cij  # np.ndarray


@numba.jit(nopython=True)
def marginalised_confusion(Cij, loc=1):
    Cm = np.zeros((2, 2), dtype=DTY_INT)

    Cm[0, 0] = Cij[loc, loc]
    Cm[0, 1] = np.sum(Cij[loc]) - Cij[loc, loc]
    Cm[1, 0] = np.sum(Cij[:, loc]) - Cij[loc, loc]

    Cm[1, 1] = np.sum(Cij) + Cij[loc, loc] - np.sum(Cij[loc]) - np.sum(Cij[:, loc])
    return Cm  # np.ndarray


@numba.jit(nopython=True)
def marginalised_split_up(y, hx, priv=1, sen=list()):
    gones_y_ = [i for i, j in zip(y, sen) if j == priv]
    gzero_y_ = [i for i, j in zip(y, sen) if j != priv]
    gones_hx = [i for i, j in zip(hx, sen) if j == priv]
    gzero_hx = [i for i, j in zip(hx, sen) if j != priv]
    return gones_y_, gzero_y_, gones_hx, gzero_hx


# @fantasy_timer
def marginalised_matrix(y, hx, pos=1, priv=1, sen=list()):
    """params
    y/hx: list, shape=(N,), true label and prediction
    pos : which label is viewed as positive, usually binary-class
    sen : which group these instances are from, including one priv-
          ileged group and one/multiple marginalised group(s).
          or list of boolean (as elements)
    priv: which one indicates the privileged group.
    """
    vY = sorted(set(y) | set(hx))  # union set
    dY = len(vY)
    gones_y_, gzero_y_, gones_hx, gzero_hx = marginalised_split_up(y, hx, priv, sen)
    g1_Cij = marginalised_contingency(gones_y_, gones_hx, vY, dY)
    g0_Cij = marginalised_contingency(gzero_y_, gzero_hx, vY, dY)

    loca = vY.index(pos)  # [[TP,FN],[FP,TN]]
    gones_Cm = marginalised_confusion(g1_Cij, loca)
    gzero_Cm = marginalised_confusion(g0_Cij, loca)
    return g1_Cij, g0_Cij, gones_Cm, gzero_Cm  # np.ndarray


# @fantasy_timer
def marginalised_pd_mat(y, hx, pos=1, idx_priv=list()):
    if isinstance(y, list) or isinstance(hx, list):
        y, hx = np.array(y), np.array(hx)
    tmp = y.tolist() + hx.tolist()
    vY = sorted(set(tmp))
    dY = len(vY)

    gones_y_ = y[idx_priv].tolist()
    gzero_y_ = y[np.logical_not(idx_priv)].tolist()
    gones_hx = hx[idx_priv].tolist()
    gzero_hx = hx[np.logical_not(idx_priv)].tolist()

    g1_Cij = marginalised_contingency(gones_y_, gones_hx, vY, dY)
    g0_Cij = marginalised_contingency(gzero_y_, gzero_hx, vY, dY)
    loca = vY.index(pos)

    gones_Cm = marginalised_confusion(g1_Cij, loca)
    gzero_Cm = marginalised_confusion(g0_Cij, loca)
    # gones_Cm: for the privileged group
    # gzero_Cm: for marginalised groups
    return g1_Cij, g0_Cij, gones_Cm, gzero_Cm  # np.ndarray


# -------------------------------------
# Group fairness measure

""" Cm
|        | hx= pos | hx= neg |
| y= pos |    TP   |    FN   |
| y= neg |    FP   |    TN   |
"""


# 1) demographic parity
# 人口统计均等
# aka. (TP+FP)/N = P[h(x)=1]


def unpriv_group_one(gones_Cm, gzero_Cm):
    N1 = np.sum(gones_Cm)
    N0 = np.sum(gzero_Cm)
    N1 = check_zero(N1.tolist())
    N0 = check_zero(N0.tolist())
    g1 = (gones_Cm[0, 0] + gones_Cm[1, 0]) / N1
    g0 = (gzero_Cm[0, 0] + gzero_Cm[1, 0]) / N0
    return float(g1), float(g0)


# 2) equality of opportunity
# 胜率均等
# aka. TP/(TP+FN) = recall
#                 = P[h(x)=1, y=1 | y=1]


def unpriv_group_two(gones_Cm, gzero_Cm):
    t1 = gones_Cm[0, 0] + gones_Cm[0, 1]
    t0 = gzero_Cm[0, 0] + gzero_Cm[0, 1]
    g1 = gones_Cm[0, 0] / check_zero(t1)
    g0 = gzero_Cm[0, 0] / check_zero(t0)
    return float(g1), float(g0)


# 3) predictive quality parity
# 预测概率均等
# aka. TP/(TP+FP) = precision
#                 = P[h(x)=1, y=1 | h(x)=1]


def unpriv_group_thr(gones_Cm, gzero_Cm):
    t1 = gones_Cm[0, 0] + gones_Cm[1, 0]
    t0 = gzero_Cm[0, 0] + gzero_Cm[1, 0]
    g1 = gones_Cm[0, 0] / check_zero(t1)
    g0 = gzero_Cm[0, 0] / check_zero(t0)
    return float(g1), float(g0)


# -------------------------------------
# Works for all three options


def group_fairness(g1, g0):
    return abs(g1 - g0)


# =====================================
# Discriminative risk (DR)


# -------------------------------------
# prelim

# original instance : (xneg, xpos, y )
# slightly disturbed: (xneg, xqtb, y')
#
# X, X': list, shape (nb_inst, nb_feat)
# y, y': list, shape (nb_inst,)
# sensitive attributes: list, (nb_feat,)
#       \in {0,1}^nb_feat
#       represent: if it is a sensitive attribute


"""
def disturb_slightly(X, loc_sen=None, ratio=.5):
  if (not loc_sen) or (not isinstance(sen, list)):
    return X

  dim = np.shape(X)
  X_qtb = np.array(X)

  for i in range(dim[0]):
    Ti = X_qtb[i]
    Tq = 1 - Ti
    Tk = np.random.rand(dim[1])
    Tk = np.logical_and(Tk, sen)
    T = [q if k else j for j, q, k in zip(Ti, Tq, Tk)]

    X_qtb[i] = T
  return X_qtb.tolist()
"""


def perturb_pandas_ver(X, sen_att, priv_val, ratio=0.5):
    """params
    X       : a pd.DataFrame
    sen_att : list, column name(s) of sensitive attribute(s)
    priv_val: list, privileged value for each sen-att
    """
    unpriv_dict = [X[sa].unique().tolist() for sa in sen_att]
    for sa_list, pv in zip(unpriv_dict, priv_val):
        sa_list.remove(pv)

    X_qtb = X.copy()
    num, dim = len(X_qtb), len(sen_att)
    if dim > 1:
        new_attr_name = "-".join(sen_att)

    for i, ti in enumerate(X.index):
        prng = np.random.rand(dim)
        prng = prng <= ratio

        for j, sa, pv, un in zip(range(dim), sen_att, priv_val, unpriv_dict):
            if not prng[j]:
                continue

            if X_qtb.iloc[i][sa] != pv:
                X_qtb.loc[ti, sa] = pv
            else:
                X_qtb.loc[ti, sa] = np.random.choice(un)

            """
      if dim > 1:
        X_qtb.loc[ti, new_attr_name] = '-'.join([
            X_qtb.iloc[i][sa] for sa in sen_att])
      """
    return X_qtb  # pd.DataFrame


def perturb_numpy_ver(X, sen_att, priv_val, unpriv_dict, ratio=0.5):
    """params
    X       : a np.ndarray
    sen_att : list, column index of sensitive attribute(s)
    priv_val: list, privileged value for each sen-att
    """

    X_qtb = X.copy()
    num, dim = len(X_qtb), len(sen_att)

    for i in range(num):
        prng = np.random.rand(dim)
        prng = prng <= ratio

        for j, sa, pv, un in zip(range(dim), sen_att, priv_val, unpriv_dict):
            if not prng[j]:
                continue

            if X_qtb[i, sa] != pv:
                X_qtb[i, sa] = pv
            else:
                X_qtb[i, sa] = np.random.choice(un)

    return X_qtb  # np.ndarray


# -------------------------------------
# Discriminative risk (DR)


# Could work on one single instance


def ell_fair_x(fx, fx_q):
    # both are: list, shape (#inst,)
    # function is symmetrical
    # value belongs to {0,1}, set

    # return np.not_equal(fx, fx_q).tolist()
    tmp = np.not_equal(fx, fx_q).astype(DTY_FLT)
    return tmp.tolist()


def ell_loss_x(fx, y):
    # both are: list, shape (#inst,)

    # return np.not_equal(fx, y).tolist()
    tmp = np.not_equal(fx, y).astype(DTY_FLT)
    return tmp.tolist()


# Works for the whole dataset


@numba.jit(nopython=True)
def hat_L_fair(fx, fx_q):
    # both are: list, shape (#inst,)
    # function is symmetrical
    # value belongs to [0,1], interval
    """
    L_fair = ell_fair_x(fx, fx_q)
    return np.mean(L_fair).tolist()  # float
    """
    L_fair = np.not_equal(fx, fx_q)
    return float(np.mean(L_fair))


@numba.jit(nopython=True)
def hat_L_loss(fx, y):
    # both are: list, shape (#inst,)
    """
    L_loss = ell_loss_x(fx, y)
    return np.mean(L_loss).tolist()  # float
    """
    L_loss = np.not_equal(fx, y)
    return float(np.mean(L_loss))


# For two different individual members,


@numba.jit(nopython=True)
def tandem_fair(fa, fa_q, fb, fb_q):
    # whole: list, shape (#inst,)
    ha = np.not_equal(fa, fa_q)
    hb = np.not_equal(fb, fb_q)
    tmp = np.logical_and(ha, hb)
    # return np.mean(tmp).tolist()  # float
    return float(np.mean(tmp))


@numba.jit(nopython=True)
def tandem_loss(fa, fb, y):
    # whole: list, shape (#inst,)
    ha = np.not_equal(fa, y)
    hb = np.not_equal(fb, y)
    tmp = np.logical_and(ha, hb)
    # return np.mean(tmp).tolist()  # float
    return float(np.mean(tmp))


# -------------------------------------
# Discriminative risk (DR)


# -------------------------------------


# =====================================
