# coding: utf-8

import numpy as np
import pandas as pd

from oracle_metric import (marginalised_contingency,
                           marginalised_confusion,
                           marginalised_split_up,
                           marginalised_matrix,
                           marginalised_pd_mat)
from oracle_metric import (unpriv_group_one, unpriv_group_two,
                           unpriv_group_thr, group_fairness)
from oracle_metric import perturb_pandas_ver, perturb_numpy_ver
from oracle_metric import (ell_fair_x, ell_loss_x,
                           hat_L_fair, hat_L_loss,
                           tandem_fair, tandem_loss)


# -------------------------------------
# prelim


def marginal_bin(n, nc):
  y = np.random.randint(nc, size=n)
  hx = np.random.randint(nc, size=n)
  vY = list(range(nc))

  Cij = marginalised_contingency(y, hx, vY, dY=nc)
  assert np.sum(Cij) == n
  Cm = marginalised_confusion(Cij, loc=1)
  assert np.sum(Cm) == n

  if nc == 2:
    # assert np.all(Cij == Cm)
    assert Cij[0, 0] == Cm[1, 1]
    assert Cij[1, 1] == Cm[0, 0]
    assert Cij[0, 1] == Cm[1, 0]
    assert Cij[1, 0] == Cm[0, 1]

  sen = np.random.randint(2, size=n)
  g1_y, g0_y, g1_hx, g0_hx = marginalised_split_up(
      y, hx, priv=1, sen=sen)
  assert len(g1_y) + len(g0_y) == n
  assert len(g1_hx) + len(g0_hx) == n
  assert sum(g1_y) + sum(g0_y) == sum(y)
  assert sum(g1_hx) + sum(g0_hx) == sum(hx)

  g1_Cij, g0_Cij, g1_Cm, g0_Cm = marginalised_matrix(
      y, hx, pos=1, priv=1, sen=sen)
  h1_Cij, h0_Cij, h1_Cm, h0_Cm = marginalised_pd_mat(
      y, hx, pos=1, idx_priv=sen == 1)
  '''
  (g1_Cij, g0_Cij, g1_Cm, g0_Cm), ut_g = marginalised_matrix(
      y, hx, pos=1, priv=1, sen=sen)
  (h1_Cij, h0_Cij, h1_Cm, h0_Cm), ut_h = marginalised_pd_mat(
      y, hx, pos=1, idx_priv=sen == 1)
  '''
  assert np.all(g1_Cm == h1_Cm)
  assert np.all(g0_Cm == h0_Cm)
  assert np.all(g1_Cij == h1_Cij)
  assert np.all(g0_Cij == h0_Cij)

  return


def test_marginal_bin():
  n, nc = 100, 2
  marginal_bin(n, nc)
  n, nc = 100, 3
  marginal_bin(n, nc)


# -------------------------------------
# oracle_metric.py


def grp_fair(n, nc):
  y = np.random.randint(nc, size=n)
  hx = np.random.randint(nc, size=n)
  pos = 1
  non_sa = np.random.randint(2, size=n).astype('bool')
  # start testing

  _, _, g1_Cm, g0_Cm = marginalised_pd_mat(y, hx, pos, non_sa)
  grp_1 = unpriv_group_one(g1_Cm, g0_Cm)
  grp_2 = unpriv_group_two(g1_Cm, g0_Cm)
  grp_3 = unpriv_group_thr(g1_Cm, g0_Cm)
  ans_1 = group_fairness(*grp_1)
  ans_2 = group_fairness(*grp_2)
  ans_3 = group_fairness(*grp_3)

  # assert all([i != 0 for i in [ans_1, ans_2, ans_3]])
  tmp = [i != 0 for i in [ans_1, ans_2, ans_3]]
  if nc == 2:
    assert all(tmp)  # sometimes error when nc=3

  return


def test_grp_fair():
  n, nc = 100, 2
  grp_fair(n, nc)
  n, nc = 100, 3
  grp_fair(n, nc)


# -------------------------------------
# oracle_bounds.py


def fair_bound(n, nd):
  X = np.random.rand(n, nd)
  sen_att = [1, 3]
  X[:, 1] = (X[:, 1] > .35).astype('int')
  # X[:, 3] = (X[:, 3] < .7).astype('int')
  X[:, 3] = np.random.randint(3, size=n)

  X_prime = pd.DataFrame(X, columns=[
      'height', 'sex', 'weight', 'race'])
  sen_att_prime = ['sex', 'race']
  priv_val = [1, 1]
  ratio = .7

  X_pertb = perturb_pandas_ver(
      X_prime, sen_att_prime, priv_val, ratio)
  X_qtb = perturb_numpy_ver(X, sen_att, priv_val, ratio)

  tmp = (X_pertb.drop(columns=sen_att_prime) ==
         X_prime.drop(columns=sen_att_prime)).all()
  assert tmp.values.all()

  generic = list(range(nd))
  for i in sen_att:
    generic.remove(i)
  assert (X[:, generic] == X_qtb[:, generic]).all()
  assert (X[:, sen_att] != X_qtb[:, sen_att]).any()
  tmp_sa = X_pertb[sen_att_prime] != X_prime[sen_att_prime]
  assert tmp_sa.any().values.all()

  return


def fair_DR_prop(n, nc):
  y = np.random.randint(nc, size=n)
  fa = np.random.randint(nc, size=n)
  fb = np.random.randint(nc, size=n)

  sen = list(range(n))
  np.random.shuffle(sen)
  sen = sen[: 10]
  fa_q, fb_q = fa.copy(), fb.copy()
  # np.random.shuffle(fa_q[sen])
  # np.random.shuffle(fb_q[sen])
  fa_q[sen] = nc - 1 - fa_q[sen]
  fb_q[sen] = nc - 1 - fb_q[sen]

  non_sa = list(range(n))
  for i in sen:
    non_sa.remove(i)
  assert (fa[non_sa] == fa_q[non_sa]).all()
  assert (fb[non_sa] == fb_q[non_sa]).all()
  if nc == 2:  # not always
    assert all(fa[sen] != fa_q[sen])
    assert all(fb[sen] != fb_q[sen])
  else:
    assert any(fa[sen] != fa_q[sen])
    assert any(fb[sen] != fb_q[sen])

  # start testing

  ans_f = ell_fair_x(fa, y)
  ans_l = ell_loss_x(fa, y)
  ans_m = ell_fair_x(fa, fa_q)
  assert np.equal(ans_f, ans_l).all()
  ans_f = hat_L_fair(fa, y)
  ans_l = hat_L_loss(fa, y)
  ans_m = hat_L_fair(fa, fa_q)
  assert ans_f == ans_l
  ans_f = tandem_fair(fa, y, fb, y)
  ans_l = tandem_loss(fa, fb, y)
  ans_m = tandem_fair(fa, fa_q, fb, fb_q)
  assert ans_f == ans_l

  return


def test_proposed_DR():
  n, nd = 100, 4
  fair_bound(n, nd)
  nc = 2
  fair_DR_prop(n, nc)
  nc = 3
  fair_DR_prop(n, nc)


# -------------------------------------

# -------------------------------------

# -------------------------------------
