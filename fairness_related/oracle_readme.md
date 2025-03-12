# Discriminative risk (DR)

## Functions

`oracle_metric.py`

- How to get three commonly-used group fairness (GF)?
  ```python
  # y : list/np.ndarray, observed label in the dataset
  # hx: list/np.ndarray, prediction by the classifier
  # pos   : positive_label, e.g., pos=1
  # non_sa: list of boolean, shape=(#inst,)
  #         It represents whether this instance belongs to the priv-
  #         ileged group, only for one single sensitive attribute.
  #         e.g., [False, True, ..., False, True, True]

  _, _, g1_Cm, g0_Cm = marginalised_pd_mat(y, hx, pos, non_sa)
  grp_1 = unpriv_group_one(g1_Cm, g0_Cm)
  grp_2 = unpriv_group_two(g1_Cm, g0_Cm)
  grp_3 = unpriv_group_thr(g1_Cm, g0_Cm)
  gf_1 = group_fairness(*grp_1)  # DP
  gf_2 = group_fairness(*grp_2)  # EO
  gf_3 = group_fairness(*grp_3)  # PQP
  ```

- How to get discriminative risk (DR)?
  ```python
  # y   : list/np.ndarray, observed label in the dataset
  # fx  : list/np.ndarray, prediction by classifier f(.) on
  #       the original dataset, that is, X
  # fx_q: list/np.ndarray, prediction by f(.) on disturbed X'

  # For only one classifier
  dr = hat_L_fair(fx, fx_q)
  lo = hat_L_loss(fx, y)
  # For two different classifiers
  dr = tandem_fair(fa, fa_q, fb, fb_q)
  lo = tandem_loss(fa, fb, y)

  # `lo` means `loss', reflecting the loss/ accuracy aspect
  ```

