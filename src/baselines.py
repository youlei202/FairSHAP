import string
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

from src.composition.data_composer import DataComposer
from src.matching.ot_matcher import OptimalTransportPolicy
from src.matching.nn_matcher import NearestNeighborDataMatcher
from src.attribution.oracle_metric import perturb_numpy_ver
from src.attribution import FairnessExplainer
from loguru import logger


class Baseline:
    def __init__(self, X_train, y_train, X_test, y_test, X_unlabel, model:string):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_unlabel = X_unlabel
        # self.y_unlabel = y_unlabel
        self.model = model  



    def get_baseline1(self):
        if self.model == 'xgboost':
            model = XGBClassifier()  # 可以替换为 RandomForestClassifier() 等其他模型
        elif model == 'randomforest':
            model = RandomForestClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)

        sen_att_name = ["sex"]
        sen_att = [self.X_train.columns.get_loc(name) for name in sen_att_name]
        priv_val = [1]
        unpriv_dict = [list(set(self.X_train.values[:, sa])) for sa in sen_att]
        for sa_list, pv in zip(unpriv_dict, priv_val):
            sa_list.remove(pv)
        dr_baseline1 = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, model)
        logger.info(f'baseline1: 使用了{self.model}, Accuracy: {accuracy:.3f}, DR: {dr_baseline1:.5f}')

    def get_baseline2(self):
        if self.model == 'xgboost':
            model = XGBClassifier()
        elif model == 'randomforest':
            model = RandomForestClassifier()
        for proporation in [0.2, 0.4, 0.6, 0.8]:
            for num_new_data in [1, 2, 3]:
                random_picks = random_pick_groups(X_unlabel=self.X_unlabel, replacement=True, proportion=proporation, num_new_data=num_new_data)
                matchings = get_matching(random_picks, self.X_train, num_new_data, matcher='nn')
                q = get_q(random_picks, matchings, num_new_data)  # q is array
                new_data = pd.DataFrame(q, columns=self.X_train.columns)
                X_train_and_new_data = pd.concat([self.X_train, new_data], axis=0)
                y_train_and_new_data = pd.concat([self.y_train]*(num_new_data+1), axis=0)
                model.fit(X_train_and_new_data, y_train_and_new_data)
                y_pred = model.predict(self.X_test)
                
                accuracy = accuracy_score(self.y_test, y_pred)
                sen_att_name = ["sex"]
                sen_att = [self.X_train.columns.get_loc(name) for name in sen_att_name]
                priv_val = [1]
                unpriv_dict = [list(set(self.X_train.values[:, sa])) for sa in sen_att]
                for sa_list, pv in zip(unpriv_dict, priv_val):
                    sa_list.remove(pv)
                dr_baseline2 = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, model)
                logger.info(f'baseline2: 使用了{self.model}, proportion: {proporation}, num_new_data: {num_new_data}, Accuracy: {accuracy:.3f}, DR: {dr_baseline2:.5f}')
    

    def get_baseline3(self):
        if self.model == 'xgboost':
            model = XGBClassifier()
        elif model == 'randomforest':
            model = RandomForestClassifier()
        for proporation in [0.2, 0.4, 0.6, 0.8]:
            for num_new_data in [1, 2, 3]:
                random_picks = random_pick_groups(X_unlabel=self.X_unlabel, replacement=True, proportion=proporation, num_new_data=num_new_data)
                matchings = get_matching(random_picks, self.X_train, num_new_data, matcher='nn')
                q = get_q(random_picks, matchings, num_new_data)  # q is array
                new_data = pd.DataFrame(q, columns=self.X_train.columns)

                # Use the original model to label the new data
                original_model = XGBClassifier()
                original_model.fit(self.X_train, self.y_train)
                y_pred_new_label = original_model.predict(new_data)
                y_pred_series = pd.Series(y_pred_new_label)
                X_train_and_new_data = pd.concat([self.X_train, new_data], axis=0)
                y_train_and_new_data = pd.concat([self.y_train, y_pred_series], axis=0)
                
                # Train the model with new data
                model.fit(X_train_and_new_data, y_train_and_new_data)
                y_pred = model.predict(self.X_test)

                accuracy = accuracy_score(self.y_test, y_pred)
                sen_att_name = ["sex"]
                sen_att = [self.X_train.columns.get_loc(name) for name in sen_att_name]
                priv_val = [1]
                unpriv_dict = [list(set(self.X_train.values[:, sa])) for sa in sen_att]
                for sa_list, pv in zip(unpriv_dict, priv_val):
                    sa_list.remove(pv)
                dr_baseline3 = fairness_value_function(sen_att, priv_val, unpriv_dict, self.X_test.values, model)
                logger.info(f'baseline3: 使用了{self.model}, proportion: {proporation}, num_new_data: {num_new_data}, Accuracy: {accuracy:.3f}, DR: {dr_baseline3:.5f}')


def random_pick_groups(X_unlabel, replacement, proportion, num_new_data):
    if replacement == True:
        # calculate the sample size
        sample_size = int(len(X_unlabel) * proportion)
        # random pick
        group_num = num_new_data
        random_picks = []
        for i in range(group_num):
            random_pick = X_unlabel.sample(n=sample_size, random_state=25+i)
            random_picks.append(random_pick)
    else:
        raise ValueError('replacement should be True')
    return tuple(random_picks)


def get_matching(random_pick, X_train, num_new_data, matcher='nn'):
    matchings = []
    for i in range(num_new_data):
        if matcher == 'nn':
            matching = NearestNeighborDataMatcher(X_labeled=X_train, X_unlabeled=random_pick[i]).match(n_neighbors=1)
            matchings.append(matching)

        elif matcher == 'ot':
            matching = OptimalTransportPolicy(X_labeled=X_train, X_unlabeled=random_pick[i]).match(n_neighbors=1)
            matchings.append(matching)
        else:
            raise ValueError('matcher should be nn or ot')
    return tuple(matchings)


def get_q(random_picks, matchings, num_new_data):
    for i in range(num_new_data):
        q = DataComposer(
            x_counterfactual=random_picks[i].values, 
            joint_prob=matchings[i], 
            method="max").calculate_q() 
        if i == 0:
            qs = q
        else:
            qs = np.vstack((qs, q))
    return qs

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