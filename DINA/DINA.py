# coding: utf-8

"""
The DINA model, including training and testing
"""


import os
import sys
import logging
import numpy as np
from tqdm import tqdm
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from SidePackage import evaluation as ev

def initial_all_knowledge_state(know_num):
    state_num = 2 ** know_num
    all_states = np.zeros((state_num, know_num))
    for i in range(state_num):
        k, quotient, residue = 1, i // 2, i % 2
        while True:
            all_states[i, know_num - k] = residue
            if quotient <= 0:
                break
            quotient, residue = quotient // 2, quotient % 2
            k += 1
    return all_states


def init_parameters(stu_num, prob_num):
    slip = np.zeros(shape=prob_num) + 0.2
    guess = np.zeros(shape=prob_num) + 0.2
    theta = np.zeros(shape=stu_num)  # the index of state
    return theta, slip, guess


class DINA():
    """
        DINA model, training (EM) and testing methods
        :param R (array): response matrix, shape = (stu_num, prob_num)
        :param q_m (array): Q matrix, shape = (prob_num, know_num)
        :param stu_num (int): number of students
        :param prob_num (int): number of problems
        :param know_num (int): number of knowledge
        :param skip_value (int): skip value in response matrix
    """

    def __init__(self, R, q_m, stu_num, prob_num, know_num, skip_value=-1):
        self.R, self.q_m, self.state_num, self.skip_value = R, q_m, 2 ** know_num, skip_value
        self.stu_num, self.prob_num, self.know_num = stu_num, prob_num, know_num
        # theta: the knowledge state combination for each student
        self.theta, self.slip, self.guess = init_parameters(stu_num, prob_num)
        self.all_states = initial_all_knowledge_state(know_num)  # shape = (state_num, know_num)
        state_prob = np.transpose(np.sum(q_m, axis=1, keepdims=True) - np.dot(q_m, np.transpose(self.all_states)))
        self.eta = 1 - (state_prob > 0)  # state covers knowledge of problem (1: yes), shape = (state_num, prob_num)

    def train(self, epoch, epsilon) -> ...:
        like = np.zeros(shape=(self.stu_num, self.state_num))  # likelihood
        post = np.zeros(shape=(self.stu_num, self.state_num))  # posterior
        theta, slip, guess, tmp_R = np.copy(self.theta), np.copy(self.slip), np.copy(self.guess), np.copy(self.R)
        tmp_R[np.where(self.R == self.skip_value)[0], np.where(self.R == self.skip_value)[1]] = 0

        print(" --- Begin training --- ")

        for iteration in range(epoch):
            post_tmp, slip_tmp, guess_tmp = np.copy(post), np.copy(slip), np.copy(guess)
            answer_right = (1 - slip) * self.eta + guess * (1 - self.eta)
            for s in range(self.state_num):
                log_like = np.log(answer_right[s, :] + 1e-9) * self.R + np.log(1 - answer_right[s, :] + 1e-9) * (
                    1 - self.R)
                log_like[np.where(self.R == self.skip_value)[0], np.where(self.R == self.skip_value)[1]] = 0
                like[:, s] = np.exp(np.sum(log_like, axis=1))
            post = like / np.sum(like, axis=1, keepdims=True)
            # i_l:the expected number of examinees with attribute pattern αl.
            i_l = np.expand_dims(np.sum(post, axis=0), axis=1)  # shape = (state_num, 1)
            # r_jl: the expected number of examinees with attribute pattern αl answering item j correctly.
            r_jl = np.dot(np.transpose(post), tmp_R)  # shape = (state_num, prob_num)
            # r_jl_0: the expected number of examinees among i_jl_0 correctly answering item j.
            # r_jl_1: the expected number of examinees among i_jl_1 correctly answering item j.
            r_jl_0, r_jl_1 = np.sum(r_jl * (1 - self.eta), axis=0), np.sum(r_jl * self.eta, axis=0)
            # i_jl_0: the expected number of examinees lacking at least one of the required attributes for item j.
            # i_jl_1: the expected number of examiness with all the required attributes for item j.
            i_jl_0, i_jl_1 = np.sum(i_l * (1 - self.eta), axis=0), np.sum(i_l * self.eta, axis=0)
            guess, slip = r_jl_0 / i_jl_0, (i_jl_1 - r_jl_1) / i_jl_1

            change = max(np.max(np.abs(post - post_tmp)), np.max(np.abs(slip - slip_tmp)),
                         np.max(np.abs(guess - guess_tmp)))
            theta = np.argmax(post, axis=1)

            print("epoch: %d" % iteration)

            if iteration > 20 or change < epsilon:
                break
        self.theta, self.slip, self.guess = theta, slip, guess

    def eval(self, test_data, prob_desc, R_true) -> tuple:
        pred_score = (1 - self.slip) * self.eta + self.guess * (1 - self.eta)
        test_rmse, test_mae, test_accuracy = [], [], []
        miss_coo = []

        # get the mastery of all knowledge concepts for each student 
        # stu_kn_pro = np.zeros(shape=[len(self.theta), self.q_m.shape[1]], dtype=float)
        stu_kn_pro = dict()
        for stu in range(len(self.theta)):
            stu_kn_pro[stu] = self.all_states[self.theta[stu]]

        _count = 0
        for i in tqdm(test_data, "evaluating"):
            stu, test_id, true_score = i['user_id'], i['item_id'], i['score']
            miss_coo.append((test_id, stu))
            # RMSE
            test_rmse.append((pred_score[self.theta[stu], test_id] - true_score) ** 2)
            # MAE
            test_mae.append(abs(pred_score[self.theta[stu], test_id] - true_score))
            # ACCURACY of objective problems
            if prob_desc[test_id] == 'Obj':  # is Obj
                 _score = round(pred_score[self.theta[stu], test_id])
                 test_accuracy.append(1.0 if _score == true_score else 0.0)
                 _count += 1
        
        # calculate the knowledge krc
        kn_krc_list = ev.cal_diag_krc(prob_desc, R_true, miss_coo, stu_kn_pro, self.q_m)
        krc = np.mean([x for x in kn_krc_list.values()])

        return np.sqrt(np.average(test_rmse)), np.average(test_mae), np.sum(test_accuracy)/_count, krc

    def save(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump({"theta": self.theta, "slip": self.slip, "guess": self.guess}, file)
            logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        with open(filepath, 'rb') as file:
            self.theta, self.slip, self.guess = pickle.load(file).values()
            logging.info("load parameters from %s" % filepath)

    def inc_train(self, inc_train_data, epoch, epsilon):  # incremental training
        for i in inc_train_data:
            stu, test_id, true_score = i['user_id'], i['item_id'], i['score']
            self.R[stu, test_id] = true_score
        self.train(epoch, epsilon)

    def transform(self, records):  # MLE for evaluating student's state
        # max_like_id: diagnose which state among all_states the student belongs to
        # dia_state: binaray vector of length know_num, 0/1 indicates whether masters the knowledge
        answer_right = (1 - self.slip) * self.eta + self.guess * (1 - self.eta)
        log_like = records * np.log(answer_right + 1e-9) + (1 - records) * np.log(1 - answer_right + 1e-9)
        log_like[:, np.where(records == self.skip_value)[0]] = 0
        max_like_id = np.argmax(np.exp(np.sum(log_like, axis=1)))
        dia_state = self.all_states[max_like_id]
        return max_like_id, dia_state