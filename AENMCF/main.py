# -*- coding: utf-8 -*-
"""
AE-NMCF model training and testing
"""

import os
import sys
import warnings
import time
import numpy as np
import pickle
import logging
import math
import pandas as pd
from scipy import integrate


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import ae_nmcf
from SidePackage import preprocessing as pre
from SidePackage import auxiliary as aux
from SidePackage import evaluation as ev




def out_to_file(path, model_name):
    class logger(object):
        
        def __init__(self, file_name, path):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, file_name), mode='a', encoding='utf8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        
        def flush(self):
            pass

    sys.stdout = logger(model_name + '.log', path=path)


def save(filepath, b, u, e, v, m):
    with open(filepath, 'wb') as file:
        pickle.dump({"B":b, "U":u, "E":e, "V":v, "M":m}, file)
        logging.info("save parameters to %s" % filepath)


def load(filepath):
    with open(filepath, 'rb') as file:
        b, u, e, v, m = pickle.load(file).values()
        logging.info("load parameters from %s" % filepath)
    return b, u, e, v, m


def aenmcf_train(train_data, train_fill, q_m, rank, gamma, step_search, gbe=False):
    """
    FUNCTION: model training

    Inputs:
    -------
    :param train_data --> numpy.ndarray
        the student-exercise data matrix which is used for training (randomly delete some entries).
        row: exercises
        col: students
    
    :param train_fill --> numpy.ndarray
        the student-exercise data matrix (train_data) whose missing entries are filled.

    :param q_m --> numpy.ndarray
        the Q-matrix
        row: exercises
        col: knowledge concepts
    
    :param rank --> int

    :param gamma--> float

    :param step_search --> str
        the step-size search method (Armijo or Lipschitz)

    Outputs:
    -------
    :return b --> numpy.ndarray
        the exercise-knowledge association matrix

    :return u --> numpy.ndarray
        the student feature matrix

    :return e --> numpy.ndarray
        the exercise feature matrix
    
    :return v --> numpy.ndarray
        the knowledge concept feature matrix
    
    :return m --> numpy.ndarray
        the exercise difficulty vector 
    """

    print("\nAE-NMCF hyperparameters: \nrank: %d\ngamma: %.2f" % (rank, gamma))

    if gbe is True:
        w = cal_weight(train_fill)
    else:
        w = cal_weight(train_data)
    
    # --- model training ---
    start = time.time()
    model = ae_nmcf.ae_nmcf(train_data, train_fill, q_m, w, rank, gamma)
    b, u, e, v, m = model.train(step_search)
    end = time.time()
    print('TIME:%.5f' %(end - start))
    
    return b, u, e, v, m


def cal_weight(student_exe):
    """
    FUNCTION: get the weight matrix of the student scoring matrix: W,
    where the element value equals 0 if the student has no response in the exercise, otherwise, 1.
    """
    weight_matrix = np.ones(shape=student_exe.shape)  # initialization
    weight_matrix[np.isnan(student_exe)] = 0
    return weight_matrix.astype(np.int)


def aenmcf_test(stu_exe, test_loc, q_m, know_graph, b, u, e, v, m, prob_desc, cl):
    """
    FUNCTION: model testing
    Inputs:
    -------
    :param stu_exe --> numpy.ndarray
        the true data of the students' scoring matrix
    """

    ex_num, st_num = stu_exe.shape
    kn_num = q_m.shape[1]

    stu_kn_pro = np.dot(v.T, u).T
    # scale the value between 0-1
    stu_kn_pro_norm = np.zeros(shape=(stu_kn_pro.shape), dtype=float)
    for kn in range(kn_num):
        stu_kn_pro_norm[:,kn] = \
        (stu_kn_pro[:,kn] - stu_kn_pro[:,kn].min()) / (stu_kn_pro[:,kn].max() - stu_kn_pro[:,kn].min())

    stu_ex_pre = cal_stu_ex_pre(b, u, v, m, st_num)

    # --- the evaluation for predicting student performance ---
    # the accuracy of the predicted student performance on objective exercises
    accuracy_obj = ev.cal_accuracy_obj(stu_exe, stu_ex_pre, test_loc, prob_desc)
    # the root mean square error
    rmse = ev.cal_rmse(stu_exe, stu_ex_pre, test_loc)

    # --- the evaluation for cognitive diagnosis task ---
    # the KRC for diagnosis results
    kn_krc_list = ev.cal_diag_krc(prob_desc, stu_exe, test_loc, stu_kn_pro, q_m)
    krc = np.mean([x for x in kn_krc_list.values()])

    print("ACCURACY: %.5f, RMSE: %.5f, KRC: %.5f" % (accuracy_obj, rmse, krc))


def cal_stu_ex_pre(b, u, v, m, st_num):
    phi_fun = np.frompyfunc(lambda x:integrate.quad(normal, -float('inf'), x)[0], 1, 1)
    delta = np.dot(np.dot(b, v.T), u) + np.tile(m, st_num)
    phi = phi_fun(delta)

    return phi

def normal(x):
    return 1/math.sqrt(2*math.pi) * math.exp(-math.pow(x,2)/2)




if __name__ == '__main__':

    MISS_R = 0.2  # the testing ratio
    is_GBE = False  # fill missing value using the Global Baseline Estimation method (default:False)
    CL = 0.05  # the confidience level for the Wilcoxon-signed-rank-test Passing Ratio (PR)

    # --- HYPER PARAMETERS ---
    # for the AE-NMCF framework
    RANK = 3  # the rank of the matrix decomposition
    gamma = 2  # the l2 regularization parameter for B

    DATASET = input("\nplease choose a dataset: [FrcSub, Quanlang-s, Junyi-s, unit-eng, unit-bio-small, unit-his-small]: ")
    print("dataset %s is choosed" % DATASET)
    if DATASET not in ['FrcSub', 'Quanlang-s', 'Junyi-s', 'unit-eng', 'unit-bio-small', 'unit-his-small']:
        warnings.warn("dataset does not exist.")
        exit()
    # 1. student-exercise matrix (row: exercises, col: students)
    stu_exe = ((np.loadtxt(BASE_DIR + "/Data/" + DATASET + "/data.txt")).astype(float)).T
    # partial correct --> wrong
    non_nan_index = np.where(~np.isnan(stu_exe))  # find the non-NaN index
    stu_exe[non_nan_index] = stu_exe[non_nan_index].astype(int)
    # 2. Q-matrix
    q_m = np.loadtxt(BASE_DIR + "/Data/" + DATASET +"/q.txt", dtype=int)
    # 3. problem description
    if os.path.exists(BASE_DIR + "/Data/" + DATASET + "/problemdesc.txt"):
        prob_desc = aux.read_problem_desc(BASE_DIR + "/Data/" + DATASET + "/problemdesc.txt")
    else:
        prob_desc = aux.build_problem_desc(stu_exe)
    # 4. the knowledge dependency map
    kn_graph = None
    if os.path.exists(BASE_DIR + "/Data/" + DATASET + "/hier.csv"):
        kn_graph = pd.read_csv(BASE_DIR + "/Data/" + DATASET +'/hier.csv')  # get the prerequisite graph

    # divdide the student-exercise matrix into traning data and testing data
    is_divide = input("re-divide the dataset? (yes or no): ")
    if is_divide == "yes":
        # get the training set and testing index of student-exercise matrix
        train_data, test_loc = pre.missing_stu_exe(stu_exe, MISS_R)
        np.savetxt(BASE_DIR + "/AENMCF/data/train@" + DATASET + ".txt", train_data, fmt='%.4f')
        np.savetxt(BASE_DIR + "/AENMCF/data/test@" + DATASET + ".txt", np.array(test_loc), delimiter=' ', fmt='%s')
        print("the data division has been completed.")
    elif is_divide == "no":
        pass
    else:
        warnings.warn("illegal input!")
        exit()
    
    # get the training set
    train_data = (np.loadtxt(BASE_DIR + "/AENMCF/data/train@" + DATASET + ".txt")).astype(float)
    # get the testing index
    test_loc = aux.read_test_loc(BASE_DIR + "/AENMCF/data/test@" + DATASET + ".txt")

    # filling the missing value
    if DATASET in ['FrcSub']:
        train_fill = pre.matrix_miss_fill_GBE(train_data)  # using the Global Baseline Estimation (GBE)
        is_GBE = True
    elif DATASET in ['Quanlang-s', 'Junyi-s', 'unit-eng', 'unit-bio-small', 'unit-his-small']:
        train_fill = pre.matrix_miss_fill(train_data)  # randomly fill
    
    # out_to_file(BASE_DIR + "/AENMCF/log/", 'aenmcf@' + DATASET)

    # -- model training ---
    step_search = 'lipschitz'  # armijo or lipschitz
    print('The step-size searching method is: %s' %step_search)
    b, u, e, v, m = aenmcf_train(train_data, train_fill, q_m, RANK, gamma, step_search, gbe=is_GBE)
    # save(BASE_DIR + "/AENMCF/params/" + step_search + "@" + DATASET, b, u, e, v, m)
    # b, u, e, v, m = load(BASE_DIR + "/AENMCF/params/" + step_search + "@" + DATASET)
    aenmcf_test(stu_exe, test_loc, q_m, kn_graph, b, u, e, v, m, prob_desc, CL)