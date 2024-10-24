# -*- coding: utf-8 -*-

import os
import sys
import warnings
import numpy as np
from sklearn.decomposition import NMF
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from SidePackage import evaluation as ev
from SidePackage import preprocessing as pre
from SidePackage import auxiliary as aux


def nmf_train_test(stu_exe, stu_exe_fill, miss_coo, exe_desc, rank_max=None):

    shape_se = stu_exe.shape

    rank_min = 1
    if rank_max is None:
      rank_max = min(shape_se)

    for RANK in range(rank_min, rank_max + 1):
        print('--- RANK:[%d] ---' % RANK)

        start = time.time()
        # model = NMF(n_components=RANK, init='random', random_state=0, max_iter=10000, regularization='both')
        model = NMF(n_components=RANK, init='random', random_state=0, max_iter=10000)
        W = model.fit_transform(stu_exe_fill)
        H = model.components_
        stu_exe_pre = np.dot(W, H)
        end = time.time()
        
        print('TIME:%.5f' %(end - start))

        cor_se_obj, rmse = nmf_testing(stu_exe=stu_exe,stu_exe_pre=stu_exe_pre,exe_desc=exe_desc,miss_coo=miss_coo)

        print("acc", cor_se_obj)
        print("RMSE = ", rmse)


def nmf_testing(stu_exe, stu_exe_pre, exe_desc, miss_coo):
  
    cor_se_obj = ev.cal_accuracy_obj(matrix_ori=stu_exe, matrix_pre=stu_exe_pre, miss_coo=miss_coo, exe_desc=exe_desc)
    rmse = ev.cal_rmse(matrix_ori=stu_exe, matrix_pre=stu_exe_pre, miss_coo=miss_coo)

    return cor_se_obj, rmse




if __name__ == '__main__':
    DATASET = input("\nplease choose a dataset: [FrcSub, Quanlang-s, Junyi-s, unit-eng, unit-bio-small, unit-his-small]: ")
    print("dataset %s is choosed" % DATASET)
    if DATASET not in ['FrcSub', 'Quanlang-s', 'Junyi-s', 'unit-eng', 'unit-bio-small', 'unit-his-small']:
        warnings.warn("dataset does not exist.")
        exit()
    
    MISS_R = 0.2  # the testing ratio

    # student-exercise matrix (row: exercises, col: students)
    stu_exe = ((np.loadtxt(BASE_DIR + "/Data/" + DATASET + "/data.txt")).astype(float)).T
    # divdide the student-exercise matrix into traning data and testing data
    is_divide = input("re-divide the dataset? (yes or no): ")
    if is_divide == "yes":
        # get the training set and testing index of student-exercise matrix
        train_data, test_loc = pre.missing_stu_exe(stu_exe, MISS_R)
        np.savetxt(BASE_DIR + "/NMF/data/train@" + DATASET + ".txt", train_data, fmt='%.4f')
        np.savetxt(BASE_DIR + "/NMF/data/test@" + DATASET + ".txt", np.array(test_loc), delimiter=' ', fmt='%s')
        print("the data division has been completed.")
    elif is_divide == "no":
        pass
    else:
        warnings.warn("illegal input!")
        exit()      
    # get the training set
    train_data = ((np.loadtxt(BASE_DIR + "/NMF/data/train@" + DATASET + ".txt")).astype(float))
    # get the testing index
    test_loc = aux.read_test_loc(BASE_DIR + "/NMF/data/test@" + DATASET + ".txt")

    # the problem description
    if os.path.exists(BASE_DIR + "/Data/" + DATASET + "/problemdesc.txt"):
        prob_desc = aux.read_problem_desc(BASE_DIR + "/Data/" + DATASET + "/problemdesc.txt")
    else:
        prob_desc = aux.build_problem_desc(stu_exe)

    # -- training and testing ---
    stu_exe_fill = pre.matrix_miss_fill(train_data)  # filling the NaN value
    R_max = 10  # manually set the maximum of the rank
    nmf_train_test(stu_exe, stu_exe_fill, test_loc, prob_desc, R_max)