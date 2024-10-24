from Initial_data import DataSet
from QRCDM_model import QRCDM
from Test_Model import test_model
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import torch
import sys
import os
import warnings
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from SidePackage import auxiliary as aux
from SidePackage import evaluation as ev
from data_pre import process


def cognitive_diagnosis(prob_desc, stu_exe, test_loc, stu_pro, q_m):
    # the KRC for diagnosis results
    kn_krc_list = ev.cal_diag_krc(prob_desc, stu_exe, test_loc, stu_pro, q_m)
    krc = np.mean([x for x in kn_krc_list.values()])
    return krc



if __name__ == '__main__':
    basedir = BASE_DIR + '/QRCDM'
    dataSet_list = ('FrcSub', 'Quanlang-s', 'Junyi-s', 'unit-eng', 'unit-bio-small', 'unit-his-small')
    data_set_name = dataSet_list[0]
    print('data set: ', data_set_name)
    batch_size = 32
    lr = 9e-3
    epochs = 15
    device = 'cpu'  # cuda / cpu

    # --- load the original data ---
    data_dir = BASE_DIR + '/Data/' + data_set_name
    stu_exe = (np.loadtxt(data_dir + "/data.txt")).astype(float)  # row:students / col:exercises
    q_m = np.loadtxt(data_dir +"/q.txt", dtype=int)  # row:exercises / col:knowledge

    exe_num = q_m.shape[0]
    _del_exes = []
    for exe in range(exe_num):
        if (q_m[exe] == 0).all():  # there are not any knowledge concept
            _del_exes.append(exe)
    # deletet the exercises that has not any knowledge concepts from the original datasets
    stu_exe = np.delete(stu_exe, _del_exes, axis=1)
    q_m = np.delete(q_m, _del_exes, axis=0)

    # --- data division ---
    is_divide = input("re-divide the dataset? (yes or no): ")
    if is_divide == "yes":
        stu_num, exe_num, kn_num = process(data_set_name, stu_exe, q_m)
        print('there are %d students, %d exercises, %d knowledge concepts' % (stu_num, exe_num, kn_num))
        print("the data division has been completed.")
    elif is_divide == "no":
        pass
    else:
        warnings.warn("illegal input!")
        exit()      

    start = time.time()

    dataSet = DataSet(basedir, data_set_name)
    Q = dataSet.get_Q()
    train_data = dataSet.train_data
    valid_data = dataSet.valid_data
    test_data = dataSet.test_data
    obj_prob_index = dataSet.obj_prob_index
    sub_prob_index = dataSet.sub_prob_index

    total_stu_list = set(train_data.index) & \
        set(valid_data.index) & set(test_data.index)
    train_loader = DataLoader(TensorDataset(torch.tensor(list(total_stu_list)).float()),
                              batch_size=batch_size, shuffle=True)

    model = QRCDM(Q=Q, lr=lr, device=device)
    model.train_model(train_loader, train_data, valid_data, obj_prob_index, sub_prob_index, epochs=epochs)

    # save_dir = dataSet.save_dir
    # model.save_parameter(dataSet.save_dir)

    W_ = model.W_.cpu().detach().numpy()  # row:exercises / col:knowledge concepts
    D_ = model.D_.cpu().detach().numpy()
    guess_ = model.guess_.cpu().detach().numpy()
    miss_ = model.miss_.cpu().detach().numpy()

    end = time.time()
    print('TIME:%.5f' %(end - start))

    # *** 1. student performance prediction ***
    test_result = test_model(W_, D_, guess_, miss_, train_loader, train_data, test_data,
                             obj_prob_index, sub_prob_index, device='cpu')

    # *** 2. cognitive diagnosis ***
    if os.path.exists(BASE_DIR + "/Data/" + data_set_name + "/problemdesc.txt"):
        prob_desc = aux.read_problem_desc(BASE_DIR + "/Data/" + data_set_name + "/problemdesc.txt")
    else:
        prob_desc = aux.build_problem_desc(stu_exe.T)
    test_loc = []
    for index, row in test_data.iterrows():
        stu = index - 1
        exe = int(row['item_id'] - 1)
        test_loc.append([exe, stu])
    
    # get the students' knowledge proficiency
    stu_pro = np.nan * np.ones(shape=[stu_exe.shape[0], q_m.shape[1]])
    stu_exe_list, stu_score_list = {}, {}
    for index, row in train_data.iterrows():
        stu = index - 1
        exe = int(row['item_id'] - 1)
        score = row['score']
        if not stu in stu_exe_list.keys():
            stu_exe_list[stu] = []
            stu_score_list[stu] = []
        stu_exe_list[stu].append(exe)
        stu_score_list[stu].append(score)
    
    W_ = torch.tensor(W_).float().to(device)
    for stu, exes in stu_exe_list.items():
         scores = stu_score_list[stu]
         X_i = torch.tensor(scores).float().to(device).reshape(1, -1)
         W_i = torch.softmax(W_[exes], dim=0)
         stu_pro[stu] = X_i @ W_i

    krc = cognitive_diagnosis(prob_desc, stu_exe.T, test_loc, stu_pro, q_m)
    print('krc: %.5f' % krc)