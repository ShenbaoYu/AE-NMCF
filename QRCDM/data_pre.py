"""
Data processing for any dataset to be suitable for QRCDM.
"""

import os
import sys
import random
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from SidePackage import auxiliary as aux

def process(dataset, stu_exe, q_m):
    
    print('choose dataset: ', dataset)
    divide_ratio = 0.8

    save_dir = BASE_DIR + '/QRCDM/data/' + dataset

    st_num, ex_num = stu_exe.shape
    kn_num = q_m.shape[1]

    # --- save item.csv ---
    # NOTE: all IDs start from 1
    _item = pd.DataFrame(columns=['item_id', 'knowledge_code'])
    for exe in range(ex_num):
        kn_ids = [(_+1) for _ in range(kn_num) if q_m[exe][_] == 1]
        _item.loc[_item.shape[0]] = [exe+1, kn_ids]
    _item.to_csv(save_dir+'/'+'item.csv', index=False)

    # --- train.csv, valid.csv, test.csv ---
    # NOTE: all IDs start from 1
    _train_data = pd.DataFrame(columns=['user_id', 'item_id', 'score'])
    _valid_data = pd.DataFrame(columns=['user_id', 'item_id', 'score'])
    _test_data = pd.DataFrame(columns=['user_id', 'item_id', 'score'])
    _del_st_num = 0
    for stu in range(st_num):
        if ex_num - len(np.argwhere(np.isnan(stu_exe[stu]))) < 3:
            _del_st_num += 1
            continue
        stu_logs = []
        _ = 0
        for exe in range(ex_num):
            if np.isnan(stu_exe[stu][exe]):continue
            log = {'user_id':int(stu+1), 'item_id':int(exe+1), 'score':stu_exe[stu][exe]}
            stu_logs.append(log)
            _ += 1
        random.shuffle(stu_logs)
        t_v_n = int(divide_ratio * _)  # the number of train and valid
        s_n = _ - t_v_n  # the number of test
        t_n = int(t_v_n * 0.8)  # the number of train
        v_n = t_v_n - t_n  # the number of test

        train_logs = stu_logs[: t_n]
        valid_logs = stu_logs[t_n: t_n+v_n]
        test_logs = stu_logs[t_n+v_n :]

        for _ in range(len(train_logs)):
            _train_data.loc[_train_data.shape[0]] = train_logs[_]
        for _ in range(len(valid_logs)):
            _valid_data.loc[_valid_data.shape[0]] = valid_logs[_]
        for _ in range(len(test_logs)):
            _test_data.loc[_test_data.shape[0]] = test_logs[_]

    _train_data.to_csv(save_dir+'/'+'train.csv', index=False)
    _valid_data.to_csv(save_dir+'/'+'valid.csv', index=False)
    _test_data.to_csv(save_dir+'/'+'test.csv', index=False)

    return st_num-_del_st_num, ex_num, kn_num