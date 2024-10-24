# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
from scipy import stats

from pandas.core.frame import DataFrame


def cal_accuracy_obj(matrix_ori, matrix_pre, miss_coo, exe_desc):
    shape = matrix_ori.shape

    matrix_pre_round = matrix_pre.copy()
    for i in range(shape[0]):
        if exe_desc[i] == 'Sub':
            continue
        for j in range(shape[1]):
            matrix_pre_round[i][j] = round(matrix_pre_round[i][j])

    cor_rate_obj = 0
    miss_count_obj = 0

    for _ in miss_coo:
        if exe_desc[_[0]] == 'Obj':
            miss_count_obj += 1
            if matrix_pre_round[_[0]][_[1]] == matrix_ori[_[0]][_[1]]:
                cor_rate_obj += 1
    try:
        cor_rate_obj = cor_rate_obj / miss_count_obj
    except ZeroDivisionError:
        cor_rate_obj = None

    return cor_rate_obj


def cal_rmse(matrix_ori, matrix_pre, miss_coo):
    rmse = 0
    for _ in miss_coo:
        rmse += math.pow(abs(matrix_ori[_[0]][_[1]] - matrix_pre[_[0]][_[1]]), 2)
    rmse = math.sqrt(rmse / len(miss_coo))

    return rmse


def cal_diag_krc(exe_desc, matrix_ori, miss_coo, stu_kn_pro, q_matrix):
   
    kn_krc_list_obj = dict()
    kn_krc_list_sub = dict()
    kn_krc_list = dict()

    exe_num, kn_num = q_matrix.shape

    kn_exe_stu_pair = dict()
    for kn in range(kn_num):
        kn_exe_stu_pair[kn] = list()
    for _ in miss_coo:
        exe = _[0]
        for kn in range(kn_num):
            if q_matrix[exe][kn] == 1:
                kn_exe_stu_pair[kn].append(_)

    kn_stu_pro_ans_obj = dict()
    kn_stu_pro_ans_sub = dict()
    for kn in range(kn_num):
        kn_stu_pro_ans_obj[kn] = list()
        kn_stu_pro_ans_sub[kn] = list()

    for kn, pairs in kn_exe_stu_pair.items():
        for _ in pairs:
            exe = _[0]
            stu = _[1]
            pro = round(stu_kn_pro[stu][kn], 5)
            ans = matrix_ori[exe][stu]

            if exe_desc[exe] == 'Obj':
                kn_stu_pro_ans_obj[kn].append((pro, ans))
            elif exe_desc[exe] == 'Sub':
                kn_stu_pro_ans_sub[kn].append((pro, ans))

    for kn in range(kn_num):
        if not len(kn_stu_pro_ans_obj[kn]):
            kn_krc_list_obj[kn] = np.NaN
            continue
        kn_krc_list_obj[kn] = __cal_binary_krc(kn_stu_pro_ans_obj[kn], label=[0.0, 1.0])

    for kn in range(kn_num):
        pairs = kn_stu_pro_ans_sub[kn]

        if not len(pairs):
            kn_krc_list_sub[kn] = np.NaN
            continue

        stu_pro_ans = dict()
        for _ in pairs:
            try:
                stu_pro_ans[_[1]].append(_)
            except:
                stu_pro_ans[_[1]] = list()
                stu_pro_ans[_[1]].append(_)

        krc = 0
        count = 0
        labels = list(stu_pro_ans.keys())
        k = len(labels)
        for i in range(0, k):
            label_1 = labels[i]
            for j in range(i+1, k):
                label_2 = labels[j]
                _pro_ans = stu_pro_ans[label_1] + stu_pro_ans[label_2]
                krc += __cal_binary_krc(_pro_ans, [label_1, label_2])
                count += 1

        try:
            kn_krc_list_sub[kn] = krc / count
        except ZeroDivisionError:
            kn_krc_list_sub[kn] = np.NaN

    for kn in range(kn_num):
        kn_krc_obj = kn_krc_list_obj[kn]
        kn_krc_sub = kn_krc_list_sub[kn]

        if kn_krc_obj is not np.NaN and kn_krc_sub is not np.NaN:
            kn_krc_list[kn] = (kn_krc_obj + kn_krc_sub) / 2
        elif kn_krc_obj is np.NaN and kn_krc_sub is not np.NaN:
            kn_krc_list[kn] = kn_krc_sub
        elif kn_krc_sub is np.NaN and kn_krc_obj is not np.NaN:
            kn_krc_list[kn] = kn_krc_obj

    return kn_krc_list


def __cal_binary_krc(stu_pro_ans, label):
    rank = 0
    num_pos = 0
    num_neg = 0

    stu_pro_ans.sort(key=lambda x: x[0])
    for i in range(len(stu_pro_ans)):
        ans = stu_pro_ans[i][1]
        if ans == max(label):
            rank = rank + i + 1
            num_pos += 1
        elif ans == min(label):
            num_neg += 1
    if num_pos == 0:
        krc = 0
    elif num_neg == 0:
        krc = 1
    else:
        krc = (rank - num_pos * (num_pos + 1) / 2) / (num_pos * num_neg)

    return krc


def sort(arr):
    for i in range(0, len(arr)):
        for j in range(0, len(arr)-1):
            if arr[j][0] >= arr[j+1][0]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

    return arr