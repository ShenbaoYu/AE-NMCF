# divide datasets; get key words

import numpy as np
import random
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from SidePackage import preprocessing as pre




# TODO: get the datasets
class data_get():
    
    def __init__(self, path):
        self.path = path
        self.arr_data = np.loadtxt(self.path + 'data.txt')
        self.raw_data = np.loadtxt(self.path + 'rawdata.txt')
        self.plm_intro = np.loadtxt(self.path + 'problemdesc.txt')
        self.q_matrix = np.loadtxt(self.path + 'q.txt')


    def divide_dense(self):

        exe_num = self.arr_data.shape[1]  # get the number of exercises
        divide_ratio = [0.8, 0, 0.2]  # the ratio of traning / validation / testing
        self.divide_num = [int(i*exe_num) for i in divide_ratio]

        # initialize
        # the shape of arr_train is same as that of arr_data
        self.arr_train = np.nan * np.ones(np.shape(self.arr_data))
        # the shape of arr_vali is same as that of arr_data
        self.arr_vali = np.nan * np.ones(np.shape(self.arr_data))
        # the shape of arr_test is same as that of arr_data
        self.arr_test = np.nan * np.ones(np.shape(self.arr_data))
        self.arr_train_vali = np.nan * np.ones(np.shape(self.arr_data))
        # insure each set has at least one student's data
        self.count = np.isnan(self.arr_train).sum()
        
        for self.exm_count in range(self.arr_data.shape[0]):

            self.li_ran = [_ for _ in range(exe_num)]
            self.slice1 = random.sample(self.li_ran, self.divide_num[0])
            self.li_ran2 = list(set(self.li_ran) - set(self.slice1))
            self.slice2 = random.sample(self.li_ran2, self.divide_num[1])
            self.slice2 = self.slice2
            self.slice3 = list(set(self.li_ran2) - set(self.slice2))
            self.slice4 = self.slice1+self.slice2

            for self.slice1_count in self.slice1:
                self.arr_train[self.exm_count][self.slice1_count] = self.arr_data[self.exm_count][self.slice1_count]
            for self.slice2_count in self.slice2:
                self.arr_vali[self.exm_count][self.slice2_count] = self.arr_data[self.exm_count][self.slice2_count]
            for self.slice3_count in self.slice3:
                self.arr_test[self.exm_count][self.slice3_count] = self.arr_data[self.exm_count][self.slice3_count]
            for self.slice4_count in self.slice4:
                self.arr_train_vali[self.exm_count][self.slice4_count] = self.arr_data[self.exm_count][self.slice4_count]
        
        np.savetxt(self.path + 'divide_data/train_data.txt', self.arr_train_vali)
        np.savetxt(self.path + 'divide_data/test_data.txt', self.arr_test)
    

    def divide_sparse(self):
        # initialize
        # # the shape of arr_train is same as that of arr_data
        # self.arr_train = np.nan * np.ones(np.shape(self.arr_data))
        # # the shape of arr_test is same as that of arr_data
        # self.arr_test = np.nan * np.ones(np.shape(self.arr_data))

        # for stu in range(self.arr_data.shape[0]):
        #     # get the index of the records that are not NaN
        #     _index_notnan = np.where(np.logical_not(np.isnan(self.arr_data[stu])))[0]
        #     random.shuffle(_index_notnan)
        #     _index_train = list(_index_notnan[0: int(self.divide_ratio[0] * len(_index_notnan))])
        #     _index_test = list(_index_notnan[int(self.divide_ratio[0] * len(_index_notnan)) :])
            
        #     for _ in _index_train:
        #         self.arr_train[stu][_] = self.arr_data[stu][_]
        #     for _ in _index_test:
        #         self.arr_test[stu][_] = self.arr_data[stu][_]

        divide_ratio = [0.8, 0, 0.2]  # the ratio of traning / validation / testing
        self.arr_train, test_loc = pre.missing_stu_exe(self.arr_data.T, divide_ratio[0])
        
        self.arr_test = np.nan * np.ones(np.shape(self.arr_data))
        for _ in test_loc:
            exe, stu = _[0], _[1]
            self.arr_test[stu][exe] = self.arr_data[stu][exe]

        np.savetxt(self.path + 'divide_data/train_data.txt', self.arr_train.T)
        np.savetxt(self.path + 'divide_data/test_data.txt', self.arr_test)
    

    def load_data(self):

        self.arr_train = np.loadtxt(self.path + 'divide_data/train_data.txt')
        self.arr_valid = np.loadtxt(self.path + 'divide_data/test_data.txt')

        return self.raw_data, self.arr_data, self.plm_intro, self.q_matrix, self.arr_train, self.arr_valid