"""
Lina Gao, Zhongying Zhao, Chao Li, Jianli Zhao, Qingtian Zeng.
Deep cognitive diagnosis model for predicting students' performance.
Future Generation Computer Systems, 126, 252-262, 2022.
"""


import time
import warnings
import os
import sys
import numpy as np

from data_preprocessing import data_get
from vector_representation import vector
from skill_proficiency import skl_proficiency
from deep_network import network_main

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from SidePackage import evaluation as ev

warnings.filterwarnings("ignore")




# TODO: data processing
class pro_data():
    
    def get_data(self, data, is_divide):
        
        self.cl_dd = data_get(path)

        if is_divide == 'y':
            # divide data
            if data in ['FrcSub']:
                self.cl_dd.divide_dense()
            elif data in ['Quanlang-s', 'Junyi-s', 'unit-eng', 'unit-bio-small', 'unit-his-small']:
                self.cl_dd.divide_sparse()
            else:
                warnings.warn("dataset does not exist.")
                exit()

        self.raw_data, self.arr_data, self.plm_intro, self.q_matrix, self.arr_train, self.arr_valid \
        = self.cl_dd.load_data()
        
        return self.raw_data, self.arr_data, self.plm_intro, self.q_matrix, self.arr_train, self.arr_valid


# TODO: problem difficulty and skill difficulty
class vec_present():
    """
    FUNCTION:
    1. vectorization for each skill (knowledge concept)
    based on (a) the skill difficulty.
    
    2. vectorization for each problem (exercise)
    based on (a) problem types, (b) problem full score, (c) problem difficulty.
    """

    def get_vector(self):
        self.cl_v = vector(plm_intro, q_matrix, arr_train, raw_data)

        self.vec_skl = self.cl_v.skl_difficulty()
        self.vec_plm = self.cl_v.plm_difficulty()

        return self.vec_skl, self.vec_plm


# TODO: skill proficiency
class skl_master():
    
    def get_skl_proficiency(self):
        self.cl_sp = skl_proficiency(vec_plm, vec_skl, q_matrix, arr_train, hyper2)
        self.skl_pfc = self.cl_sp.get_proficiency()
        
        return self.skl_pfc


# TODO: problem proficiency and grade prediction
class network_keras():
    
    def get_grades(self, data, epoch):
        self.cl_el = network_main(vec_skl, vec_plm, q_matrix, skl_pfc, arr_train, 
                                  arr_valid, hyper3, hyper4, hyper5, activation, plm_intro)
        rmse, mae, stu_pre = self.cl_el.model(data, epoch)
        
        return rmse, mae, stu_pre


def cognitive_diagnosis_ev(plm_intro, arr_data, arr_train, arr_valid, skl_pfc, q_matrix):
    """
    FUNCTION: cognitive diagnosis evaluation
    metrics:
    KRC: Knowledge-Response Consistency Coefficient

    Inputs:
    -------
    :param plm_intro --> numpy.ndarray
        the problem description (obj:0, sub:1) and full score

    :param arr_data --> numpy.ndarray
        the orginial scoring matrix (normalized)
        row: students
        col: exercises

    :param arr_train --> numpy.ndarray
        the scoring matrix for training
        row: students
        col: exercises
    
    :param arr_valid --> numpy.ndarray
        the scoring matrix for testing
        row: students
        col: exercises

    :param skl_pfc --> numpy.ndarray
        the students' skill proficiency
        row: students
        col: skills

    :param q_matrix --> numpy.ndarray
        the Q-matrix
        row: exercises
        col: skills
    """

    # 1. get the problem description for each exercise
    prob_desc = {}
    for exe in range(plm_intro.shape[0]):
        if plm_intro[exe][0] == 0:
            prob_desc[exe] = 'Obj'
        else:
            prob_desc[exe] = 'Sub'
    
    # 2. get the location of the missing values in (training) scoring matrix
    test_loc = []
    for stu in range(arr_valid.shape[0]):
        for exe in range(arr_valid.shape[1]):
            if not np.isnan(arr_valid[stu][exe]):
                test_loc.append([exe, stu])
    
    # 3. calculate the KRC
    kn_krc_list = ev.cal_diag_krc(prob_desc, arr_data.T, test_loc, skl_pfc, q_matrix)
    krc = np.mean([x for x in kn_krc_list.values()])

    return krc


def calculate_accuracy(plm_intro, arr_data, arr_valid, stu_exe_pre):
    """
    FUNCTION: the accuracy of the predicted student performance on objective problems

    Inputs:
    -------
    :param plm_intro --> numpy.ndarray
        the problem description (obj:0, sub:1) and full score
    
    :param arr_data --> numpy.ndarray
        the original scoring matrix
        row: students
        col: exercises
    
    :param arr_valid --> numpy.ndarray
        the scoring matrix for testing
        row: students
        col: exercises

    :param stu_exe_pre --> numpy.ndarray
        the predicted scoring matrix
        row: students
        col: exercises
    """

    # 1. get the problem description for each exercise
    prob_desc = {}
    for exe in range(plm_intro.shape[0]):
        if plm_intro[exe][0] == 0:
            prob_desc[exe] = 'Obj'
        else:
            prob_desc[exe] = 'Sub'

    # 2. get the location of the missing values in (training) scoring matrix
    test_loc = []
    for stu in range(arr_valid.shape[0]):
        for exe in range(arr_valid.shape[1]):
            if not np.isnan(arr_valid[stu][exe]):
                test_loc.append([exe, stu])

    accuracy_obj = ev.cal_accuracy_obj(arr_data.T, stu_exe_pre.T, test_loc, prob_desc)
    
    return accuracy_obj


class Logger(object):
    
    def __init__(self, filename = "Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass



if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    DATA = input("\nplease choose a dataset: [FrcSub, Quanlang-s, Junyi-s, unit-eng, unit-bio-small, unit-his-small]: ")
    print("dataset %s is choosed" % DATA)
    if DATA not in ['FrcSub', 'Quanlang-s', 'Junyi-s', 'unit-eng', 'unit-bio-small', 'unit-his-small']:
        warnings.warn("dataset does not exist.")
        exit()
    path = BASE_DIR + '/DeepCDF+/data/' + DATA + '/'

    EPOCH = input("\nplease choose the training epoch:")  # do not specify, e.g., 50 for unit-bio-small
        
    """
    Hyper-parameters
    -------
    hyperp2:
        aIrtSlop (the discrimination)
        [0.5, 1, 1.2, 1.5, 1.7, 2, 2.5]
        
    hyperp3:
        hidden_dim (the dimension of weights)
        [5, 10, 15, 20, 25, 30, 35]
        
    hyperp4:
        batch_size
        [1, 3, 1403, 4209]
        
    hyperp5:
        reg_l2 (the regularization parameter)
        [0.005, 0.01, 0.03, 0.05, 0.1, 0.5, 1, 1.5]
        
    activation == 'tanh'
    """

    # hyperparameters
    hyper2 = 0.5
    hyper3 = 5
    hyper4 = 1
    hyper5 = 0.005

    activation = 'tanh'  # ['softmax', 'tanh', 'relu', 'sigmoid', 'linear']

    print('*** begin ***')
    print('the dataset is:', DATA)
    print('hyperp2-aIrtSlop:', hyper2)
    print('hyperp3-hidden_dim:', hyper3)
    print('hyperp4-batch_size:', hyper4)
    print('hyperp5-reg_l2:', hyper5)
    print('activation:', activation)

    # 1. get the data
    is_divide = input("re-divide the dataset? (y or n): ")
    cl_pd = pro_data()
    raw_data, arr_data, plm_intro, q_matrix, arr_train, arr_valid = cl_pd.get_data(DATA, is_divide)

    # print("start time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start = time.time()
                                    
    # 2. the vectorization (embedding) of the skills (knowledge concepts) and problems (exercises)
    cl_vd = vec_present()
    vec_skl, vec_plm = cl_vd.get_vector()

    # 3. the students' skill proficiency
    cl_sm = skl_master()
    skl_pfc = cl_sm.get_skl_proficiency()

    # 4. cognitive diagnosis evaluation
    krc = cognitive_diagnosis_ev(plm_intro, arr_data, arr_train, arr_valid, skl_pfc, q_matrix)

    # 5. problem proficiency and performance prediction
    train_data = arr_train.copy()
    test_data = arr_valid.copy()
    cl_nk = network_keras()
    rmse, mae, stu_pre = cl_nk.get_grades(DATA, int(EPOCH))

    # 6. calculate the accuracy for objective problems
    acc_obj = calculate_accuracy(plm_intro, arr_data, test_data, stu_pre)

    # sys.stdout = Logger(os.path.abspath(os.path.dirname(__file__)) + '/log.txt')
                    
    print('=== results (the data set is: %s) ===' % DATA)
    print('acc_obj', acc_obj)
    print('rmse:', rmse)
    print('krc:', krc)
    print('===============')

    # print("finish time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    end = time.time()
    print('TIME:%.5f' %(end - start))
    print('*** end ***')