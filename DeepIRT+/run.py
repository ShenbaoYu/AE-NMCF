from moduel.LSTM import LSTM
from data.loader import DataLoader
from model.DeepModel import DeepIRT
import numpy as np
import pickle
import torch.optim as optim
import eval
import warnings
import os, sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from SidePackage import evaluation as ev

lstm = LSTM(50, 5)

# NOTE: 
# u:student; q: question; res: response; k: knowledge concept
# topicId: question ID

def preparing(path):

    # load training and testing data
    train_uid_qid_res = pickle.load(open(path + '/pkl_data/train_uid_qid_res.pkl', 'rb'))
    test_uid_qid_res = pickle.load(open(path + '/pkl_data/test_uid_qid_res.pkl', 'rb'))

    # load embedding
    qid_emb = pickle.load(open(path + '/pkl_data/qid_emb_data.pkl', 'rb'))
    kcode_emb = pickle.load(open(path + '/pkl_data/code_emb_data.pkl', 'rb'))
    qid_kcode = pickle.load(open(path + '/pkl_data/qid_kcode_data.pkl', 'rb'))

    trainLoader, testLoader, uid_index_dict, qid_index_dict, kcode_index_dict \
        = DataLoader(train_uid_qid_res, test_uid_qid_res, qid_emb, kcode_emb, qid_kcode)
    kn_num = len(kcode_emb)  # number of knowledge concepts
    st_num = len(set(train_uid_qid_res.loc[:, 'userId'].tolist()))  # number of students
    
    return trainLoader, testLoader, kn_num, st_num, uid_index_dict, qid_index_dict, kcode_index_dict

def cognitive_diagnosis_ev(plm_intro, arr_data, arr_valid, skl_pfc, q_matrix):

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




if __name__ == '__main__':

    hyper1 = 30 # dnn_hidden_size [10,20,30,40]
    print('dnn_hidden_size:', hyper1)

    DATA = input("\nplease choose a dataset: [FrcSub, Quanlang-s, Junyi-s, unit-eng, unit-bio-small, unit-his-small]: ")
    print("dataset %s is choosed" % DATA)
    if DATA not in ['FrcSub', 'Quanlang-s', 'Junyi-s', 'unit-eng', 'unit-bio-small', 'unit-his-small']:
        warnings.warn("dataset does not exist.")
        exit()
    path = BASE_DIR + '/DeepIRT+/data/' + DATA + '/'
    
    arr_data = np.loadtxt(path + 'data.txt')
    arr_train = np.loadtxt(path + 'divide_data/train_data.txt')
    arr_valid = np.loadtxt(path + 'divide_data/test_data.txt')
    q_matrix = np.loadtxt(path + 'q.txt')
    plm_intro = np.loadtxt(path + 'problemdesc.txt')

    train, test, kn_num, st_num, uid_index_dict, qid_index_dict, kcode_index_dict = preparing(path)
    
    start = time.time()
    model = DeepIRT(lstm_input_size=50, lstm_hidden_size=1, dnn_input_size=50, dnn_hidden_size=hyper1,
                    dnn_output_size=1, knowledge_nums=kn_num, student_nums=st_num, denseDim=50)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = eval.myLoss()
    model, optimizer = eval.train_epoch(model, train, optimizer, loss_func)

    acc_obj, rmse, mae, stu_kn_pro = eval.test_epoch(DATA, model, test, loss_func,
                                                     uid_index_dict, kcode_index_dict, plm_intro, q_matrix)
    if not stu_kn_pro is None:
        krc = cognitive_diagnosis_ev(plm_intro, arr_data, arr_valid, stu_kn_pro, q_matrix)
    else:
        krc, rho = None, None
    
    print('acc_obj', acc_obj)
    print('rmse:', rmse)
    print('krc:', krc)
    end = time.time()
    print('TIME:%.5f' %(end - start))