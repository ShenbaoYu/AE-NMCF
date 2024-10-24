import numpy as np
import json
import os
import logging
import warnings
import sys
import time

import DINA
import data_pre



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from SidePackage import auxiliary as aux

DATASET = input("\nplease choose a dataset: [FrcSub, Quanlang-s, Junyi-s, unit-eng, unit-bio-small, unit-his-small]: ")
print("dataset %s is choosed" % DATASET)
if DATASET not in ['FrcSub', 'Quanlang-s', 'Junyi-s', 'unit-eng', 'unit-bio-small', 'unit-his-small']:
    warnings.warn("dataset does not exist.")
    exit()

train_ratio = 0.8
valid_ratio = 0

# divdide the raw data (student-exercise matrix) into traning data and testing data
# divdide the raw data (student-exercise matrix) into traning data and testing data
is_divide = input("re-divide the dataset?(yes or no): ")
if is_divide == "yes":
    data_pre.out_files(DATASET, train_ratio, valid_ratio)
elif is_divide == "no":
        pass
else:
    warnings.warn("illegal input!")
    exit()

FILE_PATH = BASE_DIR + "/DINA/data/" + DATASET

q_m = np.loadtxt(FILE_PATH + "/q_m.csv", dtype=int, delimiter=',')
prob_num, know_num = q_m.shape[0], q_m.shape[1]

with open(FILE_PATH + "/train_data.json", encoding='utf-8') as file:
    train_set = json.load(file)

stu_num = max([x['user_id'] for x in train_set]) + 1
# stu_num = len(set([x['user_id'] for x in train_set]))

# row: student
# col: exercise
R = -1 * np.ones(shape=(stu_num, prob_num))
for log in train_set:
    R[log['user_id'], log['item_id']] = log['score']

with open(FILE_PATH + "/test_data.json", encoding='utf-8') as file:
    test_set = json.load(file)

logging.getLogger().setLevel(logging.INFO)

cdm = DINA.DINA(R, q_m, stu_num, prob_num, know_num, skip_value=-1)

start = time.time()
cdm.train(epoch=20, epsilon=1e-3)
end = time.time()
print('TIME:%.5f' %(end - start))
cdm.save(BASE_DIR + "/DINA/dina.params")

R_true = np.loadtxt(BASE_DIR + '/Data/' + DATASET + '/data.txt')
# problem-description
if os.path.exists(BASE_DIR + "/Data/" + DATASET + "/problemdesc.txt"):
    prob_desc = aux.read_problem_desc(BASE_DIR + "/Data/" + DATASET + "/problemdesc.txt")    
else:
    prob_desc = aux.build_problem_desc(R.T)

cdm.load(BASE_DIR + "/DINA/dina.params")
rmse, mae, acc, krc = cdm.eval(test_set, prob_desc, R_true.T)
print("rmse: %.5f, accuracy: %.5f, KRC: %.5f" % (rmse, acc, krc))