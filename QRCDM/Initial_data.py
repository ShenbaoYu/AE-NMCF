import pandas as pd
import numpy as np
import torch


class DataSet():
    def __init__(self, basedir, dataSetName):
        self.basedir = basedir
        self.dataSetName = dataSetName
        
        if dataSetName == 'FrcSub':
            read_dir = basedir + '/data/FrcSub/'
            save_dir = basedir + '/output/FrcSub/'
            n = 536
            m = 20
            k = 8
        elif dataSetName == 'Quanlang-s':
            read_dir = basedir + '/data/Quanlang-s/'
            save_dir = basedir + '/output/Quanlang-s/'
            n = 50
            m = 107
            k = 14
        elif dataSetName == 'Junyi-s':
            read_dir = basedir + '/data/Junyi-s/'
            save_dir = basedir + '/output/Junyi-s/'
            n = 393
            m = 9
            k = 9
        elif dataSetName == 'unit-eng':
            read_dir = basedir + '/data/unit-eng/'
            save_dir = basedir + '/output/unit-eng/'
            n = 360
            m = 362
            k = 19
        elif dataSetName == 'unit-bio-small':
            read_dir = basedir + '/data/unit-bio-small/'
            save_dir = basedir + '/output/unit-bio-small/'
            n = 100
            m = 129
            k = 7
        elif dataSetName == 'unit-his-small':
            read_dir = basedir + '/data/unit-his-small/'
            save_dir = basedir + '/output/unit-his-small/'
            n = 1057
            m = 326
            k = 14
        else:
            print('Dataset does not exist!')
            exit(0)
        
        item = pd.read_csv(read_dir + "item.csv")

        train_data = pd.read_csv(read_dir + "train.csv").set_index('user_id')
        valid_data = pd.read_csv(read_dir + "valid.csv").set_index('user_id')
        test_data = pd.read_csv(read_dir + "test.csv").set_index('user_id')

        if dataSetName in ('FrcSub', 'Quanlang-s', 'Junyi-s', 'unit-eng', 'unit-bio-small', 'unit-his-small'):
            obj_prob_index = 'All'
            sub_prob_index = None
        else:
            # type of problems
            obj_prob_index = np.loadtxt(read_dir + "obj_prob_index.csv", delimiter=',', dtype=int)
            sub_prob_index = np.loadtxt(read_dir + "sub_prob_index.csv", delimiter=',', dtype=int)

        self.total_stu_list = set(train_data.index) & set(valid_data.index) & set(test_data.index)

        self.stu_num = n
        self.prob_num = m
        self.skill_num = k
        self.item = item
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.save_dir = save_dir
        self.obj_prob_index = obj_prob_index
        self.sub_prob_index = sub_prob_index

    def get_Q(self):
        Q = np.zeros((self.prob_num, self.skill_num), dtype='bool')
        item = self.item
        for idx in item.index:
            item_id = item.loc[idx, 'item_id']
            know_list = item.loc[idx, 'knowledge_code'].replace(
                '[', '').replace(']', '').split(',')
            for know in know_list:
                Q[item_id - 1, int(know) - 1] = True
        return torch.tensor(Q, dtype=torch.float)
