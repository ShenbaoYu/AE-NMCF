import numpy as np
import math

np.set_printoptions(threshold=np.inf)




class vector():
    
    def __init__(self, plm_intro, q_matrix, arr_train, raw_data):
        self.plm_intro = plm_intro
        self.arr_train = arr_train
        self.q_matrix = q_matrix
        self.raw_data = raw_data

        self.index = 5
        
        self.arr_temp = np.nan * np.ones([self.plm_intro.shape[0], 2])


    # TODO: the problem difficulty
    def plm_difficulty(self):

        self.li_dft_all = []
        for self.p_count in range(self.arr_train.shape[1]):
            self.plm_tcount = 0
            self.plm_acount = 0
            for self.s_count in range(self.arr_train.shape[0]):
                # if self.arr_train[self.s_count][self.p_count] >= self.plm_intro[self.p_count][1]/2:
                #     self.plm_tcount += 1
                # if self.arr_train[self.s_count][self.p_count] >= 0:
                #     self.plm_acount += 1

                if np.isnan(self.arr_train[self.s_count][self.p_count]):
                    continue
                else:
                    self.plm_acount += 1
                    if self.raw_data[self.s_count][self.p_count] >= self.plm_intro[self.p_count][1]/2:
                        self.plm_tcount += 1
            
            if self.plm_acount == 0:
                self.plm_d = 1
            else:
                self.plm_d = (self.plm_acount-self.plm_tcount) / self.plm_acount  # false_num / all_num

            self.li_dft = self.get_difficulty_vec(self.plm_d)
            self.li_dft_all += self.li_dft

            self.arr_temp[self.p_count][0] = self.plm_tcount
            self.arr_temp[self.p_count][1] = self.plm_acount

        self.plm_dft = np.array(self.li_dft_all)
        self.plm_dft = np.reshape(self.plm_dft, (self.arr_train.shape[1],len(self.li_dft)))
        
        # [obj or sub; full score; plm_difficulty]
        # - keywords (the vectorization without the keywords information)
        self.plm_vec = np.append(self.plm_intro, self.plm_dft, axis=1)
        return self.plm_vec  # only considering the problem difficulty


    #TODO: The skill difficulty
    def skl_difficulty(self):
        self.li_dft_all = []
        for self.k_count in range(self.q_matrix.shape[1]):
            self.skl_tcount = 0
            self.skl_acount = 0
            for self.p_count in range(self.q_matrix.shape[0]):
                if self.q_matrix[self.p_count][self.k_count] == 1:
                    self.skl_tcount += self.arr_temp[self.p_count][0]
                    self.skl_acount += self.arr_temp[self.p_count][1]
            self.skl_d = (self.skl_acount-self.skl_tcount) / self.skl_acount

            self.li_dft = self.get_difficulty_vec(self.skl_d)
            self.li_dft_all += self.li_dft

        self.skl_dft = np.array(self.li_dft_all)
        self.skl_dft = np.reshape(self.skl_dft,(self.q_matrix.shape[1], len(self.li_dft)))

        # [skl_difficulty]
        # - keywords (the vectorization without the keywords information)
        return self.skl_dft  # only considering the skill difficulty


    def get_difficulty_vec(self, dft):
        li_dft = []

        for i in range(1,self.index):
            self.interval = 1/math.pow(2,i)
            lvalue = 0
            for j in range(int(math.pow(2,i))):
                if j == 0 and dft == 0:
                    li_dft.append(1)
                elif  lvalue < dft <= lvalue + self.interval:
                    li_dft.append(1)
                else:
                    li_dft.append(0)
                lvalue += self.interval
        
        return li_dft