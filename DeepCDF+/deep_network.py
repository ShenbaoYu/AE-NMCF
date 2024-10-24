# NOTE: the version of the tensorfolow: V1 (1.3.0)
#       the version of the keras: 2.0.6

import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Layer
from keras.models import Model
from keras.layers import Layer, Lambda
from keras import regularizers
from keras.utils import plot_model
from keras import backend as K

import numpy as np


class network_main():

    def __init__(self, vec_skl, vec_plm, q_matrix, skl_pfc, arr_train, arr_test,
                 hidden_dim, batch_size, reg_l2, acitvation, plm_intro):
        
        self.vec_skl = vec_skl
        self.tensor_skl = tf.constant(value=self.vec_skl, dtype=tf.float32) # [skl_num, embed_dim]
        self.vec_plm = vec_plm[:, 2:]
        self.tensor_plm = tf.constant(value=self.vec_plm, dtype=tf.float32) # [plm_num, embed_dim]
        self.q_matrix = q_matrix
        self.tensor_q_matrix = tf.constant(value=self.q_matrix, dtype=tf.float32)
        self.skl_pfc = skl_pfc  # skill proficiency
        self.arr_train = arr_train
        self.arr_test = arr_test
        self.arr_train_temp = arr_train.copy()
        self.arr_test_temp = arr_test.copy()
        self.skl_num = self.vec_skl.shape[0]
        self.plm_num = self.vec_plm.shape[0]
        self.exm_num = self.skl_pfc.shape[0]
        self.embed_dim = self.vec_skl.shape[1]
        self.hidden_dim = hidden_dim  # parameters dimension in Attention Layer
        self.batch_size = batch_size
        self.l2 = reg_l2
        self.acitvation = acitvation

        # after delete the nan; [exm_num, plm_num], some are set to 0
        self.train_mask_fit = self.get_mask(self.arr_train_temp)
        self.arr_train_fit = self.get_sets(self.arr_train)
        self.test_mask_fit = self.get_mask(self.arr_test_temp)  # after delete the nan
        self.arr_test_fit = self.get_sets(self.arr_test)

        # problem requires skills; [[skls that plm1 requires], [skls that plm2 requires], ...]
        self.li_gate = self.get_li_gate()

        # TODO: skill interaction
        self.li_indices1, self.li_indices2 = self.get_indices()
        self.itr_stack = self.get_itr_stack()  # [exm_num, plm_num*embed_dim]


    def get_skl_pfc(self, dim):
        self.skl_pfc0 = np.tile(self.skl_pfc, (1, dim))
        self.skl_pfc0 = np.reshape(self.skl_pfc0, (self.exm_num*dim, self.skl_num))
        return self.skl_pfc0


    def get_mask(self, sets):
        self.mask_fit = sets
        self.mask_fit[np.logical_not(np.isnan(self.mask_fit))] = 1
        self.mask_fit[np.isnan(self.mask_fit)] = 0  # [exm_num, plm_num]
        return self.mask_fit


    def get_sets(self, sets):
        self.sets = sets
        self.sets[np.isnan(self.sets)] = 0  # [exm_num, plm_num]; nan->0
        return self.sets


    def get_li_gate(self):
        self.li_gate = []
        for i in range(self.q_matrix.shape[0]):
            self.li_gate_iner = []
            for j in range(self.q_matrix.shape[1]):
                if self.q_matrix[i][j] != 0:
                    self.li_gate_iner.append(j)
            self.li_gate.append(self.li_gate_iner)
        return self.li_gate


    def get_indices(self):
        self.li_indices1 = []
        self.li_indices2 = []
        for i in self.li_gate:
            self.li_indices_temp1 = []
            self.li_indices_temp2 = []

            if len(i) == 1:
                self.li_indices_temp1.append(i[0])
                self.li_indices_temp2.append(i[0])
            else:
                for j in i:  
                    for k in i[i.index(j) + 1:]:
                        self.li_indices_temp1.append(j)
                        self.li_indices_temp2.append(k)
            self.li_indices1.append(self.li_indices_temp1)
            self.li_indices2.append(self.li_indices_temp2)
        return self.li_indices1, self.li_indices2


    def slice(self, x, index):
        return x[:, index:]


    def get_dft(self, x, index):
        return x[:, index]


    def get_itr_stack(self):
        # the itr_stack of each plm
        for i in range(self.plm_num):
            self.itr_arr1 = self.vec_skl[self.li_indices1[i], :]  # [len, emd_num]; len: itr times
            self.itr_arr2 = self.vec_skl[self.li_indices2[i], :]
            self.itr_mul = np.multiply(self.itr_arr1, self.itr_arr2)  # [len, emd_num]
            self.itr_mean = np.mean(self.itr_mul, axis=0)  # [1, emd_num]
            if i == 0:
                self.itr = self.itr_mean
            else:
                self.itr = np.append(self.itr, self.itr_mean)
        self.itr = np.reshape(self.itr, (self.plm_num,self.embed_dim))  # [plm_num, emd_num]

        self.itr_stack = np.tile(self.itr, (self.exm_num, 1))  # [exm_num, plm_num*emd_num]
        self.itr_stack = np.reshape(self.itr_stack, (self.exm_num, self.plm_num*self.embed_dim))
        return self.itr_stack


    def get_itr_tensor(self, inputs):
        x, y = inputs
        return tf.multiply(x, y)

    def get_plm_pfc(self, inputs):
        x, y = inputs
        return tf.add(x, y)


    def loss(self, y_pred, y_true):
        self.sum1 = tf.reduce_sum(K.square(y_pred - y_true))
        self.count1 = tf.math.count_nonzero(self.mask_input)  
        return tf.divide(self.sum1, tf.cast(self.count1, tf.float32))


    def rmse(self, y_pred, y_true):
        self.sum1 = tf.reduce_sum(K.square(y_pred - y_true))
        self.count1 = tf.count_nonzero(self.mask_input)
        return K.sqrt(tf.divide(self.sum1,tf.add(tf.cast(self.count1,tf.float32),tf.cast(1e-5,tf.float32))))


    def mae(self, y_pred, y_true):
        self.sum1 = tf.reduce_sum(K.abs(y_pred - y_true))
        self.count1 = tf.count_nonzero(self.mask_input)
        return tf.divide(self.sum1,tf.add(tf.cast(self.count1,tf.float32),tf.cast(1e-5,tf.float32)))


    def model(self, data, epoch):
        # TODO: attention [2]
        self.skl_pfc_input = Input(shape=(self.skl_num,), name='skill_proficiency_input')
        self.mask_input = Input(shape=(self.plm_num,), name='mask_input')

        if data in ['FrcSub']:
            # dense scoring matrix  
            # Attention
            self.plm_pfc_all = Attention(self.li_gate, self.hidden_dim, self.embed_dim, self.tensor_skl, self.tensor_plm, 
                                         name='attention_problem_proficiency',
                                         kernel_regularizer=regularizers.l2(self.l2))([self.skl_pfc_input,self.mask_input])  # [batch_size, plm_num]
        elif data in ['Quanlang-s', 'Junyi-s', 'unit-eng', 'unit-bio-small', 'unit-his-small']:
            self.plm_pfc_all = Weight(self.li_gate, self.plm_num, self.skl_num,self.tensor_q_matrix,name='Weight',
                                      kernel_regularizer=regularizers.l2(self.l2))([self.skl_pfc_input,self.mask_input])  # [batch_size,plm_num]

        # TODO: skill interaction [2]
        self.itr_tensor_input = Input(shape=(self.embed_dim*self.plm_num,),
                                      name='interaction_tensor_input')  # [batch_size, self.embed_dim*self.plm_num]
        # =>[batch_size,plm_num]; activation = softmax,tanh,relu,sigmoid,linear
        self.itr_tensor = Dense(self.plm_num, input_shape=(self.plm_num * self.embed_dim,),
                                name='skill_interaction', activation=self.acitvation,
                                kernel_regularizer=regularizers.l2(self.l2))(self.itr_tensor_input)  
        self.itr_tensor = Lambda(self.get_itr_tensor,
                                 name='itr_tensor')([self.itr_tensor,self.mask_input])  # [batch_size, plm_num]

        # TODO: problem proficiency [2]
        # [exm_num,plm_num]
        # + inter
        self.plm_pfc = Lambda(self.get_plm_pfc,
                              name='problem_proficiency')([self.plm_pfc_all,self.itr_tensor])  # [batch_size, plm_num]
        # - inter
        # self.plm_pfc = self.plm_pfc_all

        # TODO: grade prediction[3]
        # divide_num changes
        # + guess and slip
        self.exm_grade_output = GradePrediction(plm_num=self.plm_num, batch_size=self.batch_size,
                                                name='grade_prediction',
                                                kernel_regularizer=regularizers.l2(self.l2))([self.plm_pfc,self.mask_input])  # [batch_size, plm_num]
        # - guess and slip
        # self.exm_grade_output=self.plm_pfc

        # TODO: model
        model = Model(inputs=[self.itr_tensor_input,
                              self.skl_pfc_input, self.mask_input], outputs=[self.exm_grade_output])
        model.compile(loss=self.loss, optimizer='Adam', metrics=[self.rmse, self.mae])

        # from keras.callbacks import ReduceLROnPlateau
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
        callback = EarlyStopping(monitor='loss', patience=5)
        model.fit([self.itr_stack, self.skl_pfc, self.train_mask_fit], [self.arr_train], epochs=epoch, 
                  batch_size=self.batch_size, verbose=1, callbacks=[callback])  # verbose=1 show the results for each epach

        outputTensor = model.output
        variableTensors = model.trainable_weights
        gradients = K.gradients(outputTensor,variableTensors)
        # print(model.summary())
        # plot_model(model, to_file='model.jpg', dpi=600)
        # with open('log.txt', 'a+', encoding='utf-8') as f:
        #     f.write('------- outputTensor -------')
        #     f.write(outputTensor)
        #     f.write('------- variableTensors -------')
        #     f.write(variableTensors)
        #     f.write('------- gradients -------')
        #     f.write(gradients)

        # TODO: test
        loss, rmse, mae = model.evaluate([self.itr_stack, self.skl_pfc, self.test_mask_fit],
                                         [self.arr_test], batch_size=self.batch_size, verbose=1)

        stu_pre = model.predict([self.itr_stack, self.skl_pfc, self.test_mask_fit], batch_size=self.batch_size)

        return rmse, mae, stu_pre


class Attention(Layer):

    def __init__(self, li_gate, hidden_dim, concat_dim, tensor_skl, tensor_plm, kernel_regularizer=None, **kwargs):
        self.li_gate = li_gate
        self.hidden_dim = hidden_dim
        self.concat_dim = concat_dim  # q_dim
        self.skl_emd =tensor_skl
        self.plm_emd = tensor_plm
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        super(Attention, self).__init__(**kwargs)


    def build(self, input_shape):

        self.v_T = self.add_weight(name='concat_v_T'.format(self.name),
                                   shape=(1, self.hidden_dim), 
                                   initializer='random_uniform', trainable=True, regularizer=self.kernel_regularizer)
        self.W = self.add_weight(name='concat_W'.format(self.name),
                                 shape=(self.hidden_dim,self.concat_dim),
                                 initializer='random_uniform', trainable=True, regularizer=self.kernel_regularizer)
        self.U = self.add_weight(name='concat_U'.format(self.name),
                                 shape=(self.hidden_dim,self.concat_dim),
                                 initializer='random_uniform', trainable=True, regularizer=self.kernel_regularizer)
        super(Attention, self).build(input_shape)


    def call(self, x, mask=None):
        self.skl_pfc, self.tensor_mask = x  # [batch_size, skl_num]

        for i in self.li_gate:
            # a problem
            self.li_rq_skl = i
            self.one_plm_emd = self.plm_emd[self.li_gate.index(i)]
            self.one_plm = tf.transpose(tf.tile([self.one_plm_emd], [len(self.li_rq_skl),1]))  # [q_dim, rq_num]
            self.rq_skl = tf.transpose(tf.gather(self.skl_emd, axis=0, indices=self.li_rq_skl))  # [k_dim, rq_num]

            self.align_temp = tf.tanh(K.dot(self.W,self.one_plm) + K.dot(self.U,self.rq_skl))
            self.align = K.dot(self.v_T,self.align_temp)  # [1, rq_num]
            self.attention = tf.nn.softmax(self.align)   # [1, rq_num]

            self.one_skl_pfc = tf.gather(self.skl_pfc, axis=1, indices=self.li_rq_skl)  # [batch_size, rq_num]
            self.plm_pfc1 = K.dot(self.one_skl_pfc, tf.transpose(self.attention))   # [batch_size, 1]

            if self.li_gate.index(i) == 0:
                self.plm_pfc_all1 = self.plm_pfc1
            else:
                self.plm_pfc_all1 = K.concatenate([self.plm_pfc_all1, self.plm_pfc1], -1)  # [batch_size, plm_num]

        self.plm_pfc_all1 = tf.multiply(self.plm_pfc_all1, self.tensor_mask)

        return self.plm_pfc_all1  # [batch_size, plm_num]


    def compute_output_shape(self, input_shape):
        return (self.plm_pfc_all1.shape[0], self.plm_pfc_all1.shape[1])


class Weight(Layer):

    def __init__(self,li_gate,plm_num,skl_num,q_matrix,kernel_regularizer = None,**kwargs):
        self.li_gate = li_gate
        self.plm_num =plm_num
        self.skl_num = skl_num  # q_dim
        self.tensor_q_matrix = q_matrix
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        super(Weight, self).__init__(**kwargs)

    def build(self,input_shape):

        self.w = self.add_weight(name='w'.format(self.name), shape=(self.skl_num, self.plm_num), initializer='random_uniform',trainable=True,regularizer=self.kernel_regularizer)
        super(Weight, self).build(input_shape)

    def call(self,x,mask = None):
        self.skl_pfc,self.tensor_mask = x   # [batch_size,skl_num]
        self.plm_pfc_all = K.dot(self.skl_pfc,(tf.multiply(self.w, tf.transpose(self.tensor_q_matrix))))
        self.plm_pfc_all1 = tf.multiply(self.plm_pfc_all, self.tensor_mask)

        return self.plm_pfc_all1  # [batch_size,plm_num]

    def compute_output_shape(self,input_shape):
        return (self.plm_pfc_all1.shape[0],self.plm_pfc_all1.shape[1])



class GradePrediction(Layer):

    def __init__(self, plm_num, batch_size, kernel_regularizer=None, **kwargs):
        self.plm_num = plm_num  # [plm_num]
        self.batch_size = batch_size
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        super(GradePrediction, self).__init__(**kwargs)


    def build(self, input_shape):
        # [plm_num,1]; each problem has a slip and a guess
        self.B1 = self.add_weight(name='slip_B1'.format(self.name), 
                                  shape=(1,self.plm_num), 
                                  initializer='random_uniform', trainable=True,regularizer=self.kernel_regularizer)
        self.B2 = self.add_weight(name='guess_B2'.format(self.name),
                                  shape=(1,self.plm_num),
                                  initializer='random_uniform', trainable=True,regularizer=self.kernel_regularizer)


    def call(self, x, mask=None):
        # B1 and B2: plm_num->batch_size
        self.plm_pfc, self.tensor_mask = x
        self.ones = tf.ones([self.batch_size,self.plm_num], tf.float32)
        self.B1 = tf.tile(self.B1, [self.batch_size,1])
        self.B2 = tf.tile(self.B2, [self.batch_size, 1])

        self.exm_grade = tf.multiply(tf.multiply(self.plm_pfc,(self.ones-self.B1)), 
                                     self.tensor_mask) + \
        tf.multiply(tf.multiply((self.ones-self.plm_pfc),self.B2),self.tensor_mask)  # [batch_size,1]
        return self.exm_grade


    def compute_output_shape(self, input_shape):
        return (self.exm_grade.shape[0], self.exm_grade.shape[1])