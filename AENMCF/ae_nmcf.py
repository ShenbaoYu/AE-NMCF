# -*- coding: utf-8 -*-
"""
the main module of the Autoencoder-like NMCF (AE-NMCF) framework
"""


import warnings

import armijo
import lipschitz



class ae_nmcf():
    """
    The AE-NMCF Model
    """

    def __init__(self, train_data, train_fill, q_m, weight, rank, gamma):
        """
        Inputs:
        -------
        :param train_data --> numpy.ndarray
            the student scoring matrix (including missing values)
    
        :param train_fill --> numpy.ndarray
            the student scoring matrix (missing values are filled)

        :param q_m --> numpy.ndarray
            the Q matrix

        :param weight --> numpy.ndarray
            the weight matrix

        :param rank --> int
            the rank

        :param gamma --> float
        """
        self.train_data = train_data
        self.train_fill = train_fill
        self.Q = q_m
        self.W = weight
        self.rank = rank
        self.gamma = gamma 
    

    def train(self, step_search, max_iter=500, cri=5):
        """
        FUNCTION: update solutions for all parameters using Projected Gradient Method

        Inputs:
        -------
        :param step_search --> str:
            the step-size search method (Armijo or Lipschitz)
        
        :param max_iter --> float
            the maximum number of iterations
        
        :param cri --> float
            the difference between the value of the objective function
        
        Outputs:
        -------
        :return b --> numpy.ndarray
            the exercise-knowledge association matrix

        :return u --> numpy.ndarray
            the student feature matrix

        :return e --> numpy.ndarray
            the exercise feature matrix
    
        :return v --> numpy.ndarray
            the knowledge concept feature matrix
    
        :return m --> numpy.ndarray
            the exercise difficulty vector
        """

        if step_search == 'armijo':
            # fitting data using the Armijo rule for the step-size searching
            b, u, e, v, m = armijo.fit_data(self.train_data, self.train_fill, self.Q, self.W, self.rank, self.gamma, max_iter, cri)
        elif step_search == 'lipschitz':
            # fitting data using the Lipschitz constant for the step-size searching
            b, u, e, v, m = lipschitz.fit_data(self.train_data, self.train_fill, self.Q, self.W, self.rank, self.gamma, max_iter, cri)
        else:
            warnings.warn("the step-size searching method does not exist.")
            exit()
        return b, u, e, v, m