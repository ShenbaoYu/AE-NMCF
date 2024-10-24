"""
update solutions for all parameters of AE-NMCF based on Armijo rule for the step-size searching.
"""

import numpy as np
import math
import time
from scipy import integrate


# the initial step-size
BETA_B = 1e-4  # for B
BETA_U = 1e-3  # for U
BETA_E = 1e-4  # for E
BETA_V = 1e-4  # for V
BETA_M = 5e-4  # for M

RHO = 0.01  # the value of the line search based on Armijo rule




def normal(x):
    return 1/math.sqrt(2*math.pi) * math.exp(-math.pow(x,2)/2)


def likelihood_probit(weight, train_fill, B, U, V, M):
    """
    FUNCTION: Calculate the likelihood value (probit variant)
    """

    phi_fun = np.frompyfunc(lambda x:integrate.quad(normal, -float('inf'), x)[0], 1, 1)
    # phi_fun = np.frompyfunc(lambda x:norm.cdf(x), 1, 1)
    log_fun = np.frompyfunc(lambda x:math.log(x), 1, 1)

    ex_num, st_num = train_fill.shape
    delta = np.dot(np.dot(B, V.T), U) + np.tile(M, st_num)
    phi = phi_fun(delta)

    ll = sum(sum(weight * train_fill * log_fun(phi))) \
       + sum(sum(weight * (1-train_fill) * log_fun(1-phi)))

    return ll


def fit_data(train_data, train_fill, q_m, weight, rank, gamma, max_iter, cri):
    """
    FUNCTION: update solutions using "Projected Gradient Method"
              step-size search: Armijo Rule

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

    Outputs:
    -------
    :return B --> numpy.ndarray
        the exercise-knowledge association matrix

    :return U --> numpy.ndarray
        the student feature matrix

    :return E --> numpy.ndarray
        the exercise feature matrix
    
    :return V --> numpy.ndarray
        the knowledge concept feature matrix
    
    :return M --> numpy.ndarray
        the exercise difficulty vector
    """

    print("Fitting data...")

    ex_num, st_num = train_data.shape
    kn_num = q_m.shape[1]
    
    # initialize B, U, E, V, and M
    B = np.random.uniform(0, 1, size=(ex_num, kn_num))
    U = np.random.uniform(0, 1, size=(rank, st_num))
    E = np.random.uniform(0, 1, size=(ex_num, rank))
    V = np.random.uniform(0, 1, size=(rank, kn_num))
    M = np.random.uniform(0, 1, size=(ex_num, 1))

    # calculate the objective function value
    obj_old = - likelihood_probit(weight, train_fill, B, U, V, M) \
              + math.pow(np.linalg.norm(weight * (train_fill-np.dot(E, U)), ord='fro'), 2) \
              + math.pow(np.linalg.norm(q_m * (B-np.dot(E, V)), ord='fro'), 2) \
              + 0.5*gamma*np.sum([math.pow(np.linalg.norm(B[n,:]), 2) for n in range(B.shape[0])])

    # ---- iterate over the Projected Gradient Method ----
    # start_time = time.time()

    convergence = False
    i = 0
    print("Iteration %d = %s" % (i, obj_old))

    while (not convergence) and (i < max_iter):
        
        # 1. update all parameters
        # 1.1 update B in a block fashion
        B_new = update_B(train_data, train_fill, q_m, weight, B, U, E, V, M, gamma, BETA_B)
        # 1.2 update U in a block fashion
        U_new = update_U(train_data, train_fill, q_m, weight, B_new, U, E, V, M, gamma, BETA_U)
        # 1.3 update E
        E_new = update_E(train_data, train_fill, q_m, weight, B_new, U_new, E, V, M, gamma, BETA_E)
        # 1.4 update V
        V_new = update_V(train_data, train_fill, q_m, weight, B_new, U_new, E_new, V, M, gamma, BETA_V)
        # 1.5 update M
        M_new = update_M(train_data, train_fill, q_m, weight, B_new, U_new, E_new, V_new, M, gamma, BETA_M)  
        
        # 2. calculate the new objective function value
        obj_new = - likelihood_probit(weight, train_fill, B_new, U_new, V_new, M_new) \
                  + math.pow(np.linalg.norm(weight * (train_fill-np.dot(E_new, U_new)), ord='fro'), 2) \
                  + math.pow(np.linalg.norm(q_m * (B_new-np.dot(E_new, V_new)), ord='fro'), 2) \
                  + 0.5*gamma*np.sum([math.pow(np.linalg.norm(B_new[n,:]), 2) for n in range(B_new.shape[0])])
        
        # 3. update all parameters
        B, U, E, V, M = B_new.copy(), U_new.copy(), E_new.copy(), V_new.copy(), M_new.copy()

        # 4. is convergent?
        convergence = abs(obj_new - obj_old) < cri
        obj_old = obj_new

        i += 1

        print("Iteration %d = %s" % (i, obj_old))

        if i == max_iter:
            print('Maximum iterations reached.')

    # end_time = time.time()
    # gap = end_time - start_time
    # print("Time = ", gap)
    
    return B, U, E, V, M


def update_B(train_data, train_fill, q_m, weight, B, U, E, V, M, gamma, beta_b):
    """
    FUNCTION: calculate the B value in the nexit iteration
    """

    # --- update B via a block coordinate descent approach ---
    block_num = B.shape[0]

    for n in range(block_num):
        # update the n-th block
        # 1. calculate the gradient of the n-th block (row) of B
        grad_B_n = cal_grad_B_n(train_data, n, q_m, B, U, E, V, M, gamma)

        # 2. calculate the objecive function value based on B_n_old
        obj_old = - likelihood_probit(weight, train_fill, B, U, V, M) \
                  + math.pow(np.linalg.norm(weight * (train_fill-np.dot(E, U)), ord='fro'), 2) \
                  + math.pow(np.linalg.norm(q_m * (B-np.dot(E, V)), ord='fro'), 2) \
                  + 0.5*gamma*np.sum([math.pow(np.linalg.norm(B[n,:]), 2) for n in range(B.shape[0])])

        # 3. update the n-th block (row) of B 
        is_continue = True
        _step = beta_b
        while is_continue:
            # calculate the new B_n value
            B_n_new = np.maximum(B[n,:] - _step * grad_B_n, 0)
            _B = B.copy()
            _B[n,:] = B_n_new
            # calculate the new objecive function value based on B_n_new
            obj_new = - likelihood_probit(weight, train_fill, _B, U, V, M) \
                      + math.pow(np.linalg.norm(weight * (train_fill-np.dot(E, U)), ord='fro'), 2) \
                      + math.pow(np.linalg.norm(q_m * (_B-np.dot(E, V)), ord='fro'), 2) \
                      + 0.5*gamma*np.sum([math.pow(np.linalg.norm(_B[n,:]), 2) for n in range(_B.shape[0])])
            # is satisfied Armijo rule?
            if obj_new - obj_old <= RHO * sum(grad_B_n * (B_n_new-B[n,:])):
                is_continue = False
            else:
                _step = beta_b / 2  # shrink the step size
        B[n,:] = B_n_new  # replace the n-th block of B with the B_n_new
    
    return B


def cal_grad_B_n(train_data, n, q_m, B, U, E, V, M, gamma):
    st_num = train_data.shape[1]
    grad = 0

    for m in range(st_num):
        if not np.isnan(train_data[n, m]):
            delta = np.dot(B[n,:], np.dot(V.T, U[:,m])) + M[n,0]
            phi = integrate.quad(normal, -float('inf'), delta)[0]
            D = normal(delta) / (phi * (1-phi))
            grad += D * (train_data[n,m] - phi) * np.dot(U[:,m].T, V)
        
    grad = -1*grad + 2*(q_m[n,:]*B[n,:] - q_m[n,:]*np.dot(E[n,:], V)) + gamma*B[n,:]

    return grad


def update_U(train_data, train_fill, q_m, weight, B, U, E, V, M, gamma, beta_u):
    """
    FUNCTION: calculate the U value in the nexit iteration
    """

    # --- update U via a block coordinate descent approach ---
    block_num = U.shape[1]

    for m in range(block_num):
        # update the n-th block
        # 1. calculate the gradient of the m-th block (column) of U
        grad_U_m = cal_grad_U_m(train_data, train_fill, m, weight, B, U, E, V, M)

        # 2. calculate the objecive function value based on U_m_old
        obj_old = - likelihood_probit(weight, train_fill, B, U, V, M) \
                  + math.pow(np.linalg.norm(weight * (train_fill-np.dot(E, U)), ord='fro'), 2) \
                  + math.pow(np.linalg.norm(q_m * (B-np.dot(E, V)), ord='fro'), 2) \
                  + 0.5*gamma*np.sum([math.pow(np.linalg.norm(B[n,:]), 2) for n in range(B.shape[0])])

        # 3. update the m-th block (column) of U 
        is_continue = True
        _step = beta_u
        while is_continue:
            # calculate the new U_m value
            U_m_new = np.maximum(U[:,m] - _step * grad_U_m, 0)
            _U = U.copy()
            _U[:,m] = U_m_new
            # calculate the new objecive function value based on U_m_new
            obj_new = - likelihood_probit(weight, train_fill, B, _U, V, M) \
                      + math.pow(np.linalg.norm(weight * (train_fill-np.dot(E, _U)), ord='fro'), 2) \
                      + math.pow(np.linalg.norm(q_m * (B-np.dot(E, V)), ord='fro'), 2) \
                      + 0.5*gamma*np.sum([math.pow(np.linalg.norm(B[n,:]), 2) for n in range(B.shape[0])])
            # is satisfied Armijo rule?
            if obj_new - obj_old <= RHO * sum(grad_U_m * (U_m_new-U[:,m])):
                is_continue = False
            else:
                _step = beta_u / 2  # shrink the step size
        U[:,m] = U_m_new  # replace the m-th block of U with the U_m_new

    return U


def cal_grad_U_m(train_data, train_fill, m, weight, B, U, E, V, M):
    ex_num = train_data.shape[0]
    grad = 0

    for n in range(ex_num):
        if not np.isnan(train_data[n, m]):
            delta = np.dot(B[n,:], np.dot(V.T, U[:,m])) + M[n,0]
            phi = integrate.quad(normal, -float('inf'), delta)[0]
            D = normal(delta) / (phi * (1-phi))
            grad += D * (train_data[n,m] - phi) * np.dot(V, B[n,:].T)
    
    grad = -1*grad + 2*np.dot(E.T, (weight[:,m]*np.dot(E, U[:,m]) - weight[:,m]*train_fill[:,m]))

    return grad


def update_E(train_data, train_fill, q_m, weight, B, U, E, V, M, gamma, beta_e):
    """
    FUNCTION: calculate the E value in the nexit iteration
    """

    # 1. calculate the gradient of E
    grad_E = 2 * np.dot((weight*(np.dot(E, U)) - weight*train_fill), U.T) \
           + 2 * np.dot((q_m*(np.dot(E, V)) - q_m*B), V.T)

    # 2. calculate the objecive function value based on old E
    obj_old = - likelihood_probit(weight, train_fill, B, U, V, M) \
              + math.pow(np.linalg.norm(weight * (train_fill-np.dot(E, U)), ord='fro'), 2) \
              + math.pow(np.linalg.norm(q_m * (B-np.dot(E, V)), ord='fro'), 2) \
              + 0.5*gamma*np.sum([math.pow(np.linalg.norm(B[n,:]), 2) for n in range(B.shape[0])])

    # 3. update E
    is_continue = True
    _step = beta_e
    while is_continue:
        # calculate the new E value
        E_new = np.maximum(E - _step * grad_E, 0)
        # calculate the objecive function value based on E_new
        obj_new = - likelihood_probit(weight, train_fill, B, U, V, M) \
                  + math.pow(np.linalg.norm(weight * (train_fill-np.dot(E_new, U)), ord='fro'), 2) \
                  + math.pow(np.linalg.norm(q_m * (B-np.dot(E_new, V)), ord='fro'), 2) \
                  + 0.5*gamma*np.sum([math.pow(np.linalg.norm(B[n,:]), 2) for n in range(B.shape[0])])
        # is satisfied Armijo rule?
        if obj_new - obj_old <= RHO * sum(sum(grad_E * (E_new-E))):
            is_continue = False
        else:
            _step = beta_e / 2  # shrink the step size    
    E = E_new

    return E


def update_V(train_data, train_fill, q_m, weight, B, U, E, V, M, gamma, beta_v):
    """
    FUNCTION: calculate the V value in the nexit iteration
    """

    # 1. calculate the gradient of V
    grad_V = cal_grad_V(train_data, q_m, B, U, E, V, M)

    # 2. calculate the objecive function value based on old V
    obj_old = - likelihood_probit(weight, train_fill, B, U, V, M) \
              + math.pow(np.linalg.norm(weight * (train_fill-np.dot(E, U)), ord='fro'), 2) \
              + math.pow(np.linalg.norm(q_m * (B-np.dot(E, V)), ord='fro'), 2) \
              + 0.5*gamma*np.sum([math.pow(np.linalg.norm(B[n,:]), 2) for n in range(B.shape[0])])

    # 3. update V
    is_continue = True
    _step = beta_v
    while is_continue:
        # calculate the new E value
        V_new = np.maximum(V - _step * grad_V, 0)
        # calculate the objecive function value based on V_new
        obj_new = - likelihood_probit(weight, train_fill, B, U, V_new, M) \
                  + math.pow(np.linalg.norm(weight * (train_fill-np.dot(E, U)), ord='fro'), 2) \
                  + math.pow(np.linalg.norm(q_m * (B-np.dot(E, V_new)), ord='fro'), 2) \
                  + 0.5*gamma*np.sum([math.pow(np.linalg.norm(B[n,:]), 2) for n in range(B.shape[0])])
        # is satisfied Armijo rule?
        if obj_new - obj_old <= RHO * sum(sum(grad_V * (V_new-V))):
            is_continue = False
        else:
            _step = beta_v / 2  # shrink the step size   
    V = V_new

    return V


def cal_grad_V(train_data, q_m, B, U, E, V, M):
    ex_num, st_num = train_data.shape
    grad = 0

    for n in range(ex_num):
        for m in range(st_num):
            if not np.isnan(train_data[n, m]):
                delta = np.dot(B[n,:], np.dot(V.T, U[:,m])) + M[n,0]
                phi = integrate.quad(normal, -float('inf'), delta)[0]
                D = normal(delta) / (phi * (1-phi))
                grad += D * (train_data[n,m] - phi) * np.dot(U[:,m].reshape(U.shape[0],1), B[n,:].reshape(1, B.shape[1]))
    
    grad = -1*grad + 2*np.dot(E.T, (q_m*(np.dot(E, V)) - q_m*B))

    return grad


def update_M(train_data, train_fill, q_m, weight, B, U, E, V, M, gamma, beta_m):
    """
    FUNCTION: calculate the M value in the nexit iteration
    """

    ex_num = M.shape[0]
    
    # --- update the M_n one-by-one
    for n in range(ex_num):
        # 1. calculate the n-th gradient of M
        grad_M_n = cal_grad_M_n(train_data, n, B, U, V, M)

        # 2. calculate the objecive function value based on M_n_old
        obj_old = - likelihood_probit(weight, train_fill, B, U, V, M) \
                  + math.pow(np.linalg.norm(weight * (train_fill-np.dot(E, U)), ord='fro'), 2) \
                  + math.pow(np.linalg.norm(q_m * (B-np.dot(E, V)), ord='fro'), 2) \
                  + 0.5*gamma*np.sum([math.pow(np.linalg.norm(B[n,:]), 2) for n in range(B.shape[0])])
        
        # 3. update the n-th block of M
        is_continue = True
        _step = beta_m
        while is_continue:
            # calculate the new M_n value
            M_n_new = M[n,0] - _step * grad_M_n
            _M = M.copy()
            _M[n,0] = M_n_new
            # calculate the new objecive function value based on M_n_new
            obj_new = - likelihood_probit(weight, train_fill, B, U, V, _M) \
                      + math.pow(np.linalg.norm(weight * (train_fill-np.dot(E, U)), ord='fro'), 2) \
                      + math.pow(np.linalg.norm(q_m * (B-np.dot(E, V)), ord='fro'), 2) \
                      + 0.5*gamma*np.sum([math.pow(np.linalg.norm(B[n,:]), 2) for n in range(B.shape[0])])
            # is satisfied Armijo rule?
            if obj_new - obj_old <= RHO * grad_M_n * (M_n_new-M[n,0]):
                is_continue = False
            else:
                _step = beta_m / 2  # shrink the step size
        M[n,0] = M_n_new  # replace the n-th block of M with the M_n_new

    return M


def cal_grad_M_n(train_data, n, B, U, V, M):
    st_num = train_data.shape[1]
    grad = 0

    for m in range(st_num):
        if not np.isnan(train_data[n, m]):
            delta = np.dot(B[n,:], np.dot(V.T, U[:,m])) + M[n,0]
            phi = integrate.quad(normal, -float('inf'), delta)[0]
            D = normal(delta) / (phi * (1-phi))
            grad += D * (train_data[n,m] - phi)

    return -1*grad