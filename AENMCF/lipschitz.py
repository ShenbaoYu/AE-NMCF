"""
update solutions for all parameters of AE-NMCF based on Lipschitz constant for the step-size searching.
"""


import numpy as np
import math
from scipy import integrate
from scipy import linalg




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
    # shrink the value for avoiding math.domain error 
    phi = np.where(phi < 1, phi, 0.99999)
    phi = np.where(phi > 0, phi, 1e-5)   

    ll = sum(sum(weight * train_fill * log_fun(phi))) \
       + sum(sum(weight * (1-train_fill) * log_fun(1-phi)))

    return ll


def fit_data(train_data, train_fill, q_m, weight, rank, gamma, max_iter, cri):
    """
    FUNCTION: update solutions using "Projected Gradient Method"
              step-size search: Lipschitz constant

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
    
    convergence = False
    i = 0
    # print("Iteration %d = %s" % (i, obj_old))

    # calculate the lipschitz constant for B, U, E, V, M
    l_b, l_u, l_e, l_v, l_m = cal_lipschitz_constant(train_data, q_m, weight, gamma, B, U, E, V, M, Lp=1)

    while (not convergence) and (i < max_iter):
        
        # --- 1. update all parameters ---
        # 1.1 update B in a block fashion
        B_new = update_B(train_data, q_m, B, U, E, V, M, gamma, l_b)
        # 1.2 update U in a block fashion
        U_new = update_U(train_data, train_fill, weight, B_new, U, E, V, M, l_u)
        # 1.3 update E
        E_new = update_E(train_fill, q_m, weight, B_new, U_new, E, V, l_e)
        # 1.4 update V
        V_new = update_V(train_data, q_m, B_new, U_new, E_new, V, M, l_v)
        # 1.5 update M
        M_new = update_M(train_data, B_new, U_new, V_new, M, l_m)

        # --- 2. calculate the new objective function value ---
        obj_new = - likelihood_probit(weight, train_fill, B_new, U_new, V_new, M_new) \
                  + math.pow(np.linalg.norm(weight * (train_fill-np.dot(E_new, U_new)), ord='fro'), 2) \
                  + math.pow(np.linalg.norm(q_m * (B_new-np.dot(E_new, V_new)), ord='fro'), 2) \
                  + 0.5*gamma*np.sum([math.pow(np.linalg.norm(B_new[n,:]), 2) for n in range(B_new.shape[0])])

        # --- 3. is convergent? ---
        convergence = abs(obj_new - obj_old) < cri
        obj_old = obj_new
        # the maxmimum changes of all parameters
        # pc_max = max([abs(B_new - B).max(), abs(U_new - U).max(), abs(E_new - E).max(), abs(V_new - V).max(), abs(M_new - M).max()])
        # convergence = pc_max < 0.01
        
        # --- 4. update all parameters ---
        B, U, E, V, M = B_new.copy(), U_new.copy(), E_new.copy(), V_new.copy(), M_new.copy()

        i += 1
        # print("Iteration %d = %.5f" % (i, pc_max))
        print("Iteration %d = %s" % (i, obj_old))
            

        if i == max_iter:
            print('Maximum iterations reached.')

    return B, U, E, V, M


def cal_lipschitz_constant(train_data, q_m, weight, gamma, B, U, E, V, M, Lp=None):
    """
    Lp: the scalar Lipschitz constant of the inverse probit funciton, which equals 1[1].
    [1]: Lan, A. S., Waters, A. E., Studer, C., & Baraniuk, R. G. (2014).
         Sparse factor analysis for learning and content analytics.
         Journal of Machine Learning Research, 15(57), 1959-2008.
    """

    exe_num, stu_num = train_data.shape

    # initialization
    l_b = np.zeros(shape=(exe_num, 1), dtype=float)
    l_u = np.zeros(shape=(stu_num, 1), dtype=float)
    l_e = 0
    l_v = 0
    l_m = np.zeros(shape=(exe_num, 1), dtype=float)

    # --- Lipschitz constant for B ---
    for n in range(exe_num):
        C_n = U.T.dot(V)
        mis_index = np.where(np.isnan(train_data[n]))[0].tolist()  # the missing index in n-th block (row) of the training data
        obs_index = [_ for _ in range(stu_num) if _ not in mis_index]
        C_n = C_n[obs_index,:]
        l_b[n][0] = Lp*math.pow(linalg.norm(C_n,2),2) + 2*np.linalg.norm(q_m[n]) + gamma

    # --- Lipschitz constant for U ---
    for m in range(stu_num):
        C_m = V.dot(B.T)
        mis_index = np.where(np.isnan(train_data[:,m]))[0].tolist()  # the missing index in m-th block (column) of the training data
        obs_index = [_ for _ in range(exe_num) if _ not in mis_index]
        C_m = C_m[:, obs_index]
        l_u[m][0] = Lp*math.pow(linalg.norm(C_m,2),2) + 2*math.pow(np.linalg.norm(E,2),2)*np.linalg.norm(weight[:,m])

    # --- Lipschitz constant for E ---
    l_e = 2*math.pow(np.linalg.norm(U,2),2)*np.linalg.norm(weight,2) \
        + 2*math.pow(np.linalg.norm(V,2),2)*np.linalg.norm(q_m,2)

    # --- Lipschitz constant for V ---
    C = 0
    for n in range(exe_num):
        for m in range(stu_num):
            if not np.isnan(train_data[n,m]):
                C += math.pow(2*train_data[n,m]-1,2) \
                * math.pow(np.linalg.norm(U[:,m]),2) \
                * math.pow(np.linalg.norm(B[n,:]),2)
    
    l_v = Lp*C + 2*math.pow(np.linalg.norm(E,2),2)*np.linalg.norm(q_m,2) 

    # --- Lipschitz constant for M ---
    for n in range(exe_num):
        l_m[n][0] = Lp * sum([math.pow(2*x-1,2) for x in train_data[n,:] if not np.isnan(x)])

    return l_b, l_u, l_e, l_v, l_m


def update_B(train_data, q_m, B, U, E, V, M, gamma, l_b):
    """
    FUNCTION: calculate the B value in the nexit iteration
    """
    # --- update B via a block coordinate descent approach ---
    _B = B.copy()
    block_num = _B.shape[0]
    for n in range(block_num):
        # update the n-th block
        # 1. calculate the gradient of the n-th block (row) of B
        grad_B_n = cal_grad_B_n(train_data, n, q_m, _B, U, E, V, M, gamma)
        if np.isinf(grad_B_n).all():
            continue  # do not update this block
        # 2. calculate the new B_n value
        B_n_new = np.maximum(_B[n,:] - grad_B_n/l_b[n][0], 0)  # step-size equals 1/L
        # 3. replace the n-th block of B with the B_n_new
        _B[n,:] = B_n_new

    return _B


def cal_grad_B_n(train_data, n, q_m, B, U, E, V, M, gamma):
    st_num = train_data.shape[1]
    grad = 0

    kn_num = B.shape[1]

    for m in range(st_num):
        if not np.isnan(train_data[n, m]):
            delta = np.dot(B[n,:], np.dot(V.T, U[:,m])) + M[n,0]
            phi = integrate.quad(normal, -float('inf'), delta)[0]
            try:
                D = normal(delta) / (phi * (1-phi))
            except ZeroDivisionError:
                return float('inf') * np.ones(shape=(kn_num))
            grad += D * (train_data[n,m] - phi) * np.dot(U[:,m].T, V)
        
    grad = -1*grad + 2*(q_m[n,:]*B[n,:] - q_m[n,:]*np.dot(E[n,:], V)) + gamma*B[n,:]

    return grad


def update_U(train_data, train_fill, weight, B, U, E, V, M, l_u):
    """
    FUNCTION: calculate the U value in the nexit iteration
    """

    # --- update U via a block coordinate descent approach ---
    _U = U.copy()
    block_num = _U.shape[1]
    for m in range(block_num):
        # update the n-th block
        # 1. calculate the gradient of the m-th block (column) of U
        grad_U_m = cal_grad_U_m(train_data, train_fill, m, weight, B, _U, E, V, M)
        if np.isinf(grad_U_m).all():
            continue  # do not update this block
        # 2. calculate the new U_m value
        U_m_new = np.maximum(_U[:,m] - grad_U_m/l_u[m][0], 0)  # step-size equals 1/L
        # 3. replace the m-th block (column) of U with the U_m_new
        _U[:,m] = U_m_new

    return _U


def cal_grad_U_m(train_data, train_fill, m, weight, B, U, E, V, M):
    ex_num = train_data.shape[0]
    grad = 0

    latent_num = U.shape[0]

    for n in range(ex_num):
        if not np.isnan(train_data[n, m]):
            delta = np.dot(B[n,:], np.dot(V.T, U[:,m])) + M[n,0]
            phi = integrate.quad(normal, -float('inf'), delta)[0]
            try:
                D = normal(delta) / (phi * (1-phi))
            except ZeroDivisionError:
                return float('inf') * np.ones(shape=(latent_num))
            grad += D * (train_data[n,m] - phi) * np.dot(V, B[n,:].T)
    
    grad = -1*grad + 2*np.dot(E.T, (weight[:,m]*np.dot(E, U[:,m]) - weight[:,m]*train_fill[:,m]))

    return grad


def update_E(train_fill, q_m, weight, B, U, E, V, l_e):
    """
    FUNCTION: calculate the E value in the nexit iteration
    """
    _E = E.copy()
    # 1. calculate the gradient of E
    grad_E = 2 * np.dot((weight*(np.dot(_E, U)) - weight*train_fill), U.T) \
           + 2 * np.dot((q_m*(np.dot(_E, V)) - q_m*B), V.T)
    # 2. update E
    E_new = np.maximum(_E - grad_E/l_e, 0)
    # 3. replace E with E_new
    _E = E_new

    return _E


def update_V(train_data, q_m, B, U, E, V, M, l_v):
    """
    FUNCTION: calculate the V value in the nexit iteration
    """
    
    _V = V.copy()
    # 1. calculate the gradient of V
    grad_V = cal_grad_V(train_data, q_m, B, U, E, _V, M)
    if np.isinf(grad_V).all():
        return _V  # do not update
    # 2. update V
    V_new = np.maximum(_V -  grad_V/l_v, 0)
    # 3. replace V with V_new
    _V = V_new

    return _V


def cal_grad_V(train_data, q_m, B, U, E, V, M):
    ex_num, st_num = train_data.shape
    grad = 0

    latent_num, kn_num = V.shape

    for n in range(ex_num):
        for m in range(st_num):
            if not np.isnan(train_data[n, m]):
                delta = np.dot(B[n,:], np.dot(V.T, U[:,m])) + M[n,0]
                phi = integrate.quad(normal, -float('inf'), delta)[0]
                try:
                    D = normal(delta) / (phi * (1-phi))
                except ZeroDivisionError:
                    return float('inf') * np.ones(shape=(latent_num, kn_num))
                grad += D * (train_data[n,m] - phi) * np.dot(U[:,m].reshape(U.shape[0],1), B[n,:].reshape(1, B.shape[1]))
    
    grad = -1*grad + 2*np.dot(E.T, (q_m*(np.dot(E, V)) - q_m*B))

    return grad

def update_M(train_data, B, U, V, M, l_m):
    """
    FUNCTION: calculate the M value in the nexit iteration
    """
    
    _M = M.copy()
    ex_num = _M.shape[0]

    # --- update the M_n one-by-one
    for n in range(ex_num):
        # 1. calculate the n-th gradient of M
        grad_M_n = cal_grad_M_n(train_data, n, B, U, V, _M)
        if grad_M_n == float('inf'):
            continue # do not update this block
        # 2. calculate the new M_n value
        M_n_new = _M[n,0] - grad_M_n/l_m[n][0]
        # 3. replace the n-th block of M with the M_n_new
        _M[n,0] = M_n_new

    return _M


def cal_grad_M_n(train_data, n, B, U, V, M):
    st_num = train_data.shape[1]
    grad = 0

    for m in range(st_num):
        if not np.isnan(train_data[n, m]):
            delta = np.dot(B[n,:], np.dot(V.T, U[:,m])) + M[n,0]
            phi = integrate.quad(normal, -float('inf'), delta)[0]
            try:
                D = normal(delta) / (phi * (1-phi))
            except ZeroDivisionError:
                return float('inf')
            grad += D * (train_data[n,m] - phi)

    return -1*grad