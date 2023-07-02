import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def circ_shift(valin, c_input):
    valout = valin
    for d, c in enumerate(c_input):
        valout = np.roll(valout, c, axis=d)
    return valout

def propagate(f, c_input):
    for q, c in enumerate(c_input):
        f[..., q] = circ_shift(f[..., q], c)
    return f


def run_system(rhs, tau):
    system_size = (21,)
    number_of_iterations = 2000
    lhs = 1
    rho0 = (lhs + rhs)/2

    w = np.array([2/3, 1/6, 1/6])
    c = [np.array(x, dtype=int) for x in [(0,), (1,), (-1,)]]

    g = np.zeros(system_size + (3,))

    # bulk
    g[1:-1,:] = w*rho0 

    # lhs
    g[0,:] = w*lhs

    # lhs
    g[-1,:] = w*rhs


    for itr in np.arange(number_of_iterations):
        # Density
        phi = np.sum(g, axis=-1)
        # collision
        g_tmp = g - 1/tau*(g - phi.reshape(phi.shape + (1,))*w)
        # propagation
        g = propagate(g_tmp, c)
        # boundary condition
        g[0, 1] = 1/3*lhs - g[0, 2]
        g[-1, 2] = 1/3*rhs - g[-1, 1]

    return phi, (tau-0.5)/tau*(g[0, 1]-g[0, 2]) 



def dg_dphi(pos, number_of_directions):
    # val = [1, 1, 1]
    # ind = [(0, pos), (1, pos), (2, pos)]
    # return val, ind
    for alpha in np.arange(number_of_directions):
        yield 1, (alpha, pos)


def dg_dj(pos, c_list):
    for alpha, c in enumerate(c_list):
        yield c[0], (alpha, pos)


def dg_collision(alpha, pos, tau, w_list):
    yield 1-1/tau, (alpha, pos) 
    for v, i in dg_dphi(pos, w_list.size):
        yield w[alpha]*v/tau, i


def dg_dl(alpha, pos, tau, w_list, c_list):
    yield 1, (alpha, pos) 
    for v, i in dg_collision(alpha, pos-c[alpha], tau, w_list):
        yield -v, i


def dg_dl_boundary(alpha, pos, alpha_hat):
    yield 1, (alpha, pos)
    yield 1, (alpha_hat[alpha], pos)


def dg_dH(alpha, pos, tau, flux, flux_target, c_list):
    a = 2*(flux-flux_target)*(tau-0.5)/tau
    for v, i in dg_dj(pos, c_list):
        yield a*v, i


def sub2ind(alpha, pos):
    return alpha + 3*pos[0]
    

def adjoint_method(tau, phi, j, j_target, w_list, c_list, alpha_hat):
#tau = tau_input
    # Setup system size
    N = 3*np.prod(phi.size)

    M = lil_matrix((N, N))
    y = np.zeros((N,))

    for x in np.arange(1, 20):
        for alpha in np.arange(3):
            pos = np.array([x], dtype=int)
            n = sub2ind(alpha, pos)
            for v, i in dg_dl(alpha, pos, tau, w_list, c_list):
                m = sub2ind(*i)
                M[m, n] += v

    # Add boundary conditions
    pos = np.array([0], dtype=int)
    for v, i in dg_dH(0, pos, tau, j, j_target, c):
        m = sub2ind(*i)
        y[m] -= v
    for alpha in np.arange(3):
        n = sub2ind(alpha, pos)
        if alpha == 1:
            for v, i in  dg_dl_boundary(alpha, pos, alpha_hat):
                m = sub2ind(*i)
                M[m, n] += v
        else:
            for v, i in dg_dl(alpha, pos, tau, w_list, c_list):
                m = sub2ind(*i)
                M[m, n] += v
        
    pos = np.array([20], dtype=int)
    for alpha in np.arange(3):
        n = sub2ind(alpha, pos)
        if alpha == 2:
            for v, i in dg_dl_boundary(alpha, pos, alpha_hat):
                m = sub2ind(*i)
                M[m, n] += v
        else:
            for v, i in dg_dl(alpha, pos, tau, w_list, c_list):
                m = sub2ind(*i)
                M[m, n] += v

    M = M.tocsr()
    lambda_l = spsolve(M, y) # Lagrange multiplicator

    # l_2[20] = f2[-1] + f1[-1] - 1/3*rhs
    n = 2+3*20
    d_l_rhs = -1/3 
    return d_l_rhs*lambda_l[n]


w = np.array([2/3, 1/6, 1/6])
c = [np.array(x, dtype=int) for x in [(0,), (1,), (-1,)]]
alpha_reverse = np.array([0, 2, 1], dtype=int)

tau_input = 0.7
j_max = (tau_input - 0.5)/3/20
j_target = 0.5*j_max
N = 10

dH_adjoint = np.zeros(N)
dH_sim = np.zeros(N)
rhs_x = np.zeros(N)

for n, rhs in enumerate(np.linspace(0.1, 0.9, N)):
    rhs_x[n] = rhs
    phi, j = run_system(rhs, tau_input)

    dH_drhs = adjoint_method(tau_input, phi, j, j_target, w, c, alpha_reverse)
    dH_adjoint[n] = dH_drhs

    H0 = (j-j_target)**2
    dx = 0.001
    phi1, j1 = run_system(rhs + dx, tau_input)
    H1 = (j1-j_target)**2
#    print((H1 - H0)/dx)
    dH_sim[n] = (H1-H0)/dx


plt.figure()
plt.plot(rhs_x, dH_adjoint, '-', label="adjoint")
plt.plot(rhs_x, dH_sim, '--', label="simulation")
plt.legend()
plt.show()





