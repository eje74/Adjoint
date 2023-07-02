import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def run_system(rhs, tau):
    system_size = 21
    number_of_iterations = 2000
    lhs = 1
    rho0 = (lhs + rhs)/2

    # bulk
    f0 = (2/3)*rho0*np.ones(system_size) # c =  0, w = 2/3
    f1 = (1/6)*rho0*np.ones(system_size) # c =  1, w = 1/6
    f2 = (1/6)*rho0*np.ones(system_size) # c = -1, w = 1/6

    # lhs
    f0[0] = 2*lhs/3
    f1[0] =   lhs/6
    f2[0] =   lhs/6

    # lhs
    f0[-1] = 2*rhs/3
    f1[-1] =   rhs/6
    f2[-1] =   rhs/6

    for itr in np.arange(number_of_iterations):
        # Density
        phi = f0 + f1 + f2

        # collision and propagation
        f0 = f0 - 1/tau*(f0 - 2/3*phi)
        f1[1:] = f1[:-1] - 1/tau*(f1[:-1]-1/6*phi[:-1])
        f2[:-1] = f2[1:] - 1/tau*(f2[1:]-1/6*phi[1:])

        # boundary conditions
        f1[0] = 1/3*lhs - f2[0]
        f2[-1] = 1/3*rhs - f1[-1]

    return phi, (tau-0.5)/tau*(f1[0]-f2[0]) 


def dg_dphi(pos):
    val = [1, 1, 1]
    ind = [(0, pos), (1, pos), (2, pos)]
    return val, ind


def dg_dj(pos):
    val = [1, -1]
    ind = [(1, pos), (2, pos)]
    return val, ind


def dg_collision(alpha, pos, tau):
    w = (2/3, 1/6, 1/6)
    val = [1-1/tau]
    ind = [(alpha, pos)]
    for v, i in zip(*dg_dphi(pos)):
        val.append(w[alpha]*v/tau)
        ind.append(i)
    return val, ind


def dg_dl(alpha, pos, tau):
    c = [np.array(x) for x in [[0], [1], [-1]]]
    val = [1]
    ind = [(alpha, pos)]
    for v, i in zip(*dg_collision(alpha, pos-c[alpha], tau)):
        val.append(-v)
        ind.append(i)
    return val, ind


def dg_dl_boundary(alpha, pos):
    alpha_hat = [0, 2, 1]
    val = [1, 1]
    ind = [(alpha, pos), (alpha_hat[alpha], pos)]
    return val, ind


def dg_dH(alpha, pos, tau, flux, flux_target):
    val = []
    ind = []
    a = 2*(flux-flux_target)*(tau-0.5)/tau
    for v, i in zip(*dg_dj(pos)):
        val.append(a*v)
        ind.append(i)
    return val, ind


def sub2ind(alpha, pos):
    return alpha + 3*pos[0]
    

def adjoint_method(tau, phi, j, j_target):
#tau = tau_input
    # Setup system size
    N = 3*np.prod(phi.size)

    M = lil_matrix((N, N))
    y = np.zeros((N,))

    for x in np.arange(1, 20):
        for alpha in np.arange(3):
            pos = np.array([x], dtype=int)
            n = sub2ind(alpha, pos)
            for v, i in zip(*dg_dl(alpha, pos, tau)):
                m = sub2ind(*i)
                M[m, n] += v

    # Add boundary conditions
    pos = np.array([0], dtype=int)
    for v, i in zip(*dg_dH(0, pos, tau, j, j_target)):
        m = sub2ind(*i)
        y[m] -= v
    for alpha in np.arange(3):
        n = sub2ind(alpha, pos)
        if alpha == 1:
            val, ind = dg_dl_boundary(alpha, pos)
        else:
            val, ind = dg_dl(alpha, pos, tau)
        for v, i in zip(val, ind):
            m = sub2ind(*i)
            M[m, n] += v
        
    pos = np.array([20], dtype=int)
    for alpha in np.arange(3):
        n = sub2ind(alpha, pos)
        if alpha == 2:
            val, ind = dg_dl_boundary(alpha, pos)
        else:
            val, ind = dg_dl(alpha, pos, tau)
        for v, i in zip(val, ind):
            m = sub2ind(*i)
            M[m, n] += v

    M = M.tocsr()
    lambda_l = spsolve(M, y) # Lagrange multiplicator

    # l_2[20] = f2[-1] + f1[-1] - 1/3*rhs
    n = 2+3*20
    d_l_rhs = -1/3 
    return d_l_rhs*lambda_l[n]


tau_input = 1.2
j_max = (tau_input - 0.5)/3/20
j_target = 0.5*j_max
N = 10

dH_adjoint = np.zeros(N)
dH_sim = np.zeros(N)
rhs_x = np.zeros(N)

for n, rhs in enumerate(np.linspace(0.1, 0.9, N)):
    rhs_x[n] = rhs
    phi, j = run_system(rhs, tau_input)

    dH_drhs = adjoint_method(tau_input, phi, j, j_target)
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





