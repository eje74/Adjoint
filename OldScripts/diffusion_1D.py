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

def adjoint_method(tau, phi, j, j_target):
#tau = tau_input
    # Setup system size
    N = 3*np.prod(phi.size)

    M = lil_matrix((N, N))
    y = np.zeros((N,))

    for x in np.arange(21):
        n = 3*x
        # alpha = 0
        M[n, n] += 1/(3*tau)
        M[n+1, n] += -2/(3*tau)
        M[n+2, n] += -2/(3*tau)
        # alpha = 1
        if x == 0:
            M[n+1, n+1] += 1
            M[n+2, n+1] += 1
            y[n+1] -= 2*(j-j_target)*(tau-0.5)/tau 
            y[n+2] -= -2*(j-j_target)*(tau-0.5)/tau 
        else:
            M[n+1, n+1] += 1
            m = 3*(x-1)
            M[m, n+1] += -1/(6*tau)
            M[m+1, n+1] += 5/(6*tau) - 1
            M[m+2, n+1] += -1/(6*tau)
        # alpha = 2
        if x == 20:
            M[n+1, n+2] += 1
            M[n+2, n+2] += 1
        else:
            M[n+2, n+2] += 1
            m = 3*(x+1)
            M[m, n+2] += -1/(6*tau)
            M[m+1, n+2] += -1/(6*tau)
            M[m+2, n+2] += 5/(6*tau) - 1

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

    




