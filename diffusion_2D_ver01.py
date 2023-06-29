import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve




#########################################################################
#                          Simulations                                  #
#########################################################################
def circ_shift(valin, c_input):
    valout = valin
    for d, c in enumerate(c_input):
        valout = np.roll(valout, c, axis=d)
    return valout

def propagate(f, c_input):
    for q, c in enumerate(c_input):
        f[..., q] = circ_shift(f[..., q], c)
    return f

def run_system(rhs, tau, w, c, alpha_rev):
    system_size = (21, 21)
    number_of_iterations = 2000
    lhs = 1
    rho0 = (lhs + rhs)/2

    g = np.zeros(system_size + (w.size,))
    # bulk
    g[1:-1, ...] = w*rho0 
    # lhs
    g[0, ...] = w*lhs
    # lhs
    g[-1, ...] = w*rhs

    # Main loop
    for itr in np.arange(number_of_iterations):
        # Density
        phi = np.sum(g, axis=-1)
        # collision
        g_tmp = g - 1/tau*(g - phi.reshape(phi.shape + (1,))*w)
        # propagation
        g = propagate(g_tmp, c)
        # boundary condition
        # -- left hand side
        for alpha in [1, 5, 7]:
            g[0, :, alpha] = 2*w[alpha]*lhs - g[0, :, alpha_rev[alpha]]
        # -- right hand side
        for alpha in [2, 6, 8]:
            g[-1, :, alpha] = 2*w[alpha]*rhs - g[-1, :, alpha_rev[alpha]]

    # Calculate first moment of the distribution (at the left hand side)
    Mi = np.sum(g.reshape(g.shape + (1,))[0,...]*c, axis=-2)    

    return phi, (tau-0.5)/tau*np.sum(Mi[:,0])/system_size[1] 


#########################################################################
#                            Adjoint                                    #
#########################################################################
def dg_dphi(pos, w_list): # OK
    for alpha in np.arange(w_list.size):
        yield 1, (alpha, pos)

def dg_dj(pos, c_list): # OK
    for alpha, c in enumerate(c_list):
        yield c[0], (alpha, pos)

def dg_collision(alpha, pos, tau, w_list): # OK
    yield 1-1/tau, (alpha, pos) 
    for v, i in dg_dphi(pos, w_list):
        yield w_list[alpha]*v/tau, i


def dg_dl(alpha, pos, tau, w_list, c_list): # OK
    yield 1, (alpha, pos) 
    for v, i in dg_collision(alpha, pos-c_list[alpha], tau, w_list):
        yield -v, i


def dg_dl_boundary(alpha, pos, alpha_hat): # OK
    yield 1, (alpha, pos)
    yield 1, (alpha_hat[alpha], pos)


def dg_dH(pos, tau, flux, flux_target, c_list, system_size): # OK
    a = 2*(flux-flux_target)*(tau-0.5)/tau/system_size[1]
    for v, i in dg_dj(pos, c_list):
        yield a*v, i


def sub2ind(alpha, sub, system_size, w):
    pos = np.mod(sub, system_size)
    return w.size*int(pos[0]*system_size[1] + pos[1]) + alpha


def adjoint_method(tau, phi, j, j_target, w_list, c_list, alpha_hat):
    # Setup system size
    system_size = phi.shape
    N = 9*np.prod(system_size)

    M = lil_matrix((N, N))
    b = np.zeros((N,))
    drhs_dl = np.zeros((N,)) 

    for x in np.arange(1, 20):
        for y in np.arange(21):
            pos = np.array([x, y], dtype=int)
            for alpha in np.arange(len(w_list)):
                n = sub2ind(alpha, pos, system_size, w_list)
                for v, i in dg_dl(alpha, pos, tau, w_list, c_list):
                    m = sub2ind(*i, system_size, w_list)
                    M[m, n] += v


    # Add boundary conditions
    # -- left hand side
    for y in np.arange(21):
        pos = np.array([0, y], dtype=int)
        for v, i in dg_dH(pos, tau, j, j_target, c_list, system_size):
            m = sub2ind(*i, system_size, w_list)
            b[m] -= v
        for alpha in np.arange(len(w_list)):
            n = sub2ind(alpha, pos, system_size, w_list)
            if alpha in [1, 5, 7]:
                for v, i in  dg_dl_boundary(alpha, pos, alpha_hat):
                    m = sub2ind(*i, system_size, w_list)
                    M[m, n] += v
            else:
                for v, i in dg_dl(alpha, pos, tau, w_list, c_list):
                    m = sub2ind(*i, system_size, w_list)
                    M[m, n] += v
    # -- right hand side.
    for y in np.arange(21):
        pos = np.array([20, y], dtype=int)
        for alpha in np.arange(len(w_list)):
            n = sub2ind(alpha, pos, system_size, w_list)
            if alpha in [2, 6, 8]:
                drhs_dl[n] += -2*w_list[alpha]
                for v, i in dg_dl_boundary(alpha, pos, alpha_hat):
                    m = sub2ind(*i, system_size, w_list)
                    M[m, n] += v
            else:
                for v, i in dg_dl(alpha, pos, tau, w_list, c_list):
                    m = sub2ind(*i, system_size, w_list)
                    M[m, n] += v

    M = M.tocsr()
    lambda_l = spsolve(M, b) # Lagrange multiplicator


    return drhs_dl.dot(lambda_l)

if __name__ == "__main__":
    # Latiice
    c_list = [np.array(c, dtype=int) for c in [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, -1), (1, -1), (-1, 1)
        ]]
    w_list = np.array((4/9,) + 4*(1/9,) + 4*(1/36,))
    alpha_reverse = np.array( [0, 2, 1, 4, 3, 6, 5, 8, 7])

    # Input
    tau_input = 1.2
    j_max = (tau_input - 0.5)/3/20
    j_target = 0.5*j_max
    N = 10

    # Containers for plots
    dH_adjoint = np.zeros(N)
    dH_sim = np.zeros(N)
    rhs_x = np.zeros(N)


    for n, rhs in enumerate(np.linspace(0.1, 0.9, N)):
        rhs_x[n] = rhs

        phi, j = run_system(rhs, tau_input, w_list, c_list, alpha_reverse)

        dH_drhs = adjoint_method(tau_input, phi, j, j_target, w_list, c_list, alpha_reverse)


        dH_adjoint[n] = dH_drhs

        # Numerical calculation of the variation
        H0 = (j-j_target)**2
        dx = 0.001
        phi1, j1 = run_system(rhs + dx, tau_input, w_list, c_list, alpha_reverse)
        H1 = (j1-j_target)**2
    #    print((H1 - H0)/dx)
        dH_sim[n] = (H1-H0)/dx


    plt.figure()
    plt.plot(rhs_x, dH_adjoint, '-', label="adjoint")
    plt.plot(rhs_x, dH_sim, '--', label="simulation")
    plt.legend()
    plt.show()





